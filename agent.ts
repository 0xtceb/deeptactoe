import { GameEngine, Player, Memory, State } from './GameEngine';
import * as tf from '@tensorflow/tfjs-node';
import { DeepQNetwork } from './dqn';
import { ReplayMemory } from './replayMemory';

class MovingAverager {
    buffer: number[] = [];
    constructor(bufferLength: number) {
      for (let i = 0; i < bufferLength; ++i) {
        this.buffer.push(0);
      }
    }
  
    append(x:number): void {
      this.buffer.shift();
      this.buffer.push(x);
    }
  
    average(): number {
      return this.buffer.reduce((x, prev) => x + prev) / this.buffer.length;
    }
  }
  

export class Agent{
    optimizer = tf.train.adam(0.001);
    onlineNetwork = new DeepQNetwork(true, this.optimizer).model;
    targetNetwork = new DeepQNetwork(false, null).model;
    replayMemory: ReplayMemory;
    gamma: number;
    epsilon: number = 0 ;
    epsilonFinal: number;
    epsilonInit: number;
    decay: number;
    gameCount: number = 0;
    maxGames: number;
    game: GameEngine;
    iaPlayer: Player;
    wonGames: number = 0;
    lostGames: number = 0;
    tieGames: number = 0;
    batchSize: number;
    currentLoss = 0;
    epsilonIncrement: number;
    movingAverage100: MovingAverager = new MovingAverager(100);

    constructor(epsilonInit: number, epsilonFinal: number, gamma: number, decay: number, maxGames: number, memorySize: number, batchSize: number) {
        this.gamma = gamma;
        this.decay = decay;
        this.maxGames = maxGames;
        this.replayMemory = new ReplayMemory(memorySize);
        this.game = new GameEngine();
        this.iaPlayer = this.game.aiPlayer;
        this.batchSize = batchSize;
        this.epsilonFinal = epsilonFinal;
        this.epsilonInit = epsilonInit;
        this.epsilonIncrement = (this.epsilonFinal - this.epsilonInit ) / this.decay;
    }

    randomValidMove(): number {
        const validMoves = [];
        for(let i=0; i < this.game.board.length; i++) {
            if(this.game.board[i] === Player.NAN) {
                validMoves.push(i);
            }
        }
        return validMoves[Math.floor(Math.random() * validMoves.length)];
    }


    invalidStateMove(state: State | State[]): tf.Tensor {
        if(!Array.isArray(state)) {
            state = [state];
        }
        const buffer = tf.buffer([state.length, 9]);

        for(let n=0; n<state.length; n++) {
            state[n].ai.forEach((v, i) => {
                if(v !== 0) {
                    buffer.set(-1e9, n, i);
                } else {
                    buffer.set(0, n, i);
                }
            });
            state[n].ennemy.forEach((v, i) => {
                if(v !== 0) {
                    buffer.set(-1e9, n, i);
                } else {
                    buffer.set(0, n, i);
                }
            });
        }
        return buffer.toTensor();
    }

    invalidMoves(): tf.Tensor {
        const buffer = tf.buffer([9]);
        for(let i=0; i<this.game.board.length; i++) {
            if(this.game.board[i] !== Player.NAN) {
                buffer.set(-1e7, i);
            } else {
                buffer.set(0, i);
            }
        }
        return buffer.toTensor();
    }

    playStep(): Memory {
        let memory: Memory;
        this.epsilon = this.gameCount >= this.decay ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement  * this.gameCount;
        if(Math.random() < this.epsilon) {
            memory = this.game.makeMove(this.randomValidMove());        
        } else {
            const pred = tf.tidy(() => {
                const state = this.game.state(this.game.currentPlayer);
                const stateTensor = this.game.stateToTensor(state);
                const illegalMask = this.invalidMoves();

                return (<tf.Tensor>this.onlineNetwork.predict(stateTensor)).add(illegalMask).argMax(-1).dataSync()[0];
            })
            memory = this.game.makeMove(pred);
        }



        return memory;


    }

    async train(adversary?: tf.Sequential | tf.LayersModel, oldMemory?: ReplayMemory) {
        let tmpMemoryIA: any = null;
        let tmpMemoryAdversary: any = null;
        if(!oldMemory) {
            console.log('===== FILLING UP MEMORY =====');
            while(this.replayMemory.length < this.replayMemory.maxLen) {
                if(this.game.currentPlayer === this.iaPlayer) {
                    tmpMemoryIA = this.playStep();
                    if(tmpMemoryIA.doneFlag) {
                        tmpMemoryIA.reward = tmpMemoryIA.tie ? 0 : 2;
                        this.replayMemory.append(tmpMemoryIA);
                        this.game.reset();
                        this.iaPlayer = this.game.aiPlayer;
                    } else {
                        this.replayMemory.append(tmpMemoryIA);
                    }
                } else {
                    const memory = this.game.makeMove(this.randomValidMove());
                    if(memory.doneFlag) {
                        tmpMemoryIA.reward = memory.tie ? 0 : -2;
                        tmpMemoryIA.doneFlag = true;
                        this.replayMemory.eraseLastMemory(tmpMemoryIA);
                        this.game.reset();
                        this.iaPlayer = this.game.aiPlayer;
                    } else {
                        this.replayMemory.append(memory);
                    }
                }
            }
            tmpMemoryIA = null;
            this.game.reset();
            this.iaPlayer = this.game.aiPlayer;
            console.log('====== MEMORY FILLED =====');
            console.log('Launching training');
        } else {
            this.replayMemory = oldMemory;
        }

        let iter = 0;
     
        while(true) {
            if(this.game.currentPlayer === this.iaPlayer) {
                this.trainOnReplayBatch(this.batchSize, this.optimizer);
                tmpMemoryIA = this.playStep();
                if(tmpMemoryIA.doneFlag) {
                    tmpMemoryIA.reward = tmpMemoryIA.tie ? 0 : 2;
                    tmpMemoryAdversary.reward = tmpMemoryIA? 0: -2;
                    tmpMemoryAdversary.doneFlag = true;
                    this.replayMemory.eraseLastMemory(ReplayMemory.cloneMemory(tmpMemoryAdversary));
                    this.movingAverage100.append(tmpMemoryIA.tie ? 1 : 2);

                    this.replayMemory.append(ReplayMemory.cloneMemory(tmpMemoryIA));
                    this.gameCount++;
                    iter ++;
                    this.wonGames = tmpMemoryIA.tie ? this.wonGames : this.wonGames + 1;
                    this.tieGames = tmpMemoryIA.tie ? this.tieGames + 1 : this.tieGames;
                    tmpMemoryIA = null;
                    console.log(`IA PLAYER ${this.game.numberToPlayer(this.iaPlayer)}`)
                    console.log(`Total Games ${this.gameCount} - Games won ${this.wonGames} - Games lost ${this.lostGames} - Ties ${this.tieGames} - AVG ${this.movingAverage100.average()} - epsilon ${this.epsilon} - loss ${this.currentLoss}`);
                    this.game.displayBoard();
                    this.game.reset();
                    this.iaPlayer = this.game.aiPlayer;
                } else {
                    this.replayMemory.append(ReplayMemory.cloneMemory(tmpMemoryIA));
                }
            } else {

                if(!adversary) {
                    tmpMemoryAdversary = this.game.makeMove(this.randomValidMove());
                } else {
                    const pred = tf.tidy(() => {
                        const state = this.game.state(this.game.currentPlayer);
                        const stateTensor = this.game.stateToTensor(state);
                        const illegalMoves = this.invalidMoves();
                        const qvalues = (<tf.Tensor>adversary.predict(stateTensor)).add(illegalMoves).argMax(-1).dataSync()[0];
                        return qvalues;
                    });
                    tmpMemoryAdversary = this.game.makeMove(pred);
                }
                if(tmpMemoryAdversary.doneFlag) {
                    tmpMemoryIA.reward = tmpMemoryAdversary.tie ? 0 : -2;
                    tmpMemoryIA.doneFlag = true;
                    this.movingAverage100.append(tmpMemoryAdversary.tie ? 2 : -2);
                    this.replayMemory.eraseLastMemory(ReplayMemory.cloneMemory(tmpMemoryIA));
                    const mem = ReplayMemory.cloneMemory(tmpMemoryAdversary);
                    mem.reward = mem.tie ? 0 : 2;
                    this.replayMemory.append(mem);
                    this.gameCount++;
                    iter ++;
                    this.lostGames = tmpMemoryAdversary.tie ? this.lostGames : this.lostGames + 1;
                    this.tieGames = tmpMemoryAdversary.tie ? this.tieGames + 1 : this.tieGames;
                    console.log(`IA PLAYER ${this.game.numberToPlayer(this.iaPlayer)}`)
                    console.log(`Total Games ${this.gameCount} - Games won ${this.wonGames} - Games lost ${this.lostGames} - Ties ${this.tieGames} - AVG ${this.movingAverage100.average()} - epsilon ${this.epsilon} - loss ${this.currentLoss}`);
                    this.game.displayBoard();
                    this.game.reset();
                    this.iaPlayer = this.game.aiPlayer;
                } else {
                    this.replayMemory.append(ReplayMemory.cloneMemory(tmpMemoryAdversary));
                }
            }

            if(this.movingAverage100.average() >= 1.80) {
                this.movingAverage100 = new MovingAverager(100);
                iter = 0;
                this.gameCount = 0;
                this.tieGames = 0;
                this.lostGames = 0;
                this.wonGames = 0;
                await this.onlineNetwork.save('file://./../tictactoe-ui/src/assets');
                break;
            } else {
                if(iter >= 2000) {
                    iter = 0;
                    DeepQNetwork.copyWeights(this.targetNetwork, this.onlineNetwork);
                }
            } 
        }

        console.log('training finished & model saved !');
    }

    trainOnReplayBatch(batchSize:number, optimizer: tf.Optimizer) {
        const batch = this.replayMemory.sample(batchSize);

        const lossFunction = () => tf.tidy(() => {
            const stateTensor = this.game.stateToTensor(batch.map(example => {
                return example.state;
            }));
            const illegalMoveMask = this.invalidStateMove(batch.map(example => example.state));

            const actionTensor = tf.tensor1d(batch.map(example => example.action), 'int32');
            const qs = (<tf.Tensor>this.onlineNetwork.apply(stateTensor, {training: true})).mul(tf.oneHot(actionTensor, 9)).sum(-1)
            const rewardTensor = tf.tensor1d(batch.map(example => example.reward));
            const nextStateTensor = this.game.stateToTensor(batch.map(example => example.nextState));
            const maskeqNextMaxQTensor =(<tf.Tensor>this.targetNetwork.predict(nextStateTensor)).add(illegalMoveMask).max(-1);
            const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.map(example => example.doneFlag)));
           
            const targetQs = rewardTensor.add(maskeqNextMaxQTensor.mul(doneMask).mul(this.gamma));
            const loss = tf.losses.meanSquaredError(targetQs, qs);
            this.currentLoss = loss.dataSync()[0];
            return loss;
        });
    
        // Calculate the gradients of the loss function with respect to the weights
        // of the online DQN.
        const grads = tf.variableGrads(<any>lossFunction);
        // Use the gradients to update the online DQN's weights.
        optimizer.applyGradients(grads.grads);
        tf.dispose(grads);


    }
}