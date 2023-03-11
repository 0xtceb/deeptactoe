import { GameEngine, Player, Memory, State } from './GameEngine';
import * as tf from '@tensorflow/tfjs-node';
import { DeepQNetwork } from './dqn';
import { ReplayMemory } from './replayMemory';

export class Agent{
    onlineNetwork = new DeepQNetwork().model;
    targetNetwork = new DeepQNetwork(false).model;
    replayMemory: ReplayMemory;
    gamma: number;
    epsilon: number;
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
    trainerModel: tf.LayersModel;
    constructor(gamma: number, epsilon: number, decay: number, maxGames: number, memorySize: number, batchSize: number, trainer: tf.LayersModel) {
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.decay = decay;
        this.maxGames = maxGames;
        this.replayMemory = new ReplayMemory(memorySize);
        this.game = new GameEngine();
        this.iaPlayer = this.game.aiPlayer;
        this.batchSize = batchSize;
        this.trainerModel = trainer;
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
                }
            });
            state[n].ennemy.forEach((v, i) => {
                if(v !== 0) {
                    buffer.set(-1e9, n, i);
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

    playStep(log: boolean): Memory {
        let memory: Memory;

        if(this.game.currentPlayer === this.iaPlayer) {
                const pred = tf.tidy(() => {
                    const state = this.game.state(this.game.aiPlayer);
                    const stateToTensor = this.game.stateToTensor(state);
                    const invalidMovesMask = this.invalidMoves();
                    const predictions = (<tf.Tensor>this.onlineNetwork.predict(stateToTensor));
                    const maskedPredictions = predictions.add(invalidMovesMask);
                    return maskedPredictions.argMax(-1).dataSync()[0];
                })
                memory = this.game.makeMove(pred);
        } else {
            if(Math.random() < 0.5 && this.gameCount < 1000) {
                memory = this.game.makeMove(this.randomValidMove());
            } else {
                const pred = tf.tidy(() => {
                    const state = this.game.state(this.game.currentPlayer);
                    const stateToTensor = this.game.stateToTensor(state);
                    const invalidMovesMask = this.invalidMoves();
                    const predictions = (<tf.Tensor>this.targetNetwork.predict(stateToTensor));
                    const maskedPredictions = predictions.add(invalidMovesMask);
                    return maskedPredictions.argMax(-1).dataSync()[0];
                })
                memory = this.game.makeMove(pred);
            }
        }
        this.replayMemory.append(memory);

        if(memory.doneFlag) {
            if(log) {
                if(memory.tie) {
                    this.tieGames ++;
                }
                if(memory.won === this.iaPlayer) {
                    this.wonGames ++;
                }
                if(!memory.tie && memory.won !== this.iaPlayer) {
                    this.lostGames ++;
                }
                console.log(`IA PLAYER ${this.game.numberToPlayer(this.iaPlayer)}`)
                console.log(`Total Games ${this.gameCount} - Games won ${this.wonGames} - Games lost ${this.lostGames} - Ties ${this.tieGames} - Reward ${memory.reward} - loss ${this.currentLoss}`);
                this.game.displayBoard();
            }
            this.game.reset();
            this.iaPlayer = this.game.aiPlayer;
        }
        return memory;
    }

    train() {
        console.log('===== FILLING UP MEMORY =====');
        while(this.replayMemory.length < this.replayMemory.maxLen) {
            const memory = this.game.makeMove(this.randomValidMove());
            if(memory.doneFlag) {
                this.game.reset();
                this.iaPlayer = this.game.aiPlayer;
            }
            this.replayMemory.append(memory);
        }
        console.log('====== MEMORY FILLED =====');
        console.log('Launching training');
        const optimizer = tf.train.adam(0.001);
        while(this.gameCount < this.maxGames) {
            this.trainOnReplayBatch(this.batchSize, optimizer);
            const memory = this.playStep(true);
            if(memory.doneFlag) {
                const currentEpsilon = this.epsilon * this.decay * this.gameCount;
                this.gameCount ++;
            }
            if(this.gameCount % 1000 === 0) {
                DeepQNetwork.copyWeights(this.onlineNetwork, this.targetNetwork);
            }
        }

        this.onlineNetwork.save('file://./');
        console.log('training finished & model saved !');
    }

    trainOnReplayBatch(batchSize:number, optimizer: tf.Optimizer) {
        const batch = this.replayMemory.sample(batchSize);

        const lossFunction = () => tf.tidy(() => {
            const stateTensor = this.game.stateToTensor(batch.map(example => {
                return example.state;
            }));
            const actionTensor = tf.tensor1d(batch.map(example => example.action), 'int32');
            const qs = (<tf.Tensor>this.onlineNetwork.apply(stateTensor, {training: true})).mul(tf.oneHot(actionTensor, 9)).sum(-1);
            const rewardTensor = tf.tensor1d(batch.map(example => example.reward));
            const nextStateTensor = this.game.stateToTensor(batch.map(example => example.nextState));
            const illegalMoveMask = this.invalidStateMove(batch.map(example => example.nextState));
            const maskeqNextMaxQTensor =(<tf.Tensor>this.targetNetwork.predict(nextStateTensor)).mul(illegalMoveMask).max(-1);
            const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.map(example => example.doneFlag)).asType('float32'));
            const targetQs = rewardTensor.add(maskeqNextMaxQTensor.mul(this.gamma));
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