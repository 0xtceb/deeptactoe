import * as tf from '@tensorflow/tfjs-node';

export enum Player {
    O = 1,
    X = 2,
    NAN = 0
}

export interface State {
    ai: number[],
    ennemy: number[]
}

export interface Memory {
    state: State,
    action: number,
    doneFlag: boolean,
    currentPlayer: Player,
    won: Player | null,
    tie: boolean,
    nextState: State,
    reward: number
}
export type board = number[];

export class GameEngine {

    board: board;
    currentPlayer: Player;
    aiPlayer: Player;
    reward: number = 0;
    winningMoves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    constructor() {
        this.board = new Array(9).fill(Player.NAN);
        this.currentPlayer = Math.floor(Math.random() * 2) + 1;
        this.aiPlayer = Math.floor(Math.random() * 2) + 1;
        console.log('=== TIC TAC TOE ===');
        this.displayBoard();
    }

    makeMove(index: number): Memory {
        const currentState = this.state(this.currentPlayer);
        const oldBoard = this.board.slice();
        this.board[index] = this.currentPlayer;
        const nextState = this.state(this.currentPlayer);
        let winner = null;
        let tie = false;
        let done = false;
        winner = this.winner();

        if(winner !== null || (this.tie() === true && winner === null)) {
            done = true;
        }


        /*if(blockedMove && this.currentPlayer === this.aiPlayer && !done) {
            this.reward = this.reward + 0.2;
        }*/
        /*if(missedBlock && this.currentPlayer === this.aiPlayer && !done) {
            this.reward = this.reward - 0.5;
        }*/

        /*if(blockedLittleMove && this.currentPlayer === this.aiPlayer && !done) {
            this.reward = this.reward + 0.2;
        }*/

        
        if(this.tie() === true && winner === null) {
            tie = true;
        }

        this.currentPlayer = this.currentPlayer === Player.X ? Player.O : Player.X;

        return {state: currentState, reward: 0, action: index,currentPlayer: this.currentPlayer, doneFlag: done, won: winner, tie: tie, nextState: nextState};
    }

    blockedMove(previousBoard: board, nextBoard: board, player: Player) {
        let blocked = false;
        this.winningMoves.forEach(w => {
            let betterBlock = 0;
            let canBlock = false;
            for(let i=0; i<w.length; i++) {
                if(previousBoard[w[i]] !== player && previousBoard[w[i]] !== Player.NAN) {
                    betterBlock ++;
                }
                if(previousBoard[w[i]] === Player.NAN) {
                    canBlock = true;
                }
            }

            if(betterBlock === 2 && canBlock === true) {
                for(let i=0; i<w.length; i++) {
                    if(nextBoard[w[i]] === player) {
                        blocked = true;
                    }
                }   
            }
        });

        return blocked;
    }

    missedBlock(previousBoard: board, nextBoard: board, player: Player) {
        let blocked = false;
        this.winningMoves.forEach(w => {
            let betterBlock = 0;
            let canBlock = false;
            for(let i=0; i<w.length; i++) {
                if(previousBoard[w[i]] !== player && previousBoard[w[i]] !== Player.NAN) {
                    betterBlock ++;
                }
                if(previousBoard[w[i]] === Player.NAN) {
                    canBlock = true;
                }
            }

            if(betterBlock === 2 && canBlock === true) {
                for(let i=0; i<w.length; i++) {
                    if(nextBoard[w[i]] === player) {
                        blocked = true;
                    }
                }   
            }
        });

        return !blocked;
    }

    state(playerPov: Player): State {
        
        const aiBoard = this.board.map((value) => {
            return (value === playerPov) ? 1 : 0;
        });

        const ennemyBoard = this.board.map((value) => {
            return value !== Player.NAN && value !== playerPov ? 1 : 0;
        });

        return {
            ai: aiBoard,
            ennemy: ennemyBoard,
        }
    }

    reverseState(state: State): State {
        return {
            ai: state.ennemy,
            ennemy: state.ai
        }
    }

    stateToTensor(state: State | State[]): tf.Tensor {
        if(!Array.isArray(state)) {
            state = [state];
        }

        const numStates = state.length;
        const aiGrid = new Array(numStates);
        const ennemyGrid = new Array(numStates);
        for(let n=0; n < numStates; n++) {
            aiGrid[n] = [
                state[n].ai.slice(0, 3),
                state[n].ai.slice(3, 6),
                state[n].ai.slice(6, 9)
            ];

            ennemyGrid[n] = [
                state[n].ennemy.slice(0, 3),
                state[n].ennemy.slice(3, 6),
                state[n].ennemy.slice(6, 9)
            ];

        
        }
        const buffer = tf.buffer([numStates, 3, 3, 2]);

        for(let n=0; n < numStates; n++) {
            for(let row=0; row < 3; row ++) {
                for(let col=0; col < 3; col ++) {
                    buffer.set(aiGrid[n][row][col], n, row , col, 0);
                    buffer.set(ennemyGrid[n][row][col], n, row, col, 1);
                }
            }
        }

        return buffer.toTensor();
    }

    winner(): Player | null {
        let winner = null;
        this.winningMoves.forEach(winMove => {
            if(this.board[winMove[0]] === Player.O && this.board[winMove[1]] === Player.O && this.board[winMove[2]] === Player.O) {
                winner = Player.O;
            }
            if(this.board[winMove[0]] === Player.X && this.board[winMove[1]] === Player.X && this.board[winMove[2]] === Player.X) {
                winner = Player.X;
            }
        })
        return winner;
    }

    tie(): boolean {
        return this.board.every(box => box !== Player.NAN);
    }

    reset() {
        this.board = new Array(9).fill(0);
        this.currentPlayer = Math.floor(Math.random() * 2) + 1;
        this.aiPlayer = Math.floor(Math.random() * 2) + 1;
        this.reward = 0;
    }

    numberToPlayer(num: number): string {
        switch(num) {
            case 0: return '-';
            case 1: return 'O';
            case 2: return 'X';
            default:
                return '';
        }
    }

    displayBoard(): void {
        const humanBoard = this.board.map(box => this.numberToPlayer(box));
        console.log(`     [${humanBoard[0]}, ${humanBoard[1]}, ${humanBoard[2]}]
     [${humanBoard[3]}, ${humanBoard[4]}, ${humanBoard[5]}]
     [${humanBoard[6]}, ${humanBoard[7]}, ${humanBoard[8]}] `);
    }


}