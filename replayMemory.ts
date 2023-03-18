import * as tf from '@tensorflow/tfjs-node';
import { Memory } from './GameEngine';
export class ReplayMemory {

    buffer: Memory[] = [];
    length: number;
    maxLen: number;
    index: number;
    bufferIndices: number[] = [];
    lastMemoryIndex: number = 0;
    constructor(buffSize: number) {
        this.maxLen = buffSize;
        for(let i=0; i< this.maxLen; i++) {
            this.buffer.push(null as any);
        }
        for(let i=0; i<this.maxLen; i++) {
            this.bufferIndices.push(i);
        }
        this.index = 0;
        this.length = 0;
    }


    eraseLastMemory(memory: Memory) {
        this.buffer[this.lastMemoryIndex] = memory;
    }

    append(memory: Memory) {
        this.buffer[this.index] = memory;
        this.lastMemoryIndex = this.index;
        this.length = Math.min(this.index + 1, this.maxLen);
        this.index = (this.index + 1) % this.maxLen;
    }

    sample(batchSize: number): Memory[] {
        if(batchSize > this.maxLen) {
            throw new Error('Batch size exceeds memory length');
        }
        tf.util.shuffle(this.bufferIndices);
        const out = [];
        for (let i = 0; i < batchSize; ++i) {
            if(this.buffer[this.bufferIndices[i]]) {
              out.push(this.buffer[this.bufferIndices[i]]);
            }
        }


        return out;
    }

    public static cloneMemory(memory: Memory): Memory {
        return {...memory};
    }
}