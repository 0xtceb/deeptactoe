import * as tf from '@tensorflow/tfjs-node';
import { Memory } from './GameEngine';
export class ReplayMemory {

    buffer: Memory[];
    length: number;
    maxLen: number;
    index: number;
    bufferIndices: number[] = [];
    constructor(buffSize: number) {
        this.maxLen = buffSize;
        this.buffer = new Array(buffSize).fill(null);
        for(let i=0; i<this.maxLen; i++) {
            this.bufferIndices.push(i);
        }
        this.index = 0;
        this.length = 0;
    }

    append(memory: Memory) {
        this.buffer[this.index] = memory;
        this.index = (this.index + 1) % this.maxLen;
        this.length = Math.min(this.index + 1, this.maxLen);
    }

    sample(batchSize: number): Memory[] {
        if(batchSize > this.maxLen) {
            throw new Error('Batch size exceeds memory length');
        }
        tf.util.shuffle(this.bufferIndices);
        const out = [];
        for (let i = 0; i < batchSize; ++i) {
          out.push(this.buffer[this.bufferIndices[i]]);
        }
        return out;
    }
}