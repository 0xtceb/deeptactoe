import { DeepQNetwork } from './dqn';
import { Agent } from './agent';
import * as tf from '@tensorflow/tfjs-node';
const agent = new Agent(0.3, 0.1, 0.9, 1000, 20000, 10000, 200);

agent.train();

