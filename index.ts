import * as tf from '@tensorflow/tfjs-node';
import { Agent } from './agent';
import { GameEngine } from './GameEngine';


tf.loadLayersModel('file://./oldModels/model.json').then(model =>{
    const agent = new Agent(0.9, 0.2, 0.001, 10000, 10000, 200, model);
    agent.train();
})
