import * as tf from '@tensorflow/tfjs-node';

export class DeepQNetwork {
    model: tf.Sequential;

    constructor(trainable: boolean = true, optimizer: tf.Optimizer | null) {
        this.model = tf.sequential();
        this.model.trainable = trainable;
        this.model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            activation: 'relu',
            padding: 'same',
            inputShape: [3, 3, 2]
        }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dense({ units: 100, activation: 'relu'}));
        this.model.add(tf.layers.dropout({rate: 0.1}));
        this.model.add(tf.layers.dense({ units: 100, activation: 'relu'}));
        this.model.add(tf.layers.dropout({rate: 0.1}));
        this.model.add(tf.layers.dense({ units: 9, activation: 'relu'}));
        optimizer ? this.model.compile({optimizer: optimizer, loss: 'meanSquaredError'}) : null;
    }

    public static copyWeights(destNetwork: tf.Sequential, srcNetwork: tf.Sequential) {
        let originalDestNetworkTrainable;
        if (destNetwork.trainable !== srcNetwork.trainable) {
          originalDestNetworkTrainable = destNetwork.trainable;
          destNetwork.trainable = srcNetwork.trainable;
        }
        destNetwork.setWeights(srcNetwork.getWeights());
        if (originalDestNetworkTrainable != null) {
          destNetwork.trainable = originalDestNetworkTrainable;
        }
    }
}