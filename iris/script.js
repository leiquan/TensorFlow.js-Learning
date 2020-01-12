import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getIrisData, IRIS_CLASSES } from "./data.js";

window.onload = async() => {

    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    xTrain.print();
    yTrain.print();
    xTest.print();
    yTest.print();

    console.log(xTrain);
    console.log(yTrain);
    console.log(xTest);
    console.log(xTest);

    console.log(IRIS_CLASSES);

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'
    }));

    model.add(tf.layers.dense({
        units: 3,  // 输出类别的个数，和为1
        activation: "softmax"
    }));

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks({
            name: '训练效果'
        },
        ['loss', 'cal_loss', 'acc', 'val_acc'],
        {
            callbacks: ["onEpochEnd"]
        }
        )
    })

};