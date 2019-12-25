import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data.js";

window.onload = async() => {
    const data = getData(400);
    console.log(data);

    tfvis.render.scatterplot({ 'name': '逻辑回归训练数据' }, {
        values: [
            data.filter(p => p.label === 1),
            data.filter(p => p.label === 0)
        ]
    });

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2], // 注意
        activation: 'sigmoid' // 将输出值压缩到0-1之间
    }));

    model.compile({ loss: tf.losses.logLoss });

};