import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data.js";

window.onload = async() => {
    const data = getData(400);
    console.log(data);

    tfvis.render.scatterplot({ name: "XOR训练数据" }, {
        values: [
            data.filter(p => p.label === 1),
            data.filter(p => p.label === 0)
        ]
    });

    const model = tf.sequential();

    // 隐藏层
    model.add(
        tf.layers.dense({
            units: 4,
            inputShape: [2],
            activation: "relu"
        })
    );

    // 输出层
    model.add(
        tf.layers.dense({
            units: 1, // 输出一个value
            activation: "sigmoid" // 0-1的概率
        })
    );

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    // 如何把特征数量为2的数据转为tensor
    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks({
            name: "训练过程"
        }, ['loss'])
    });

    console.log(model);
    console.log(model.predict(tf.tensor([
        [2, 2]
    ])));

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([
            [form.x.value * 1, form.y.value * 1]
        ]))

        alert(`预测结果：${pred.dataSync()[0]}`);
    }
};