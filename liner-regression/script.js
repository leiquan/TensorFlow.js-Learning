import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

window.onload = async() => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];
    // 散点图，用的时候谷歌找文档
    // 这个map需要理解一下
    tfvis.render.scatterplot({
        name: "线性回归训练集"
    }, {
        values: xs.map((x, i) => ({ x, y: ys[i] }))
    }, {
        xAxisDomain: [0, 5],
        yAxisDomain: [0, 8]
    });

    // why 连续性
    const model = tf.sequential();
    // 全链接层
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.sgd(0.1)
    });

    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);

    // API参考
    // 超参数的理解
    await model.fit(inputs, labels, {
        batchSize: 4,
        epochs: 100,
        callbacks: tfvis.show.fitCallbacks({
            name: "训练过程"
        }, ['loss'])
    });
};