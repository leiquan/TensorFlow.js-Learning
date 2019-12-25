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
};