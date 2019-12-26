import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getIrisData, IRIS_CLASSES } from "./data.js";

window.onload = async() => {

    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    xTrain.print();
    yTrain.print();
    xTest.print();
    xTest.print();

    console.log(xTrain);
    console.log(yTrain);
    console.log(xTest);
    console.log(xTest);

    console.log(IRIS_CLASSES);



};