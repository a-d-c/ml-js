import Matrix from 'ml-matrix'
import { toMatrix } from '../util.js'
import { sigmoid } from '../math_util.js'
import math from 'mathjs'

import AbstractModel from './AbstractModel.js'

/**
 * Neural Network
 * @class Simple Neural Network
 * @param {OptimizerBase} optimizer - Optimizer (e.g. Gradient Descent)
*/
export default class NeuralNetwork extends AbstractModel {
       constructor(optimizer, inputSize, hiddenNodeCount, numLabels) {
           super(optimizer);

           this._inputSize = inputSize;
           this._hiddenNodeCount = hiddenNodeCount;
           this._numLabels = numLabels;
       }

       set nodes(value) {
           this._nodes = value;
       }

       set outputLayer(value) {
           this._outputLayer = value;
       }
 
       prepareModel() {
            this._nodes = [];

            for(let i=0;i<this._hiddenNodeCount;i++) {
                let weights = new Matrix(this._inputSize + 1, 1);
                this._nodes.push(weights);
            }

            this._outputLayer = [];
            
            for(let j=0; j<this._numLabels; j++) {
                let weights = new Matrix(this_hiddenNodeCount + 1, 1);
                this._outputLayer.push(weights);
            }
       }

        /**
         * Trains the model using the provided training inputs and outputs.
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @returns {Matrix} - mimimized theta.
         */
        train(X, y) {

            this._trained = true;

        }

        /**
         * Predicts a class (0 or 1) based on the given input
         * @param {Matrix} X - Input features (exclude bias feature)
         * @returns {Number} - predicted class
         */
        predict(X) {
            //debugger;
            var self = this;

            if(!self._trained)
                throw('Model has not been trained.');

            if(!Array.isArray(X))
                throw('X must be an array');

            X.splice(0,0,1);

            let mX = new Matrix(1, X.length);
            mX.setRow(0, X);

            return this.h(mX);
        }

        /**
         * Computes the hypothesis h(x)
         * @param {Matrix} X - Matrix of input features
         * @returns {Matrix}
         */
        h(X) {
            let y = new Matrix(this._numLabels, 1);

            let layer1 = new Matrix(1, this._hiddenNodeCount + 1);
            layer1.set(0, 0, 1);

            for(let i = 0; i<this._hiddenNodeCount; i++) {
               //debugger;
                let lx = X.clone();
                let z = lx.dot(this._nodes[i]);
                let a = sigmoid(z);

                layer1.set(0, i+1, a)
            }

            for(let j=0; j<this._numLabels; j++) {
                let lclone = layer1.clone();
                let z2 = lclone.dot(this._outputLayer[j]);
                //let z2s = sigmoid(z2) >= 0.5 ? 1 : 0;
                let z2s = sigmoid(z2);
                y.setRow(j, [z2s])
            }

            return y;
        }


        /**
         * Computes the cost of applying provided theta
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @param {array} nodes - 2D array of layers/nodes. Value is matrix of weights.
         * @returns {number}
         */
        computeCost(X, y) {
            debugger;
            const m = X.length;
            //let _X = X.clone();
            let _y = y.clone();
            
            let errorSum = 0;
            for(var i=0; i<m; i++) {
                let xi = X[i];

                let hxi = this.h(xi)

                for(var k = 0; k<this._numLabels; k++) {
                    let yk = y.get(k,0);
                    let hxik = hxi.get(k, 0);

                    let loghx = Math.log(hxik);
                    let loghxNegY = (yk * -1)*loghx;

                    let minusY = 1-yk;
                    let oneMinusHxik = 1 - hxik;
                    let minusLogHx = Math.log(oneMinusHxik);

                    let mYMLogHx = minusY * minusLogHx;

                    let innerVal = loghxNegY - mYMLogHx;

                    errorSum += innerVal;
                }
            }
            debugger;
            return (1/m) * errorSum;
        }

        /**
         * Computes the partial derivative of the cost function.
         * Used by gradient descent
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @param {Matrix} theta - coeficients used by the hypothesis
         * @param {Number} lambda - optional reqularization parameter
         * @returns {Matrix} - vector of ∂J(θ)/∂θj 
         */
        gradient(X, y, theta) {
            let m = X.rows;
            let hx_t = this.h(X, theta)
                        .subColumnVector(y)
                        .transpose();
            
            let temp = new Matrix(theta.rows, 1);

            let thetaArray = theta.getColumn(0);
            let reg = null;

            if(this._lambda != null) {
               let sqTheta = thetaArray
                    .filter((value, index, array) => {
                        return index != 0;
                    }).map((value, index) => {
                        return Math.pow(value, 2);
                    }).reduce((p, c) => {
                        return p + c;
                    }, 0);

                let reg = (this._lambda / (2 * m)) * sqTheta;
            }

            let gradients = thetaArray.map((j, index) => {

                let col = X.subMatrixColumn([index]);

                let tempsum = hx_t.clone()
                        .mulRowVector(col)
                        .sum();
    
                let tempcost = (1/m) * tempsum;

                if(index > 0 && reg != null) {
                    tempcost += reg;
                }

                return tempcost;
            });

            return toMatrix(gradients);
        }

        /* refactor - this doesn't belong in the model. */
        plotData(X,y,predictions) {

            var xData = X.getColumn(0);
            var yData = y.map(val => { return val[0];});

            var labeledDiv= document.createElement("div");
            var predictionDiv = document.createElement("div");
            var holder = document.getElementById("modelChart");

            document.body.insertBefore(labeledDiv, holder);
            document.body.insertBefore(predictionDiv, holder);

            let class1X = [];
            let class1Y = [];
            let class2X = [];
            let class2Y = [];
            for(let i = 0; i<X.rows; i++) {
                if(yData[i] == 0) {
                    class1X.push(X.get(i, 0));
                    class1Y.push(X.get(i, 1));
                } else {
                    class2X.push(X.get(i, 0));
                    class2Y.push(X.get(i, 1));
                }
            }
            debugger;
           var trace1 = {
                x: class1X,
                y: class1Y,
                mode: 'markers',
                type: 'scatter',
                color: '#E74C3C'
            };

            var trace2 = {
                x: class2X,
                y: class2Y,
                mode: 'markers',
                type: 'scatter',
                color: '#3498DB'
            };

            var data = [trace1, trace2];

            var layout = {
                title: 'Labeled Values'
            }

            Plotly.newPlot(labeledDiv, data, layout);
            

            class1X = [];
            class1Y = [];
            class2X = [];
            class2Y = [];
            for(let i = 0; i<X.rows; i++) {
                //debugger;
                if(predictions[i] == 0) {
                    class1X.push(X.get(i, 0));
                    class1Y.push(X.get(i, 1));
                } else {
                    class2X.push(X.get(i, 0));
                    class2Y.push(X.get(i, 1));
                }
            }
            debugger;
           trace1 = {
                x: class1X,
                y: class1Y,
                mode: 'markers',
                type: 'scatter',
                color: '#E74C3C'
            };

            trace2 = {
                x: class2X,
                y: class2Y,
                mode: 'markers',
                type: 'scatter',
                color: '#3498DB'
            };

            data = [trace1, trace2];

            layout = {
                title: 'Predicted Values'
            }

            Plotly.newPlot(predictionDiv, data, layout);
        }
}
