import Matrix from 'ml-matrix'
import { toMatrix } from '../util.js'
import { sigmoid, toPolynomialTerms } from '../math_util.js'
import math from 'mathjs'

import AbstractModel from './abstractModel.js'

/**
 * Logistic Regression
 * @class LogisticRegression
 * @param {OptimizerBase} optimizer - Optimizer (e.g. Gradient Descent)
 * that will be used to minimize the cost funciton.
*/
export default class LogisticRegression extends AbstractModel {
       constructor(optimizer, regularized = false, lambda = null) {
           super(optimizer);

           this._regularized = regularized;
           this._lambda = lambda;
       }

        /**
         * Trains the model using the provided training inputs and outputs.
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @returns {Matrix} - mimimized theta.
         */
        train(X, y) {
            if(this.theta != null)
                throw('Model has already been trained. Create a new model instead.');

            if(this._regularized) {
                debugger
                let X1 = X.subMatrixColumn([0]);
                let X2 = X.subMatrixColumn([1]);
                let degree = 6;

                X = toPolynomialTerms(X1, X2, degree);
            } else {
                //attach col of ones for bias feature
                X = X.addColumn(0, Matrix.ones(X.rows, 1));
            }

            this.theta = this._optimizer.optimize(X, y);
    
            return this.theta;
        }

        /**
         * Predicts a class (0 or 1) based on the given input
         * @param {Matrix} X - Input features (exclude bias feature)
         * @returns {Number} - predicted class
         */
        predict(X) {
            var self = this;
            if(self.theta != null) {
                if(!Array.isArray(X)) {
                    X = [X];
                }

                let mX = new Matrix(1, X.length);
                mX.setRow(0, X);

                if(this._regularized) {
                    let X1 = mX.subMatrixColumn([0]);
                    let X2 = mX.subMatrixColumn([1]);
                    let degree = 6;

                    mX = toPolynomialTerms(X1, X2, degree);
                } else {
                    mX.addColumn(0, Matrix.ones(mX.rows, 1));
                }

                var output = this.h(mX, this.theta).get(0,0);
                var outVal = output >= 0.5 ? 1 : 0;

                return outVal;
            } else {
                throw('Model has not been trained.');
            }
        }

        /**
         * Computes the hypothesis h(x)
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} theta - coeficients used by the hypothesis
         * @returns {Matrix}
         */
        h(X, theta) {
            var z= X.clone().mmul(theta);

            return z.apply((i,j) => {
                z.set(i,j, sigmoid(z.get(i,j)));
            });
        }


        /**
         * Computes the cost of applying provided theta
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @param {Matrix} theta - coeficients used by the hypothesis
         * @returns {number}
         */
        computeCost(X, y, theta) {
            const m = X.rows;
            let _X = X.clone();
            let _y = y.clone();
            let _theta = theta.clone();
            let z = this.h(_X, _theta);
            let logz = z.clone();
            
            logz.apply((i,j) => {
                if(logz.get(i,j) > 0) {
                    logz.set(i, j, Math.log(logz.get(i,j)));
                }
            });
            
            let yneg = _y.clone().neg();
            let ynegXlogz = yneg.clone().mulColumnVector(logz);
            let oneMinusY = yneg.clone().addRowVector(Matrix.ones(1,1));
            let oneMinushX = z.clone().neg().addRowVector(Matrix.ones(1,1));

            let logOneMinushX = oneMinushX.clone();

            logOneMinushX.apply((i,j) => {
                if(logOneMinushX.get(i,j) > 0) {
                    logOneMinushX.set(i, j, Math.log(logOneMinushX.get(i,j)));
                }
            });

            let ytimesloghx = oneMinusY.clone().mulColumnVector(logOneMinushX);
            let subparts = ynegXlogz.clone().subColumnVector(ytimesloghx);

            let sum = subparts.sum();

            return (1/m) * sum;
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