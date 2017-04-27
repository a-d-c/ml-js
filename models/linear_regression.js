import Matrix from 'ml-matrix'
import { toMatrix } from '../util.js'
import math from 'mathjs'

import AbstractModel from './abstractModel.js'

/**
 * Linear Regression
 * @class LinearRegression
 * @param {OptimizerBase} optimizer - Optimizer (e.g. Gradient Descent)
 * that will be used to minimize the cost funciton.
*/
export default class LinearRegression extends AbstractModel{
        constructor(optimizer) {
            super(optimizer);
        }

        /**
         * Normalize input features to improve optimization performance.
         * @param {Matrix} X - Matrix of input features
         * @returns {Matrix} - Normalized features
         */
        normalize(X) {
            this.normalized = true;
            
            const nbrOfFeatures = X.columns;
            const m = X.rows;
            
            let newX = Matrix.ones(m, 1);
            
            this.means = [];
            this.ranges = [];
            this.deviations= [];

            for (var i = 1; i < nbrOfFeatures; i++) {
                let feature = toMatrix(X.getColumn(i));
                
                let mean = feature.mean();
                let std = math.std(feature);

                this.means.push(mean);
                this.deviations.push(std);
               
                let f_m_mean = feature.subRowVector([mean]);
                let f_mul_std = f_m_mean.mulRowVector([1/std]);
                let normalizedFeature = f_mul_std;

                newX.addColumn(i, normalizedFeature);
            }

            return newX;
        }

        /**
         * Trains the model using the provided training inputs and outputs.
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @returns {Matrix} - mimimized theta.
         */
        train(X, y) {
            //attach col of ones for bias feature
            X = X.addColumn(0, Matrix.ones(X.rows, 1));
            
            let xData = X;

            if(this._optimizer.requiresNormalization)
                xData = this.normalize(X);

            let theta = this._optimizer
                        .optimize(xData, y, this.computeCost, this.h);

            this.theta = theta;

            return theta;
        }

        predict(X) {
            if(this.theta != null) {
                if(!Array.isArray(X)) {
                    X = [X];
                }

                if(this.normalized) {
                    X = X.map((val, index) => {
                        return (val - this.means[index]) / this.deviations[index]; 
                    });
                }

                let mX = new Matrix(1, X.length);
                
                mX.setRow(0, X);
                mX.addColumn(0, Matrix.ones(mX.rows, 1));
                
                let thetaT = this.theta.transpose();
            
                return thetaT.mulRowVector(mX).sum('row')
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
            return X.mmul(theta);
        }

        /**
         * Computes the cost of applying provided theta
         * J = sum((X * theta - y).^2 )/ (2 * m)
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @param {Matrix} theta - coeficients used by the hypothesis
         * @returns {number}
         */
        computeCost(X, y, theta) {
            const m = X.rows;
            let z = this.h(X, theta);
            let z1 = z.subColumnVector(y);
            let z2 = z1.clone()

            z2.apply(function(i, j) {
                var val = z2.get(i, j);
                z2.set(i, j, val * val);
            });

            let sum = z2.sum();

            return (1 / (2 * m)) * sum;
        }

                /**
         * Computes the partial derivative of the cost function.
         * Used by gradient descent
         * @param {Matrix} X - Matrix of input features
         * @param {Matrix} y - n row X 1 col matrix of outputs 
         * @param {Matrix} theta - coeficients used by the hypothesis
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

        /*refactor - move to own class */
        plotData(X,y,predictions) {
            var xData = X.getColumn(0);
            var yData = y.map(val => { return val[0];});

            var newDiv = document.createElement("div");
            var holder = document.getElementById("modelChart");

            document.body.insertBefore(newDiv, holder);

            if(X.columns > 1) 
                this.plotMultiDimension(xData, yData, predictions, newDiv);
            else
                this.plotSingleDimension(xData,yData ,predictions, newDiv);
        }

        plotSingleDimension(X,y,predictions,holder) {
            var trace1 = {
                x: X,
                y: y,
                mode: 'markers',
                type: 'scatter'
            };

            var trace2 = {
                x: X,
                y: predictions,
                mode: 'lines',
                type: 'scatter'
            };

            var data = [trace1, trace2];

            Plotly.newPlot(holder, data);

        }

        plotMultiDimension(X, y, predictions, holder) {
           
            for(let i = 0; i< X.length; i++) {
                var newSpan = document.createElement("div");
                let val = X[i] + ' : ' + y[i] + ' : ' + predictions[i];
                newSpan.innerText = val;
                holder.appendChild(newSpan);
            }


        }

}