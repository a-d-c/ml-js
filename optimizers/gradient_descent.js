import Matrix from 'ml-matrix'
import AbstractOptimizer from './abstractOptimizer.js'

/**
 * Gradient Descent
 * @class GradientDescent
 * @param {number} alpha - learning rate 0 < alpha <= 1, default: .01
 * @param {number} iterations - number of iters to attempt to reach convergence: default: 100
*/
export default class GradientDescent extends AbstractOptimizer { 
    constructor(alpha = 0.01, iterations = 100, method = 'vector') {
        super();

        this._alpha = alpha;
        this._iterations = iterations
        this._requiresNormalization = true;
        this._friendlyName = "Gradient Descent";
        this._method = method;
    }

    get friendlyName() {
        return this._friendlyName;
    }
    get requiresNormalization() {
        return this._requiresNormalization;
    }

    get alpha() {
        return this._alpha;
    }

    set alpha(value) {
        this._alpha = value;
    }

    get iterations() {
        return this._iterations;
    }
    set iterations(value) {
        this._iterations = value;
    }

    /**
     * Optimizes theta values for the given inputs and outputs
     * @param {Matrix} X - matrix of input features
     * @param {Matrix} y - matrix of outputs
     * @param {function} J - the const function we want to mimimize
     * @return {object} - contains optimized theta and cost history
     */
    optimize(X, y) {
        const h = this.model.h.bind(this.model);
        const J = this.model.computeCost.bind(this.model);

        let theta = Matrix.zeros(X.columns, 1); //theta is vector, rows = # features (including bias feature)

        if(this.method == 'iterative') {
            theta = this._iterativeOptimize(X, y, h, J);
        }
        else {
            theta = this._vectorOptimize(X, y, h, J);
        }

        return theta;
    }

    _iterativeOptimize(X, y, h, J) {
        let theta = Matrix.zeros(X.columns, 1); //theta is vector, rows = # features (including bias feature)
        const m = X.rows

        for(var i = 0; i < this.iterations; i++) {
            let errorSum = 0;

            for(var r = 0; r < X.rows; r++) {
                let row = X.subMatrixRow([r]);
                let yhat = h(row, theta).get(0,0);
                let error = y.get(r,0) - yhat;

                errorSum += (error * error);

                for(var j = 0; j < theta.rows; j++) {
                    let tj = theta.get(j, 0);
                    let ttemp = tj + (this.alpha * error * yhat 
                                * (1.0 - yhat) * row.get(0, j)); 
                    theta.set(j,0, ttemp);
                }
            }
        }

        return theta;
    } 

    _vectorOptimize(X, y, h, J) {
        const gradient = this.model.gradient.bind(this.model);

        let theta = Matrix.zeros(X.columns, 1); //theta is vector, rows = # features (including bias feature)
        const m = X.rows
        
        for(var i=0; i<this.iterations; i++) {
            
            let grad = gradient(X, y, theta);
            let gradeXAlpha = grad.clone().mulColumn(0, (this.alpha / m));
            theta = theta.subColumnVector(gradeXAlpha);

           /* let hx_t = h(X, theta)
                        .subColumnVector(y)
                        .transpose();
            
            let temp = new Matrix(theta.rows, 1);

            for(var j= 0; j<theta.rows; j++) {
                
                let col = X.subMatrixColumn([j]);
                let tempsum = hx_t
                                .clone()
                                .mulRowVector(col)
                                .sum();
    
                let cost = (1/m) * tempsum;

                let t = theta.get(j, 0);
                
                let tempTheta = t - ((this.alpha/m) * cost);
                
                temp.set(j, 0, tempTheta);
            }

            theta = temp.clone();*/
        }

        return theta;
    }
}