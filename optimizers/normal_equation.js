import Matrix from 'ml-matrix'
import VMATH from '../vmath.js';
import AbstractOptimizer from './abstractOptimizer.js'
/**
 * Normal Equation - fits linear regression via normal equation rather than Gradient Descent
 * @class Normal Equation
*/
export default class NormalEquation extends AbstractOptimizer{ 
    constructor() {
        super();

        this._requiresNormalization = false;
        this._friendlyName = "Normal Equation";
    }

    get friendlyName() {
        return this._friendlyName;
    }
    get requiresNormalization() {
        return this._requiresNormalization;
    }
    /**
     * Optimizes theta values for the given inputs and outputs
     * theta = (Xt*X)^-1 * Xt * y
     * @param {Matrix} X - matrix of input features
     * @param {Matrix} y - matrix of outputs
     * @param {function} J - IGNORED
     * @return {object} - contains optimized theta and cost history
     */
    optimize(X, y) {
        var xT = X.transpose();
        var xTx = xT.mmul(X);
        var pinv = xTx.inv();
        var pInvX = pinv.mmul(xT);
        var theta = pInvX.mmul(y);

        return theta;
        
    }

}