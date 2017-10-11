/**
 * Abstract Optimizer - base class for all optimizer types
 * @class AbstractOptimizer
*/
export default class AbstractOptimizer { 

    get model() {
        return this._model;
    }
    set model(value) {
        this._model = value;
    }
}
