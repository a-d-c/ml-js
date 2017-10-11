

export default class AbstractModel {
     constructor(optimizer) {
        this._optimizer = optimizer;
        
        if(this._optimizer != null)
            this._optimizer.model = this;
    }

    get optimizer() {
        return this._optimizer;
    }

    train(X, y) {
        throw new Error("Not Implemented");
    }

    predict(X) {
        throw new Error("Not Implemented");
    }

    h(X, theta) {
        throw new Error("Not Implemented");
    }

    computeCost(X, y, theta) {
        throw new Error("Not Implemented");
    }

    plotData(X, y, predictions) {
        throw new Error("Not Implemented");
    }

}
