
import assert from 'assert'
import Matrix from 'ml-matrix'
import mnist from 'mnist'
import rn from 'random-number'

import { sigmoid, toPolynomialTerms } from '../math_util.js'

import GradientDescent from '../optimizers/GradientDescent.js'
import NeuralNetwork from '../models/NeuralNetwork.js'

describe('Neural Network', function() {

  /*it('calculates cost', function() {
    let hidden1 = new Matrix(3, 1);
    hidden1.setColumn(0, [-30, 20, 20]);
    
    let hidden2 = new Matrix(3, 1);
    hidden2.setColumn(0, [10, -20, -20]);

    let output = new Matrix(3, 1);
    output.setColumn(0, [-10, 20, 20]);

    let network = new NeuralNetwork(null, 2, 2, 1);
    network.nodes = [hidden1, hidden2];
    network.outputLayer = [output];

    network.train();

    let X = [];
    X[0] = new Matrix(1, 3);
    X[0].setRow(0, [1, 0, 0]);

    X[1] = new Matrix(1, 3);
    X[1].setRow(0, [1, 0, 1]);

    X[2] = new Matrix(1, 3);
    X[2].setRow(0, [1, 1, 0]);

    X[3] = new Matrix(1, 3);
    X[3].setRow(0, [1, 1, 1]);

    let y = new Matrix([[1], [0], [0], [1]]);

    let  cost = network.computeCost(X, y);
    
    y = new Matrix([[0], [0], [0], [0]]);

    let cost2 = network.computeCost(X,y);
debugger;
    assert(cost2 > cost);

    X[0] = new Matrix(1, 3);
    X[0].setRow(0, [1, 0, 0]);

    X[1] = new Matrix(1, 3);
    X[1].setRow(0, [1, 0, 1]);

    X[2] = new Matrix(1, 3);
    X[2].setRow(0, [1, 1, 0]);

    X[3] = new Matrix(1, 3);
    X[3].setRow(0, [1, 1, 1]);

   // y = new Matrix([[1], [1], [1], [1]]);

    //debugger;
   // cost = network.computeCost(X,y);

  })

  it('predict XNOR', function() {

    let hidden1 = new Matrix(3, 1);
    hidden1.setColumn(0, [-30, 20, 20]);
    
    let hidden2 = new Matrix(3, 1);
    hidden2.setColumn(0, [10, -20, -20]);

    let output = new Matrix(3, 1);
    output.setColumn(0, [-10, 20, 20]);

    let network = new NeuralNetwork(null, 2, 2, 1);
    network.nodes = [hidden1, hidden2];
    network.outputLayer = [output];

    network.train();

    const X1 = [0, 0]
    const X2 = [0, 1]
    const X3 = [1, 0] 
    const X4 = [1, 1]

    assert.equal(1, network.predict(X1).get(0,0));
    assert.equal(0, network.predict(X2).get(0,0));
    assert.equal(0, network.predict(X3).get(0,0));
    assert.equal(1, network.predict(X4).get(0,0));
  })


*/

  it('calculate mnist cost', function() {

      var getRand = rn.generator({
          min:  -10
        , max:  10
        , integer: true
      });

      var set = mnist.set(500, 0);
  
      var trainingSet = set.training;
      var testSet = set.test;

      var inputSize = trainingSet[0].input.length;

      let hidden1 = new Matrix(inputSize + 1, 1);

      hidden1.apply((j, k) => {
        hidden1.set(j, k, getRand())
      });
    
      let hidden2 = new Matrix(inputSize + 1, 1);
      
      hidden2.apply((j, k) => {
        hidden2.set(j, k, getRand())
      });

      let output = new Matrix(3, 1);
      output.setColumn(0, [-10, 20, 20]);

      let network = new NeuralNetwork(null, 2, 2, 1);
      network.nodes = [hidden1, hidden2];
      network.outputLayer = [output];
  })
})
