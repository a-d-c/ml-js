
import assert from 'assert'
import Matrix from 'ml-matrix'
import fs from 'fs'
import path from 'path'
import parse from  'csv-parse'
import { sigmoid } from '../math_util.js'

import GradientDescent from '../optimizers/GradientDescent.js'
import NormalEquation from '../optimizers/NormalEquation.js'
import LinearRegression from '../models/LinearRegression.js'

describe('LinearRegression', function() {
  this.timeout(60000)
  let trainX = null;
  let trainy = null;

  let trainXOrig = null;
  let trainyOrig = null;

  before(
    function() {
     
       return new Promise(function (resolve) {
          var parser = parse({delimiter: ','}, function(err, data) {
              const records = data
              const numFeatures = records[0].length;
              
              trainX = new Matrix(records.length, numFeatures - 1);
              trainy = new Matrix(records.length, 1);

              records.forEach(function(value, index){
                  let xVals = value.slice(0, numFeatures-1);
                  xVals = xVals.map((val)=> {
                      return parseInt(val);
                  });

                  let yVal = value.slice(numFeatures-1, numFeatures);

                  yVal = yVal.map((val)=> {
                      return parseInt(val);
                  });

                  trainX.setRow(index, [...xVals]);
                  trainy.setRow(index, [...yVal]);
              });

              trainXOrig = trainX;
              trainyOrig = trainy;
              resolve();
          });

          var mypath = path.join(process.cwd(), '/testdata/multivariateregression.txt')
          fs.createReadStream(mypath).pipe(parser);
        })
    }
  )

  beforeEach( function() {
      return new Promise(function(resolve) {
          trainX = trainXOrig.clone();
          trainy = trainyOrig.clone();

          resolve();
      })
  }
  )

  it('should predict approx 220K using gradient descent', function() {
    const p = [1000, 1];
    
    let model = new LinearRegression(new GradientDescent(0.1, 10000));
    model.train(trainX, trainy);
    
    var o = model.predict(p).get(0,0);
    console.log(o);
    assert.equal(220060.8150352043, o);
  })

    it('should predict approx 220K using normal-equation', function() {
    const p = [1000, 1];
    
    let model = new LinearRegression(new NormalEquation());
    model.train(trainX, trainy);
    
    var o = model.predict(p).get(0,0);
    console.log(o);
    assert.equal(220070.56444809606, o);
  })
})
