
import assert from 'assert'
import Matrix from 'ml-matrix'
import fs from 'fs'
import path from 'path'
import parse from  'csv-parse'
import { sigmoid } from '../math_util.js'

import GradientDescent from '../optimizers/GradientDescent.js'
import LogisticRegression from '../models/LogisticRegression.js'

describe('LogisticRegression', function() {
  this.timeout(60000)
  let trainX = null;
  let trainy = null;

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

              resolve();
          });

          var mypath = path.join(process.cwd(), '/testdata/logisticdata.txt')
          fs.createReadStream(mypath).pipe(parser);
        })
    }
  )

  it('should have initial cost of 0.0683', function() {
      const theta = new Matrix([[0], [0], [0]]);
      const X = new Matrix([[1, 1, 1]]);
      const y = new Matrix([[1]]);

      let model = new LogisticRegression(new GradientDescent(0.01, 500));

      assert.equal(0.6931471805599453, model.computeCost(X, y, theta));
  })

  it('should predict class is 1', function() {
    const p = [4, 4];
    
    let model = new LogisticRegression(new GradientDescent(0.01, 500));
    model.train(trainX, trainy);
    
    assert.equal(1, model.predict(p));
  })
})