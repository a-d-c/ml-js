if (module.hot) {
  module.hot.accept()
}

import Matrix from 'ml-matrix'
import parse from  'csv-parse/lib/sync'
import { sigmoid, toPolynomialTerms } from './math_util.js'

import GradientDescent from './optimizers/GradientDescent.js'
import NormalEquation from './optimizers/NormalEquation.js'
import LinearRegression from './models/LinearRegression.js'
import LogisticRegression from './models/LogisticRegression.js'

var fileInput = document.getElementById('testFile');
var button = document.getElementById('predButton');

var model;

function importData(e) {
    console.log(e);
    
    var output;
    const reader = new FileReader();
    reader.onload = onImported;

    reader.readAsText(e.target.files[0]);
}

function onImported(e) {
   var output =  e.target.result;
   var records = parse(output, {auto_parse: true});
   
   //const gdModel = new LinearRegression(new GradientDescent(0.01, 10000));
   //const nModel = new LinearRegression(new NormalEquation());
   
  // performRegression(records, gdModel);
  // performRegression(records, nModel);

  const model = new LogisticRegression(new GradientDescent(0.02, 150000), true, 1);

  performRegression(records, model);
}

function performRegression(records, model) {
    debugger;
    let numFeatures = records[0].length;
    
    let X = new Matrix(records.length, numFeatures - 1);
    let y = new Matrix(records.length, 1);

    
    records.forEach(function(value, index){
       let xVals = value.slice(0, numFeatures-1);
       let yVal = value.slice(numFeatures-1, numFeatures);
       X.setRow(index, [...xVals]);
       y.setRow(index, [...yVal]);
    });

    debugger
    var result = model.train(X, y);

    var predictions = [];

    var xData = X;
    //var xData = X.subMatrix(0, X.rows-1, 1, X.columns-1); //
    //var xData = X.getColumn(1);
    //var xData = X.subMatrixColumn()
    var yData = y;

    xData.forEach(function(val, i) {
        //debugger;
        let pred = model.predict(val);
        predictions.push(pred);
    });

    debugger;
    onRegressionComplete(xData, yData, predictions, model);
}

function onRegressionComplete(X, y, predictions, model) {
    //var data = prepChartData(X, y, predictions);
    
    model.plotData(X, y, predictions);
    //drawChart(data, model);
}

function predict(e) {
    var input = document.getElementById('valIn');
    var inVal = Number(input.value);

    var output = model.predict(inVal);
    console.log(output);
}

fileInput.addEventListener('change', importData);
predButton.addEventListener('click', predict);