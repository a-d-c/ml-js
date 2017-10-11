# ml-js
## A Simple JS machine learning test project

I found the idea of using Octave to implement the projects within Andrew Ng's Machine Learning course to be a bit painful - so instead I took the opportunity to attempt to implement some of these models in ES6 (which also gave me the opportunity to play with Webpack and Mocha). 

**Semi*-Implemented Models
+Linear Regression
+Logistic Regression
+Regularized Logistic Regression
+Simple Neural-Net (in progress)

You'll find that I take advantage of the ml-matrix library's ```clone``` method quite a bit. While entirely inefficient, it makes debugging much easier as I can quickly compare the two matrices to observe what has changed.
