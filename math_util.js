import math from 'mathjs'
import Matrix from 'ml-matrix'

/**
 * The Sigmoid Function
 * @param {number} z - matrix of input features
 * @return {number} - for large positive values of z, should return 
 * near1, for large negative return near 0
 */
function sigmoid(z) {
    let e = 2.718281828459045;

    var sigmoid = 1 / ( 1  + math.pow(e, z * -1));

    return sigmoid;
}

/**
 * To Polynomial Terms
 * @param {Matrix} X1 - matrix containing the first feature
 * @param {Matrix} X2 - matrix containing the second feature
 * @param {number} degree - the power to which we will generate terms
 * @return {Matrix} - a higher dimension feature vector for logistic regression 
 */
function toPolynomialTerms(X1, X2, degree) {
    
    let terms = Matrix.ones(X1.rows, 1);

    for(let i = 1; i <= degree; i++) {
        for(let j = 0; j <= i; j++) {
            let z1 = X1.clone()
            z1.apply(function(a, b) {
               z1.set(a, b, Math.pow(z1.get(a, b), (i-j))); 
            });

            let z2 = X2.clone()
            z2.apply(function(a, b) {
               z2.set(a, b, Math.pow(z2.get(a, b), j)); 
            });

            let z3 = new Matrix(z1.rows, 1);
            
            z1.apply(function(a,b) {
                z3.set(a,b, (z1.get(a,b) * z2.get(a,b)));
            })

            terms.addColumn(terms.columns, z3);
        }
    }

    return terms;
}

export { sigmoid, toPolynomialTerms }