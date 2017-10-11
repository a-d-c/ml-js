
import assert from 'assert'
import Matrix from 'ml-matrix'
import { sigmoid, toPolynomialTerms } from '../math_util.js'

describe('Polynomial Terms', function() {

  it('should generate 28 columns', function() {
    const X1 = new Matrix([[2], [3], [4], [5],[6]])
    const X2 = new Matrix([[3], [4], [5], [6], [7]])

    const degree = 6

    const terms = toPolynomialTerms(X1, X2, degree)
    assert.equal(28, terms.columns)
  })
})

describe('Matrix Multiplication', function() {
  it('should calculate the proper output', function() {
      let x = new Matrix(1, 3)
      let theta = new Matrix(3,1);

      x.setRow(0, [2,3,4]);

      theta.setRow(0, [2]);
      theta.setRow(1, [3]);
      theta.setRow(2, [4]);

      let product = x.dot(theta);

      assert.equal(29, product);
      
  })
})
