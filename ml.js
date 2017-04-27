//#! /user/bin/env node

/*require('./include/gpu.js')
debugger;
console.log('starting up');
console.error('this is error');
console.warn('this is warn');
var gpu = new GPU();

var benchmark = function() {
    debugger;
    console.log('benchmarking');
}
var mat_mult = gpu.createKernel(function(A, B) {
    var sum = 0;
    for (var i=0; i<512; i++) {
        sum += A[this.thread.y][i] * B[i][this.thread.x];
    }
    return sum;
}).dimensions([512, 512]);

// Perform matrix multiplication on 2 matrices of size 512 x 512
var C = mat_mult(A, B);

console.log(C);*/
