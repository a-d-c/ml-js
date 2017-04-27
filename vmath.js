import Matrix from 'ml-matrix'
//import './include/gpu.js';
import { toMatrix } from './util.js'

export default class VMATH {
    constructor(mode) {
        this.mode = mode;
        this.gpu = new GPU();
    }
    
    multiply(A, B, modeOverride) {
        if(A.columns !== B.rows)
            throw('Multiplication Error: A.columns !== B.rows');

        const m = A.rows;
        const n = A.columns;
        const p = B.columns;

        const size = m * p

        var opt = {
            dimensions: [m, p],
            mode: this.mode,
            loopMaxIterations: size
        };

        /*const f = this.gpu.createKernel(function(A, B, size) {
            
            var sum = 0;
            for (var i=0; i<size; i++) {
                sum += A[this.thread.y][i] * B[i][this.thread.x];
            }
            
            return sum;
        }
        , opt);*/

        return toMatrix(f(A.to2DArray(),B.to2DArray(), size));
    }    
}

//module.exports = VMATH;
//test


/*(function VMATH(mode) {
    return {
        mode: mode,
        multiply = function(A, B, modeOverride) {
            document.write('i am multiple');
        }
    }
})();*/

/*(function VMATH() {
var gpu = new GPU();

return {
    createMultiplier = function(mode,

}

function createMult(mode) {
	var opt = {
        dimensions: [mat_size, mat_size],
        mode: mode
    };

    return gpu.createKernel(function(A, B) {
        var sum = 0;
        for (var i=0; i<512; i++) {
            sum += A[this.thread.y][i] * B[i][this.thread.x];
        }
        return sum;
    }, opt);
}

var mult = {
	cpu: createMult('cpu'),
	gpu: createMult('gpu')
};
}());*/
