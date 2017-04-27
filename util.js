import Matrix from 'ml-matrix'

function toMatrix(arrIn) {

    var len = arrIn.length;

    var isArray = Array.isArray(arrIn[0]) || (arrIn[0] instanceof Float32Array);
    var y =  isArray ? arrIn[0].length : 1;

    var matrix = new Matrix(len, y);
    arrIn.forEach((x, i) => { 
        if(!isArray) x = [x];

        matrix.setRow(i, x); 
    });

    return matrix;
}

export { toMatrix }