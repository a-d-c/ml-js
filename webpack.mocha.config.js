module.exports = {
    target: 'node',
    entry: [
       "./index"
    ],
    output: {
        path: __dirname,
        filename: "mochabundle.js",
        publicPath: '/'

    },
    resolve: {
        modules: ['./node_modules', 'node_modules', '../node_modules']
    }
};