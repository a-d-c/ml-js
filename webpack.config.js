//var path = require('path');
//var fs = require('fs');
var HtmlWebpackPlugin = require('html-webpack-plugin')

module.exports = {
    entry: [
        "./index"
        ],
    output: {
        path: __dirname,
        filename: "bundle.js",
        publicPath: '/'

    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './templates/index.html'
        })
    ]
    ,module: {
        loaders: [
            { test: /\.css$/, loader: "style-loader!css-loader" }
        ]
    },
    resolve: {
        modules: ['./node_modules', 'node_modules', '../node_modules']
    },
    node: {
        fs: "empty"
       , path: "empty"
    }
};