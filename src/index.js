const read = require('./Read');
const Network = require('./Network');

let network = new Network([784, 30, 10])
let input = read();
network.stochasticGradientDescent(input, 30, 10, 3)