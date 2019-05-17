const Reader = require('./Read');
const Network = require('./Network');

//create a network for the deep learning, you may change any of the hidden layer to improve accuracy
//however, be aware increasing layer would decrease the speed of learning as well
let network = new Network([784, 30, 10])
let input = Reader.readInput();
let test = Reader.readTest();


network.stochasticGradientDescent(input, 100, 10, 5, test)
//evaluate once after learning complete
network.evaluate(input);