# Mnist digit recognition

  A deep learning algorithem that can be used to differentiate 


## Getting Started

copy the repository through
```
https://github.com/Yimaha/MNIST-digit-recognistion-Neural-Network.git
```
after cloning, just run 
```npm i``` to install all dependencies
```npm start ``` to start the learning process, the AI is not pre-trained

### Prerequisites

Make sure you have npm, node and tensorflow.js installed, for specific installing instruction, 
please refer to ->
https://www.npmjs.com/get-npm
https://nodejs.org/en/download/
https://www.tensorflow.org/js



## Built With

* [TensorFlow](https://www.tensorflow.org/js) - Tensor library used


## Authors

* **Justin Cai**  - [Yimaha](https://github.com/Yimaha)

## License

This project is licensed under the MIT License

## Acknowledgments

3Blue1Brown, his video series on youtube related with machie learning inspired me to start this project -> 
his youtube channel: https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw
his github: https://github.com/3b1b?tab=repositories

Michael Nielsen, his online book related with Machine learning significantly helped me
understand the algorithem and created the project, If you are interested I recommand you to read it as well.
his website -> http://michaelnielsen.org/
the online book -> http://neuralnetworksanddeeplearning.com/about.html
another interesting book -> http://www.deeplearningbook.org/

## About the Project ##

* **NOTE**: this project serves as a more educational purpose. After going through the tutorial which is written in Python, I want to implement it in javascript so I can learn more, that's why it is in javascript. Python is perfectly fine if you like it better

The goal of the project is to recognize hand written digits to a certain acceptable accuracy. In the current state, the accurcy rate is: 
* **97%** after 100 round of study, which takes around 4 - 5 hours on my laptop
Because I had it running on my laptop, I couldn't maximize it potential, but I might try some of the cloud services in the future.

The project uses deep learning and neural network. In simple language, this is an algorithem that simulate human's brain by creating a network of simulated neurons.

The image provided is only 28 x 28 px, which is fairly small, but with enough computing power and data set, it can easily be used recognize any image.

the project was initially written in pure javascript without the support of tensorflow, but I quickly found the code to be repetative and I would love to have some Linear Algebra library to help me out. After completing the project, I decided to rework with tensorflow, which significantly reduce the complexity of the code.

The core of the project is Network.js, which only require you to pass in the shape of the neural network such as

```
const network = New Network([784, 30, 10])
```
after that, you will get your network setup and you can input data as you wish.

The reading process of the training_cases are done in Read.js, where it output an array of objects in the shape of this -> 
```
  {
    output : new Array(10)
    input : new Array(784)
  }
```
make sure the input and output correspond to the input and the output of the network

initial state of algorithem would likely to perform horribly, since the intial state is completely randomly generated, but after around 3 round of studying you should see some significant improvement.

after that the improvement speed would significantly decrease, but the AI would still likely to improve it's accuracy to around 97% after around 100 round.

I woud not go to deep into how the entire algorithm is done because I will be repeating the resources I posted above, BUT I will give a quick summary

1. input a training set, randomize it
2. go through every example.
3. check the error by comparing current output and actual result
4. using backpropagation to back track the error of each node of every layer, error on both the weight and the bias
5. improve the algorithem by reducing the error in the speed of a certain learning rate

This the entire algorithem, please please please watch the video or read the book I posted above if you want to learn, their explaination is extremely precise and easy to understand.

the output result would be place in src/result, and there are some example data in srctestData.txt if you want to see it

