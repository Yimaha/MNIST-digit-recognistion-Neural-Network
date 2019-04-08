require('@tensorflow/tfjs-node')

const read = require('./Read');
const fs = require('fs');
const tf = require('@tensorflow/tfjs');

fs.writeFileSync('testing.txt', '');

const testSize = 1000;

class Network {
  constructor(sizes) {
    let input = read();
    this.num_layers = sizes.length;
    //randomized value of bias and weight between 0 - 1;
    this.sizes = sizes;
    this.biases = sizes.reduce((root, value, index) => {
      if (index !== 0) {
        root.push(tf.randomNormal([value, 1], 0, 1, "int32"));
      }
      return root;
    }, [])
    this.weights = sizes.reduce((root, value, index, array) => {
      if (index !== 0) {
        root.push(tf.randomNormal([value, array[index - 1]], 0, 1, "int32"));
      }
      return root;
    }, [])
  }



  //used to output the result after learning
  feedForWard(input) {
    input = tf.tensor(input).expandDims(1);
    let output = this.biases.reduce((root, bias_layer, layer_index) => {
      return this.sigmoid(tf.matMul(this.weights[layer_index], root).add(bias_layer));
    }, input)
    return output;
  }

  async evaluate(data) {
    let evaluate = 0;
    for (let i = n - testSize; i < n; i++) {
      let a = await this.feedForWard(data[i].input).array();
      let biggestIndex = 0;
      a.map((val, index, array) => {
        if (val[0] > array[biggestIndex][0])
          biggestIndex = index;
      })
      if (data[i].output[biggestIndex])
        evaluate++;
    }
    console.log('finished, ' + `the correct rate is ${evaluate} / ${testSize} in epochs ${i} \n`)
  }
  // eta = learning speed
  //epoches is not used since we are not planning on using yet  
  //training data format: 
  /** {
   * input: [array of input]
   *    * output: [expected output for array(1 for desire digit and 0 for all others)]
   * } */
  async stochasticGradientDescent(training_data, epochs, mini_batch_size, eta, test_data = null) {

    let n_test = test_data && test_data.length;
    let n = training_data.length;
    for (let i = 0; i < epochs; i++) {
      console.log('Right now in epochs ' + i)
      //raindomize the data
      let data = training_data.map(a => [Math.random(), a])
        .sort((a, b) => a[0] - b[0])
        .map(a => a[1]);
      //split the batches into bunch of mini batches


      //////construction testing phase//////
      // let mini_batch = [];
      // mini_batch[0] = data[0];
      // this.updateMiniBatch(mini_batch, eta);
      /////////////////////////////////////////

      //////////actual code//////////////////////////////////////////////////

      let mini_batches = [];
      for (let i = 0; i < (n - testSize); i += mini_batch_size)
        mini_batches.push(data.slice(i, mini_batch_size + i));
      //updating the algorithm by letting it learn
      console.log('amount of mini_batches is ' + mini_batches.length)
      for (let i = 0; i < mini_batches.length; i++) {
        this.updateMiniBatch(mini_batches[i], eta)
      }
      let evaluate = 0;
      for (let i = n - testSize; i < n; i++) {
        let a = await this.feedForWard(data[i].input).array();
        let biggestIndex = 0;
        a.map((val, index, array) => {
          if (val[0] > array[biggestIndex][0])
            biggestIndex = index;
        })
        if (data[i].output[biggestIndex])
          evaluate++;
      }
      console.log('finished, ' + `the correct rate is ${evaluate} / ${testSize} in epochs ${i} \n`)
      fs.appendFileSync('testing.txt', `the correct rate is ${evaluate} / ${testSize} in epochs ${i} \n`);
      ////////////////////////////////
    }
  }



  updateMiniBatch(mini_batch, eta) {
    //creating an identical array, but has no entry
    let nabla_biases = this.zeroOut(this.biases);
    let nabla_weights = this.zeroOut(this.weights);
    for (let i = 0; i < mini_batch.length; i++) {
      //get the delta we need
      let { delta_nabla_b, delta_nabla_w } = this.backprop(mini_batch[i].input, mini_batch[i].output);
      // update the nabla bias and weight
      nabla_biases = nabla_biases.map((value, index) => {
        return value.add(delta_nabla_b[index]);
      })
      nabla_weights = nabla_weights.map((value, index) => {
        // return value + delta_nabla_w[index];
        return value.add(delta_nabla_w[index]);
      })
    }
    //update 
    this.weights = this.weights.map((value, index) => {
      return value.sub(nabla_weights[index].mul(eta / mini_batch.length))
    })
    this.biases = this.biases.map((layer, index) => {
      return layer.sub(nabla_biases[index].mul(eta / mini_batch.length))
    })

  }


  //learning algorithm
  backprop(input, output) {
    output = tf.tensor(output).expandDims(1);
    let nabla_b = this.zeroOut(this.biases);
    let nabla_w = this.zeroOut(this.weights);
    //feed forward
    let activation = tf.tensor(input).expandDims(1);
    let activations = [activation.div(255)];
    let zs = [] // all the z which is = wa + b
    for (let i = 0; i < this.biases.length; i++) {
      let z = tf.matMul(this.weights[i], activations[i]).add(this.biases[i]);
      zs.push(z);
      activation = this.sigmoid(z);
      activations.push(activation);
    }
    //backward
    // output layer
    let delta = this.cost_derivative(activations[activations.length - 1], output)
    delta = delta.mul(this.sigmoid_prime(zs[zs.length - 1]))
    nabla_b[nabla_b.length - 1] = delta;
    nabla_w[nabla_w.length - 1] = tf.matMul(delta, activations[activations.length - 2].transpose())
    //  For each l=L−1,L−2,…,2 compute δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l).
    for (let i = 2; i < this.num_layers; i++) {
      let z = zs[zs.length - i];
      let sp = this.sigmoid_prime(z);
      //calculate the error
      delta = tf.matMul(this.weights[this.weights.length - i + 1].transpose(), delta).mul(sp);
      //update the gredient;
      nabla_b[nabla_b.length - i] = delta;
      nabla_w[nabla_w.length - i] = tf.matMul(delta, activations[activations.length - i - 1].transpose());
    }
    return { delta_nabla_b: nabla_b, delta_nabla_w: nabla_w };
  }
  cost_derivative(output_activation, expected_output) {
    return output_activation.sub(expected_output);
  }

  //helper functions
  zeroOut(array) {
    return array.map(value => {
      return tf.zeros(value.shape);
    })
  }

  sigmoid(x) {
    return tf.sigmoid(x);
  }

  sigmoid_prime(x) {
    return tf.sigmoid(x).mul(tf.sigmoid(x).mul(-1).add(1));
  }




}


module.exports = Network;


let test = new Network([784, 30, 10])
// console.log(test.biases)
// console.log(test.weights)