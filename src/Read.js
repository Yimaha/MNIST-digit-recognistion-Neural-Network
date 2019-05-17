const fs = require('fs');

const imageFileBuffer = fs.readFileSync(__dirname + '/training-cases/train-images.idx3-ubyte');
const labelFileBuffer = fs.readFileSync(__dirname + '/training-cases/train-labels.idx1-ubyte');
const testFileBuffer = fs.readFileSync(__dirname + '/training-cases/t10k-images.idx3-ubyte');
const testLabelFileBuffer = fs.readFileSync(__dirname + '/training-cases/t10k-labels.idx1-ubyte');




function readInput() {
  const pixelValues = [];
  for (let image = 0; image < 60000; image++) {
    const pixel = [];
    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        pixel.push(imageFileBuffer[(image) * 28 * 28 + (x + y * 28) + 16])
      }
    }
    let imageData = {};
    let output = new Array(10).fill(0);
    output[labelFileBuffer[image + 8]] = 1;
    imageData.output = output;
    imageData.input = pixel;
    pixelValues.push(imageData);
  }
  return pixelValues;
}

function readTest() {
  const pixelValues = [];
  for (let image = 0; image < 10000; image++) {
    const pixel = [];
    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        pixel.push(testFileBuffer[(image) * 28 * 28 + (x + y * 28) + 16])
      }
    }
    let imageData = {};
    let output = new Array(10).fill(0);
    output[testLabelFileBuffer[image + 8]] = 1;
    imageData.output = output;
    imageData.input = pixel;
    // imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixel;
    // // used for write to file
    // if (image <= 1000) {
    //   fs.appendFileSync(__dirname + 'testData.txt', `${JSON.stringify(labelFileBuffer[image + 8])} : [${pixel}], `)
    // }
    pixelValues.push(imageData);
  }
  return pixelValues;
}


module.exports = {
  readInput,
  readTest
}


