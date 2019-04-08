const fs = require('fs');
const app = require('express')();

const imageFileBuffer = fs.readFileSync(__dirname + '/training-cases/train-images.idx3-ubyte')
const labelFileBuffer = fs.readFileSync(__dirname + '/training-cases/train-labels.idx1-ubyte')

const pixelValues = [];

// fs.writeFileSync(__dirname + 'testData.txt', '')


module.exports = function ReadInput() {
  for (let image = 0; image < 59999; image++) {
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
    // imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixel;
    // // used for write to file
    // if (image <= 1000) {
    //   fs.appendFileSync(__dirname + 'testData.txt', `${JSON.stringify(labelFileBuffer[image + 8])} : [${pixel}], `)
    // }

    pixelValues.push(imageData);
  }
  return pixelValues;
}


// app.listen(3001, () => {
//   console.log('finish');
//   console.log(pixelValues)
// })

