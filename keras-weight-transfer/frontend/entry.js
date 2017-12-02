import { sample } from './data'
import { Model } from 'keras-js'

const model = new Model({
  filepaths: {
    model:  'model.json',
    weights: 'model_weights.buf',
    metadata: 'model_metadata.json'
  },
  gpu: true
})

model.ready()
  .then(() => {
    return model.predict({
      'input': new Float32Array(sample)
    })
  })
  .then(outputData => {
    const predictions = outputData['output']
    let max = -1;
    let digit = null;
    for (let i in predictions) {
      let probability = predictions[i];
      if (probability > max) {
        max = probability;
        digit = i;
      }
    }
    document.write(
      "Predicted digit " + digit + " with probability " + max.toFixed(3) + "."
    )
    console.log(outputData)

  })
  .catch(err => {
    console.log(err)
  })
