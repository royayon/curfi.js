# curfi.js

Automated, fast, minimalist Curve-Fitting tool for the browser and [node.js](https://nodejs.org/).

## Demo

Check the [demo site](https://curfi.netlify.app) built with curfi.js.

## Installation

Using npm:

```bash
$ npm install curfi
```

Using CDN:

```html
<script src="https://unpkg.com/curfi@1.0.3/curfi.min.js"></script>
```

## Usage

### CommonJS usage

```js
const curfi = require("curfi");

// create an instance of curfi
let model = new curfi();

// dataset
//   xi, yi
//   1, 0.5
//   2, 2.5
//   3, 2.0
//   4, 4.0
//   5, 3.5
//   6, 6.0
//   7, 5.5
let trainX = [[1, 2, 3, 4, 5, 6, 7]];
let trainY = [[0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5]];

// auto fit the curve for the dataset
model.AutoTrain(trainX, trainY, null, null, 3);

// model.predict() for prediction
let prediction = model.predict([[8]]);
console.log(prediction); //  [[ 6.428571428569853 ]]
```

### Browser usage

```html
<!DOCTYPE html>
<html>
  <head></head>

  <body>
    <script src="https://unpkg.com/curfi/curfi.min.js"></script>
    <script>
      // create an instance of curfi
      let model = new curfi();

      // dataset
      let trainX = [[1, 2, 3, 4, 5, 6, 7]];
      let trainY = [[0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5]];
      let testX = [[1, 8]];
      let testY = [[0.762, 7.265]];

      // auto fit the curve for the dataset
      model.AutoTrain(trainX, trainY, testX, testY, 3);

      // model.predict() for prediction
      console.log(model.predict([[8]])); // //  [[ 6.428571428569853 ]]
    </script>
  </body>
</html>
```

### R2_Score

To check how good a model is:

```js
model.r2_score(y_true, y_pred);
```

### Save Model

To save a trained model for reuse

```js
model.saveModel(); // creates a download in browser
```

Save model with a model name

```js
model.saveModel("myModel"); // will download myModel.json
```

### Load Model

Load a saved model

```js
model.loadModel("myModel.json"); // file reader api to open the .json file
```

### Extra

```js
let cf = new curfi();

cf.modelEqnnHTML(); // returns HTML of the curve equation

cf.round(num, digits); // Round a number up to digits after decimal

cf.round3(num); // Round a number up to 3 digits after decimal

cf.round2(num); // Round a number up to 2 digits after decimal

cf.matrix_multiply(a, b); // returns Matrix Multiply of a and b

cf.matrix_transpose(a); // returns Matrix Transpose of a

cf.matrix_invert(a); // returns Matrix Inverse of a
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
