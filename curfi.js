class Curfi {
    constructor() {
        this.modelName = '';
        this.weights = [];
        this.weightsLen = 0;
        this.inputShape = [];
        this.outputShape = [];
        this.accuracy = {
            r2_score: 0,
        };
        this.additionalParams = {};
    }

    modelParams(modelName, weights, inputShape, outputShape) {
        this.weights = [...weights];
        this.modelName = modelName;
        this.weightsLen = weights.length;
        this.inputShape = inputShape;
        this.outputShape = outputShape;
    }

    modelAccuracy(accuracy) {
        this.accuracy = { ...accuracy };
    }

    modelAdditionalParams(params) {
        this.additionalParams = { ...params };
    }

    loadModel(modelc) {
        this.modelParams(modelc.modelName, modelc.weights, modelc.inputShape, modelc.outputShape);
        this.modelAdditionalParams(modelc.additionalParams);
        this.modelAccuracy(modelc.accuracy);
        return this;
    }

    saveModel(exportName = 'model') {
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({ ...this }));
        var downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", exportName + ".json");
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    // Failed to inverse '-111'

    fit_AllModels(trainX, trainY, highestOrder = 3) {
        let models = {
            singleX: {},
            multiX: {}
        };
        // For Multiple X variables
        if (trainX.length >= 1) {
            models.multiX.MLR = { ...this.fit_MLR(trainX, trainY) };
            models.multiX.EXP = { ...this.fit_EXP(trainX, trainY) };
            models.multiX.LogLR = { ...this.fit_LogLR(trainX, trainY) };
        }
        // For Single X variable
        if (trainX.length == 1) {
            models.singleX.LR = { ...this.fit_LR(trainX, trainY) };
            models.singleX.PLR = { ...this.fit_PLR(trainX, trainY, highestOrder) };
            models.singleX.SinLR = { ...this.fit_SinLR(trainX, trainY) };
        }

        function clean(obj) {
            for (let propNameo in obj) {
                for (let propName in obj[propNameo]) {
                    if (Object.keys(obj[propNameo][propName]).length === 0 && obj[propNameo][propName].constructor === Object || obj[propNameo][propName] === null || obj[propNameo][propName] === undefined) {
                        delete obj[propNameo][propName];
                    }
                    else if (Number.isNaN(obj[propNameo][propName].weights[0][0])) {
                        delete obj[propNameo][propName];
                    }
                }
            }
            return obj
        }

        clean(models);
        return { ...models };
    }


    // Linear Regression functions
    fit_LR(trainX, trainY) {
        return this.fit_LinearRegression(trainX, trainY);
    }
    fit_LinearRegression(trainX, trainY) {
        let obj = this.fit_MultipleLinearRegression(trainX, trainY);
        let A = [...obj.weights];
        this.modelParams("LinearRegression", [...A], trainX.length, trainY.length);
        let accuracy = {
            r2_score: obj.accuracy.r2_score,
        };
        this.modelAccuracy(accuracy);
        return this;
    }
    // MultipleLinearRegression Functions
    fit_MLR(trainX, trainY) {
        return this.fit_MultipleLinearRegression(trainX, trainY);
    }

    fit_MultipleLinearRegression(trainX, trainY) {
        // no of data points, no of rows
        let n = trainX[0].length;
        // all 1s array
        let x0 = new Array(n).fill(1);
        let x = [...trainX];
        let y = [...trainY];
        // inserting xo(all 1s) at the beginning
        x.unshift(x0);
        // order = m = columns in the dataset
        let order = trainX.length;
        let a = [...Array(order + 2)].map(e => Array(order + 2).fill(0));
        // let a = [...aa];

        // Coefficient matrix calculation A[]
        for (let i = 1; i <= order + 1; i++) {
            for (let j = 1; j <= i; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += x[i - 1][l] * x[j - 1][l];
                }
                a[i][j] = sum;
                a[j][i] = sum;
            }
            let sum = 0;
            for (let l = 0; l < n; l++) {
                sum += y[0][l] * x[i - 1][l];
            }
            a[i][order + 2] = sum;
        }

        // Only the coefficients and not the Y part
        let Z = [];
        let Y = [];
        for (let z = 1; z < a.length; z++) {
            let zzz = [];
            let yyy = [];
            for (let zz = 1; zz < a[z].length; zz++) {
                if (zz != a[z].length - 1) {
                    zzz.push(a[z][zz]);
                }
                // order+2 are the Y matrix
                if (zz == a[z].length - 1) {
                    yyy.push(a[z][zz]);
                }
            }
            Z.push(zzz);
            Y.push(yyy);
        }
        // console.log(a, Z, Y)
        // calculation for A[] from A =  INV(Z) * Y
        let Zinv = this.matrix_invert(Z);
        if (Zinv === -111) { return {}; }
        // console.log(Zinv)
        let A = this.matrix_multiply(Zinv, Y);
        // console.log(A)

        // set this models parameters
        this.modelParams("MultipleLinearRegression", [...A], trainX.length, trainY.length);
        let accuracy = {
            r2_score: this.r2_score(trainY[0], this.predict(trainX)[0]),
        };
        this.modelAccuracy(accuracy);
        return this;
    }


    // Exponential LinearRegression Functions
    fit_EXP(trainX, trainY) {
        return this.fit_ExpLinearRegression(trainX, trainY);
    }

    fit_ExpLinearRegression(trainX, trainY) {
        // no of data points, no of rows
        let n = trainX[0].length;
        // all 1s array
        let x0 = new Array(n).fill(1);
        let isNaN = 0;
        let x = [...trainX].map(el1 => el1.map(el2 => { if (Math.exp(el2) == Number.POSITIVE_INFINITY || Math.exp(el2) == Number.NEGATIVE_INFINITY) { isNaN = 1; } return Math.exp(el2); }));
        if (isNaN) {
            return {};
        }
        let y = [...trainY];
        // inserting xo(all 1s) at the beginning
        x.unshift(x0);
        // order = m = columns in the dataset
        let order = trainX.length;
        let a = [...Array(order + 2)].map(e => Array(order + 2).fill(0));
        // let a = [...aa];

        // Coefficient matrix calculation A[]
        for (let i = 1; i <= order + 1; i++) {
            for (let j = 1; j <= i; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += x[i - 1][l] * x[j - 1][l];
                }
                a[i][j] = sum;
                a[j][i] = sum;
            }
            let sum = 0;
            for (let l = 0; l < n; l++) {
                sum += y[0][l] * x[i - 1][l];
            }
            a[i][order + 2] = sum;
        }

        // Only the coefficients and not the Y part
        let Z = [];
        let Y = [];
        for (let z = 1; z < a.length; z++) {
            let zzz = [];
            let yyy = [];
            for (let zz = 1; zz < a[z].length; zz++) {
                if (zz != a[z].length - 1) {
                    zzz.push(a[z][zz]);
                }
                // order+2 are the Y matrix
                if (zz == a[z].length - 1) {
                    yyy.push(a[z][zz]);
                }
            }
            Z.push(zzz);
            Y.push(yyy);
        }
        // console.log(a, Z, Y)
        // calculation for A[] from A =  INV(Z) * Y
        let Zinv = this.matrix_invert(Z);
        if (Zinv === -111) { return {}; }
        // console.log(Zinv)
        let A = this.matrix_multiply(Zinv, Y);
        // console.log(A)

        // set this models parameters
        this.modelParams("ExponentialLinearRegression", [...A], trainX.length, trainY.length);
        let accuracy = {
            r2_score: this.r2_score(trainY[0], this.predict(trainX)[0]),
        };
        this.modelAccuracy(accuracy);
        return this;
    }

    // Polynomial LinearRegression Functions
    fit_PLR(trainX, trainY, order = 3) {
        return this.fit_PolynomialLinearRegression(trainX, trainY, order);
    }

    fit_PolynomialLinearRegression(trainX, trainY, order = 3) {
        // no of data points, no of rows
        let n = trainX[0].length;
        // all 1s array
        let x0 = new Array(n).fill(1);
        let x = [...trainX];
        let y = [...trainY];
        // inserting xo(all 1s) at the beginning
        x.unshift(x0);
        // order = m = columns in the dataset
        //let order = trainX.length;
        let a = [...Array(order + 2)].map(e => Array(order + 2).fill(0));
        // let a = [...aa];
        // Coefficient matrix calculation A[]
        for (let i = 1; i <= order + 1; i++) {
            for (let j = 1; j <= i; j++) {
                let k = i + j - 2;
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += Math.pow(x[1][l], k);
                }
                a[i][j] = sum;
                a[j][i] = sum;
            }
            let sum = 0;
            for (let l = 0; l < n; l++) {
                sum += y[0][l] * Math.pow(x[1][l], i - 1);
            }
            a[i][order + 2] = sum;
        }
        // Only the coefficients and not the Y part
        let Z = [];
        let Y = [];
        for (let z = 1; z < a.length; z++) {
            let zzz = [];
            let yyy = [];
            for (let zz = 1; zz < a[z].length; zz++) {
                if (zz != a[z].length - 1) {
                    zzz.push(a[z][zz]);
                }
                // order+2 are the Y matrix
                if (zz == a[z].length - 1) {
                    yyy.push(a[z][zz]);
                }
            }
            Z.push(zzz);
            Y.push(yyy);
        }
        // console.log(a, Z, Y)
        // calculation for A[] from A =  INV(Z) * Y
        let Zinv = this.matrix_invert(Z);
        if (Zinv === -111) { return {}; }
        // console.log(Zinv)
        let A = this.matrix_multiply(Zinv, Y);
        // set this models parameters
        this.modelParams("PolynomialLinearRegression", [...A], trainX.length, trainY.length);
        this.modelAdditionalParams({ order });
        let accuracy = {
            r2_score: this.r2_score(trainY[0], this.predict(trainX)[0]),
        };
        this.modelAccuracy(accuracy);
        return this;
    }

    // Logarithmic LinearRegression Functions
    fit_LogLR(trainX, trainY) {
        return this.fit_LogarithmicLinearRegression(trainX, trainY);
    }

    fit_LogarithmicLinearRegression(trainX, trainY) {
        // no of data points, no of rows
        let n = trainX[0].length;
        // if any zero in x then log will be NaN
        // In general, the function y=logbx where b is base,x>0 and b≠1 is a continuous and one-to-one function. Note that the logarithmic functionis not defined for negative numbers or for zero. The graph of the function approaches the y -axis as x tends to ∞ , but never touches it.

        let isZero = 0;
        // all 1s array
        let x0 = new Array(n).fill(1);
        let x = [...trainX].map(el1 => el1.map(el2 => { if (el2 <= 0) { isZero = 1; } return Math.log(el2); }));
        if (isZero) {
            return {};
        }
        let y = [...trainY];
        // inserting xo(all 1s) at the beginning
        x.unshift(x0);
        // order = m = columns in the dataset
        let order = trainX.length;
        let a = [...Array(order + 2)].map(e => Array(order + 2).fill(0));
        // let a = [...aa];

        // Coefficient matrix calculation A[]
        for (let i = 1; i <= order + 1; i++) {
            for (let j = 1; j <= i; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += x[i - 1][l] * x[j - 1][l];
                }
                a[i][j] = sum;
                a[j][i] = sum;
            }
            let sum = 0;
            for (let l = 0; l < n; l++) {
                sum += y[0][l] * x[i - 1][l];
            }
            a[i][order + 2] = sum;
        }

        // Only the coefficients and not the Y part
        let Z = [];
        let Y = [];
        for (let z = 1; z < a.length; z++) {
            let zzz = [];
            let yyy = [];
            for (let zz = 1; zz < a[z].length; zz++) {
                if (zz != a[z].length - 1) {
                    zzz.push(a[z][zz]);
                }
                // order+2 are the Y matrix
                if (zz == a[z].length - 1) {
                    yyy.push(a[z][zz]);
                }
            }
            Z.push(zzz);
            Y.push(yyy);
        }
        // console.log(a, Z, Y)
        // calculation for A[] from A =  INV(Z) * Y
        let Zinv = this.matrix_invert(Z);
        if (Zinv === -111) { return {}; }
        // console.log(Zinv)
        let A = this.matrix_multiply(Zinv, Y);
        // console.log(A)

        // set this models parameters
        this.modelParams("LogarithmicLinearRegression", [...A], trainX.length, trainY.length);
        let accuracy = {
            r2_score: this.r2_score(trainY[0], this.predict(trainX)[0]),
        };
        this.modelAccuracy(accuracy);
        return this;
    }

    // Sinusoidal Regression Functions
    fit_SinLR(trainX, trainY) {
        return this.fit_SinusoidalRegression(trainX, trainY);
    }

    fit_SinusoidalRegression(trainX, trainY) {
        // no of data points, no of rows
        let n = trainX[0].length;
        // all 1s array
        let x0 = new Array(n).fill(1);
        let x = [...trainX].map(el1 => el1.map(el2 => Math.sin(el2 * (Math.PI / 180))));
        let y = [...trainY];
        // inserting xo(all 1s) at the beginning
        x.unshift(x0);
        x.push(trainX[0].map(el => Math.cos(el * (Math.PI / 180))));
        // order = m = columns in the dataset
        let order = x.length - 1;
        let a = [...Array(order + 2)].map(e => Array(order + 2).fill(0));
        // let a = [...aa];

        // Coefficient matrix calculation A[]
        for (let i = 1; i <= order + 1; i++) {
            for (let j = 1; j <= i; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += x[i - 1][l] * x[j - 1][l];
                }
                a[i][j] = sum;
                a[j][i] = sum;
            }
            let sum = 0;
            for (let l = 0; l < n; l++) {
                sum += y[0][l] * x[i - 1][l];
            }
            a[i][order + 2] = sum;
        }

        // Only the coefficients and not the Y part
        let Z = [];
        let Y = [];
        for (let z = 1; z < a.length; z++) {
            let zzz = [];
            let yyy = [];
            for (let zz = 1; zz < a[z].length; zz++) {
                if (zz != a[z].length - 1) {
                    zzz.push(a[z][zz]);
                }
                // order+2 are the Y matrix
                if (zz == a[z].length - 1) {
                    yyy.push(a[z][zz]);
                }
            }
            Z.push(zzz);
            Y.push(yyy);
        }
        // console.log(a, Z, Y)
        // calculation for A[] from A =  INV(Z) * Y
        let Zinv = this.matrix_invert(Z);
        if (Zinv === -111) { return {}; }
        // console.log(Zinv)
        let A = this.matrix_multiply(Zinv, Y);
        // console.log(A)
        let A1 = Math.sqrt(Math.pow(A[1][0], 2) + Math.pow(A[2][0], 2)); // the amplitude
        let delta = Math.atan(A[2][0] / A[1][0]) * (180 / Math.PI); // phase shift
        A[1][0] = A1;
        A[2][0] = delta;

        // So the final sinusoidal model will be in this form:
        // y = A0 + A1 sin (x + delta) where A0, A1 and delta are the three parameters

        // set this models parameters
        this.modelParams("SinusoidalRegression", [...A], trainX.length, trainY.length);
        let accuracy = {
            r2_score: this.r2_score(trainY[0], this.predict(trainX)[0]),
        };
        this.modelAccuracy(accuracy);
        return this;
    }

    AutoTrain(trainX, trainY, testX = null, testY = null, highestOrder = 3) {
        let models = this.fit_AllModels(trainX, trainY, highestOrder);
        let multimodels = { ...models.multiX };
        let singlemodels = { ...models.singleX };
        let allmodels = { ...multimodels, ...singlemodels };

        let sortedModel = [];
        Object
            .keys(allmodels).sort(function (a, b) {
                return allmodels[b].accuracy.r2_score - allmodels[a].accuracy.r2_score;
            })
            .forEach(function (key) {
                if (testY === null) {
                    let obj = new Curfi();
                    obj.loadModel(allmodels[key]);

                    sortedModel.push(obj);
                } else if (testX !== null && testY !== null) {
                    let obj = new Curfi();
                    obj.loadModel(allmodels[key]);
                    obj.accuracy.r2_score_test = obj.r2_score(testY[0], obj.predict(testX)[0]);

                    sortedModel.push(obj);
                }
            });
        this.loadModel(sortedModel[0]);
        return sortedModel;
    }


    // textX is in columnwise
    predict(testX) {
        let wt = [...this.weights];
        let testY = [];
        for (let r = 0; r < testX[0].length; r++) {
            let sum = wt[0][0];
            if (this.modelName === "PolynomialLinearRegression") {
                for (let c = 0; c < this.additionalParams.order; c++) {
                    sum += wt[c + 1][0] * this.coefficientFunction(testX[0][r], c + 1);
                }
            } else {
                for (let c = 0; c < testX.length; c++) {
                    sum += wt[c + 1][0] * this.coefficientFunction(testX[c][r], c + 1);
                }
            }
            testY.push(sum);
        }
        return [[...testY]];
    }

    coefficientFunction(val, pos) {
        switch (this.modelName) {
            case "LinearRegression":
                return val;
                break;
            case "PolynomialLinearRegression":
                return Math.pow(val, pos);
                break;
            case "MultipleLinearRegression":
                return val;
                break;
            case "ExponentialLinearRegression":
                return Math.exp(val);
                break;
            case "LogarithmicLinearRegression":
                return Math.log(val);
                break;
            case "SinusoidalRegression":
                return Math.sin((val + this.weights[pos + 1][0]) * Math.PI / 180);
                break;

            default:
                return val;
                break;
        }
    }

    // r2 value function
    r2_score(y_true, y_pred) {
        let numOr0 = n => isNaN(n) ? 0 : n;
        let y_true_Sum = y_true.reduce((a, b) => numOr0(a) + numOr0(b));
        let y_true_Mean = y_true_Sum / y_true.length;

        let St = 0;
        let Sr = 0;
        for (let yi = 0; yi < y_true.length; yi++) {
            St += (y_true[yi] - y_true_Mean) * (y_true[yi] - y_true_Mean);
            Sr += (y_true[yi] - y_pred[yi]) * (y_true[yi] - y_pred[yi]);
        }
        return (St - Sr) / St;
    }

    // Round upto digits after decimal
    round(num, digits) {
        return Math.round((num + Number.EPSILON) * Math.pow(10, digits)) / Math.pow(10, digits);
    }
    // Round up to 3 digits after decimal
    round3(num) {
        return Math.round((num + Number.EPSILON) * 1000) / 1000;
    }
    // Round up to 2 digits after decimal
    round2(num) {
        return Math.round((num + Number.EPSILON) * 100) / 100;
    }

    // Matrix Functions
    matrix_multiply(a, b) {
        var aNumRows = a.length, aNumCols = a[0].length,
            bNumRows = b.length, bNumCols = b[0].length,
            m = new Array(aNumRows);  // initialize array of rows
        for (var r = 0; r < aNumRows; ++r) {
            m[r] = new Array(bNumCols); // initialize the current row
            for (var c = 0; c < bNumCols; ++c) {
                m[r][c] = 0;             // initialize the current cell
                for (var i = 0; i < aNumCols; ++i) {
                    m[r][c] += a[r][i] * b[i][c];
                }
            }
        }
        return m;
    }

    matrix_transpose(a) {

        // Calculate the width and height of the Array
        var w = a.length || 0;
        var h = a[0] instanceof Array ? a[0].length : 0;

        // In case it is a zero matrix, no transpose routine needed.
        if (h === 0 || w === 0) { return []; }

        /**
         * @var {Number} i Counter
         * @var {Number} j Counter
         * @var {Array} t Transposed data is stored in this array.
         */
        var i, j, t = [];

        // Loop through every item in the outer array (height)
        for (i = 0; i < h; i++) {

            // Insert a new row (array)
            t[i] = [];

            // Loop through every item per item in outer array (width)
            for (j = 0; j < w; j++) {

                // Save transposed data.
                t[i][j] = a[j][i];
            }
        }

        return t;
    }


    // Returns the inverse of matrix `M`.
    matrix_invert(M) {
        // I use Guassian Elimination to calculate the inverse:
        // (1) 'augment' the matrix (left) by the identity (on the right)
        // (2) Turn the matrix on the left into the identity by elemetry row ops
        // (3) The matrix on the right is the inverse (was the identity matrix)
        // There are 3 elemtary row ops: (I combine b and c in my code)
        // (a) Swap 2 rows
        // (b) Multiply a row by a scalar
        // (c) Add 2 rows

        //if the matrix isn't square: exit (error)
        if (M.length !== M[0].length) { return -111; }

        //create the identity matrix (I), and a copy (C) of the original
        var i = 0, ii = 0, j = 0, dim = M.length, e = 0, t = 0;
        var I = [], C = [];
        for (i = 0; i < dim; i += 1) {
            // Create the row
            I[I.length] = [];
            C[C.length] = [];
            for (j = 0; j < dim; j += 1) {

                //if we're on the diagonal, put a 1 (for identity)
                if (i == j) { I[i][j] = 1; }
                else { I[i][j] = 0; }

                // Also, make the copy of the original
                C[i][j] = M[i][j];
            }
        }

        // Perform elementary row operations
        for (i = 0; i < dim; i += 1) {
            // get the element e on the diagonal
            e = C[i][i];

            // if we have a 0 on the diagonal (we'll need to swap with a lower row)
            if (e == 0) {
                //look through every row below the i'th row
                for (ii = i + 1; ii < dim; ii += 1) {
                    //if the ii'th row has a non-0 in the i'th col
                    if (C[ii][i] != 0) {
                        //it would make the diagonal have a non-0 so swap it
                        for (j = 0; j < dim; j++) {
                            e = C[i][j];       //temp store i'th row
                            C[i][j] = C[ii][j];//replace i'th row by ii'th
                            C[ii][j] = e;      //repace ii'th by temp
                            e = I[i][j];       //temp store i'th row
                            I[i][j] = I[ii][j];//replace i'th row by ii'th
                            I[ii][j] = e;      //repace ii'th by temp
                        }
                        //don't bother checking other rows since we've swapped
                        break;
                    }
                }
                //get the new diagonal
                e = C[i][i];
                //if it's still 0, not invertable (error)
                if (e == 0) { return -111; }
            }

            // Scale this row down by e (so we have a 1 on the diagonal)
            for (j = 0; j < dim; j++) {
                C[i][j] = C[i][j] / e; //apply to original matrix
                I[i][j] = I[i][j] / e; //apply to identity
            }

            // Subtract this row (scaled appropriately for each row) from ALL of
            // the other rows so that there will be 0's in this column in the
            // rows above and below this one
            for (ii = 0; ii < dim; ii++) {
                // Only apply to other rows (we want a 1 on the diagonal)
                if (ii == i) { continue; }

                // We want to change this element to 0
                e = C[ii][i];

                // Subtract (the row above(or below) scaled by e) from (the
                // current row) but start at the i'th column and assume all the
                // stuff left of diagonal is 0 (which it should be if we made this
                // algorithm correctly)
                for (j = 0; j < dim; j++) {
                    C[ii][j] -= e * C[i][j]; //apply to original matrix
                    I[ii][j] -= e * I[i][j]; //apply to identity
                }
            }
        }

        //we've done all operations, C should be the identity
        //matrix I should be the inverse:
        return I;
    }

    modelEqnnHTML(model = this, rnd = this.round3) {
        this.loadModel(model);
        let ystr = '';
        switch (this.modelName) {
            case "LinearRegression":
                ystr = `y = (${rnd(this.weights[0][0])})`;
                for (let a = 1; a < this.weightsLen; a++) {
                    ystr += ` + (${rnd(this.weights[a][0])}) x`;
                }
                return ystr;
                break;
            case "PolynomialLinearRegression":
                ystr = `y = (${rnd(this.weights[0][0])})`;
                for (let a = 1; a < this.weightsLen; a++) {
                    ystr += ` + (${rnd(this.weights[a][0])}) x<sup>${a}</sup>`;
                }
                return ystr;
                break;
            case "MultipleLinearRegression":
                ystr = `y = (${rnd(this.weights[0][0])})`;
                for (let a = 1; a < this.weightsLen; a++) {
                    ystr += ` + (${rnd(this.weights[a][0])}) x<sub>${a}</sub>`;
                }
                return ystr;
                break;
            case "ExponentialLinearRegression":
                ystr = `y = (${rnd(this.weights[0][0])})`;
                for (let a = 1; a < this.weightsLen; a++) {
                    ystr += ` + (${rnd(this.weights[a][0])}) e<sup>x<sub>${a}</sub></sup>`;
                }
                return ystr;
                break;
            case "LogarithmicLinearRegression":
                ystr = `y = (${rnd(this.weights[0][0])})`;
                for (let a = 1; a < this.weightsLen; a++) {
                    ystr += ` + (${rnd(this.weights[a][0])}) log(x<sub>${a}</sub>)`;
                }
                return ystr;
                break;
            case "SinusoidalRegression":
                ystr = `y = (${rnd(this.weights[0][0])}) + (${rnd(this.weights[1][0])}) Sin(x + (${rnd(this.weights[2][0])}))`;
                return ystr;
                break;

            default:
                return ystr = `Couldn't Create Equation 🌋`;
                break;
        }
    }
}

// class curfi extends Curfi {
//     constructor() {
//         super();
//     }
// }
if (typeof exports != "undefined") {
    module.exports = Curfi;
} else {
    var curfi = Curfi;
}


