use Math;
use IO;
use DataHandling;
use Time;

config const quiet: bool = false;

/*** Calculate mean of values in array ***/
proc mean(A: [?D] real, col: int): real{

  return (+ reduce A[..,col])/A[..,col].size;

}

/*** Calculate variance of values in array ***/
proc variance(A: [?D] real, col: int, m: real): real{

  var sum: real;
  var mean = m;

  for i in 1..A[..,col].size {
    sum += ((A[i,col] - mean)**2);
  }

  return sum;
}

/*** Calculate covariance of two groups of numbers ***/
proc covariance(A: [?D] real, mx: real, my: real): real{

  var covar: real;

  for i in 1..A[..,1].size {
    covar += (A[i,1] - my) * (A[i,2] - my);
  }

  return covar;

}

/*** Estimate coefficients ***/
proc coefficients(A: [?D] real): (real, real){

  var b0: real;
  var b1: real;
  var mx, my, cv, vx: real;

  mx = mean(A, 1);
  my = mean(A, 2);
  cv = covariance(A, mx, my);
  vx = variance(A, 1, mx);

  b1 = cv / vx;
  b0 = my - b1 * mx;

  return (b0, b1);
}

/*** Calculate Root Mean Squared Error ***/
proc rmse_metric(A: [?DA] real, B: [?DB] real): real{

  var sum_error = 0.0;
  var prediction_error = 0.0;

  for i in 1..A.size {
    prediction_error = B[i] - A[i];
    sum_error += (prediction_error**2);
  }

  var mean_error = sum_error / A.size;
  return sqrt(mean_error);
}

proc evaluateAlgorithm(A: [?D] real, split: real) {

  var trainsize: int;
  var testsize: int;

  (_,_,trainsize,testsize) = trainTestShuffleSplit(A, split,1);

  var Training: [1..trainsize,1..2] real;
  var Testing: [1..testsize,1..2] real;

  (Training, Testing,_,_) = trainTestShuffleSplit(A, split,1);

  var Predicted: [1..testsize] real;
  var Actual: [1..testsize] real;
  Actual = Testing[1..testsize,2];

  Predicted = simpleLinearRegression(Training, Testing);

  var rmse = rmse_metric(Actual, Predicted);
  writeln("RMSE = ", rmse);

}

proc simpleLinearRegression(A: [?DA] real, B: [?DB] real) {

  var rows = B[..,1].size;
  var Predictions: [1..rows] real;
  var yhat: real;
  var b0, b1: real;

  (b0, b1) = coefficients(A);

  for i in 1..B[..,1].size {
      yhat = b0 + b1 * B[i,1];
      Predictions[i] = yhat;
  }

  return Predictions;
}

/*** MAIN ***/
proc main(): int{

  var filename = "contrived.csv";
  var split = 0.6;
  var t: Timer;

  var r,c: int;
  (r,c) = inputSize(filename);
  var Dataset: [1..r, 1..c] real;

  Dataset = load_csv(filename);

  t.start();
  evaluateAlgorithm(Dataset, split);
  t.stop();

  if !quiet then
  writeln("Time: ", t.elapsed(TimeUnits.microseconds), " microseconds");

  return 0;
}
