use Math;
use IO;
use Time;

config const quiet: bool = false;


/*** Size of input (i.e. number of rows in CSV file) ***/
proc inputSize(filename: string): int {

  var inputFile = open(filename, iomode.r);
  var readChannel = inputFile.reader();
  var row:string;
  var count = 0;

  while (readChannel.readline(row)){
    count += 1;
  }
  return count;
}

/*** Load input data ***/
proc load_csv(filename: string) {

    var inputFile = open(filename, iomode.r);
    var readChannel = inputFile.reader();
    var row:string;
    var n = inputSize(filename);

    var Rows: [1..n] string;
    var Dataset: [1..n, 1..2] real;

    var r = 1;

    while (readChannel.readline(row)){
      Rows[r]=row;
      r += 1;
    }

    for i in 1..Rows.size {
      (Dataset[i,1],_,Dataset[i,2]) = Rows[i].partition(",") : (real, string, real);
    }

    return Dataset[1..n,1..2];
}

/*** Split input data to training and testing ***/
proc train_test_split(A: [?D] real, split: real) {

    var s = A[..,1].size;
    var ts = ceil(split * s): int;

    var Training = A[1..ts,1..2];
    var Testing = A[ts+1..s,1..2];

    return (Training, Testing, ts, (s-ts));
}

/*** Calculate mean of values in array ***/
proc mean(A: [?D] real, col: int): real{

  return (+ reduce A[..,col])/A[..,col].size;

}

/*** Calculate variance of values in array ***/
proc variance(A: [?D] real, col: int, m: real): real{

  var sum: real;
  var mean = m;

  coforall i in 1..A[..,col].size with (ref sum){
    sum += ((A[i,col] - mean)**2);
  }

  return sum;
}

/*** Calculate covariance of two groups of numbers ***/
proc covariance(A: [?D] real, mx: real, my: real): real{

  var covar: real;

  coforall i in 1..A[..,1].size with (ref covar) {
    covar += (A[i,1] - my) * (A[i,2] - my);
  }

  return covar;
}

/*** Estimate coefficients ***/
proc coefficients(A: [?D] real): (real, real){

  var b0: real;
  var b1: real;
  var mx, my, cv, vx: real;

  cobegin with (ref mx, ref my){
    mx = mean(A, 1);
    my = mean(A, 2);
  }

  cobegin with (ref cv, ref vx){
    cv = covariance(A, mx, my);
    vx = variance(A, 1, mx);
  }

  b1 = cv / vx;
  b0 = my - b1 * mx;

  return (b0, b1);
}

/*** Calculate Roomt Mean Squared Error ***/
proc rmse_metric(A: [?DA] real, B: [?DB] real): real{

  var sum_error = 0.0;
  var prediction_error = 0.0;

  forall i in 1..A.size with (ref prediction_error, ref sum_error){
    prediction_error = B[i] - A[i];
    sum_error += (prediction_error**2);
  }

  var mean_error = sum_error / A.size;
  return sqrt(mean_error);
}


proc evaluateAlgorithm(A: [?D] real, split: real) {

  var trainsize: int;
  var testsize: int;

  (_,_,trainsize,testsize) = train_test_split(A, split);

  var Training: [1..trainsize,1..2] real;
  var Testing: [1..testsize,1..2] real;

  (Training, Testing,_,_) = train_test_split(A, split);

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
  var b0, b1: real;

  (b0, b1) = coefficients(A);

  coforall i in 1..B[..,1].size {
      var yhat = b0 + b1 * B[i,1];
      Predictions[i] = yhat;
  }

  return Predictions;
}

/*** MAIN ***/
proc main(): int{

  var filename = "contrived.csv";
  var split = 0.7;
  var t: Timer;

  var s = inputSize(filename);
  var Dataset: [1..s, 1..2] real;

  Dataset = load_csv(filename);

  t.start();
  evaluateAlgorithm(Dataset, split);
  t.stop();

  if !quiet then
  writeln("Time: ", t.elapsed(TimeUnits.microseconds), " microseconds");

  return 0;
}
