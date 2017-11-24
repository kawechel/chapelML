use Math;
use IO;
use DataHandling;
use Time;

config const quiet: bool = false;
config var filename = "../data/contrived.csv";
config var lr = 0.001;
config var ne = 50;
config var nf = 5;

/*** Estimate coefficients ***/
proc coefficientsSGD(A: [?D] real, lRate: real, nEpoch: int){

  var col = A[1,..].size;
  var row = A[..,1].size;
  var Coeff: [1..col] real;
  var sum_error: real;
  var yhat: real;
  var error: real;

  writeln("*** row=",row,"\n");

  for e in 1..nEpoch{
    sum_error = 0.0;
    for r in 1..row{
      yhat = predict(A[r,..],Coeff);
      error = yhat - A[r,col];
      sum_error += error**2;
      Coeff[1] = Coeff[1] - lRate * error;
      for c in 1..col-1 {
        Coeff[c+1] = Coeff[c+1] - lRate * error * A[r,c];
      }
    }
  }

  return Coeff;
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

proc evaluateAlgorithm(A: [?D] real, nf: int, lRate: real, nEpoch: int) {

  var col = A[1,..].size;
  var row = A[..,1].size;
  var fs = row/nf;
  var Folds: [1..nf,1..fs,1..col] real;
  var Scores: [1..nf] real;

  var seed = 1;

  var testsize = fs;
  var trainsize = fs * (nf-1);

  var Training: [1..trainsize,1..col] real;
  var Testing: [1..testsize,1..col] real;

  Folds = kFoldsShuffleSplit(A, nf, seed);

  for n in 1..nf {

    var idx = 1;
    // create training and testing sets
    for i in 1..nf{
      if n==i then Testing = Folds[i,..,..];
      else {
        Training[idx..idx+fs-1,..] = Folds[i,..,..];
        idx = idx + fs;
      }
    }

    /*writeln("Testing: \n",Testing);
    writeln("Training: \n",Training);*/

    var Predicted: [1..fs] real;
    var Actual: [1..fs] real;
    Actual = Testing[1..fs, col];

    Predicted = linearRegressionSGD(Training, Testing, lRate, nEpoch);

    Scores[n] = rmse_metric(Actual, Predicted);
    //writeln("RMSE = ", Scores[n]);
  }

  return Scores;

}

proc linearRegressionSGD(A: [?DA] real, B: [?DB] real, lRate: real, nEpoch: int) {

  var rows = B[..,1].size;
  var cols = A[1,..].size;
  var Predictions: [1..rows] real;
  var Coeff: [1..cols] real;
  var yhat: real;
  var seed = 1;

  Coeff = coefficientsSGD(A, lRate, nEpoch);

  for i in 1..rows {
    yhat = predict(B[i,..],Coeff);
    Predictions[i] = yhat;
  }

  return Predictions;
}

proc predict(A: [?DA] real, C: [?DC] real): real{

  var yhat = C[1];
  var s = A.size;

  for i in 1..A.size-1 {
    yhat += C[i+1] * A[i];
  }
  return yhat;
}

/*** MAIN ***/
proc main(): int{

  var t: Timer;

  var r,c: int;
  (r,c) = inputSize(filename);
  var Dataset: [1..r, 1..c] real;

  Dataset = load_csv(filename);
  var MinMax: [1..c,1..c] real;

  var Scores: [1..nf] real;

  for i in 1..c {
    MinMax[i,..] = datasetMinMax(Dataset[..,i]);
  }

  /*writeln(Dataset[1,..]);*/

  // normalise the dataset
  normaliseDataset(Dataset, MinMax);

  /*writeln("Coefficients: ", coefficientsSGD(Dataset, lr, ne));*/

  t.start();
  Scores = evaluateAlgorithm(Dataset, nf, lr, ne);
  t.stop();

  writeln("Mean RSME: ", ((+ reduce Scores) / Scores.size));

  if !quiet then
  writeln("Time: ", t.elapsed(TimeUnits.milliseconds), " milliseconds");

  return 0;
}
