use DataHandling;
use Time;

config const quiet: bool = false;
config var filename = "../data/contrived.csv";
config var lRate = 0.1;
config var nEpoch = 500;
config var nFolds = 3;

proc accuracyMetric(A: [?DA] real, P: [?DP] real): real{

  var correct = 0;
  for i in 1..A.size {
    if A[i] == P[i] then correct += 1;
  }
  /*writeln("correct preditions=",correct);*/

  return correct / A.size:real * 100.0;

}

proc predict(Row: [?DR] real, W: [?DW] real): real {
  var activation = W[1];
  var length = Row.size;
  for l in 1..length {
    activation += W[l+1] * Row[l];
  }
  return if activation >= 0.0 then 1.0 else 0.0;
}

proc trainWeights(A: [?D] real, lr: real, ne: int){
  var r = A[..,1].size;
  var c = A[1,..].size;
  var prediction: real;
  var error: real;
  var sum_error: real;

  var Weigths: [1..c] real;

  for n in 1..ne {
    sum_error = 0.0;
    for i in 1..r{
      prediction = predict(A[i,..], Weigths);
      error = A[i,c] - prediction;
      sum_error += error**2;
      Weigths[1] = Weigths[1] + lr * error;
      for j in 1..c-1 {
        Weigths[j+1] = Weigths[j+1] + lr * error * A[i,j];
      }
    }
    //writeln("epoch=",n,", lrate=",lr,", error=",sum_error);
  }
  return Weigths;
}

proc perceptron(Train: [?DTr] real, Test: [?DTe] real, lr: real, ne: int){

    var nrows = Test[..,1].size;
    var P: [1..nrows] real;

    var W = trainWeights(Train, lRate, nEpoch);

    for r in 1..nrows {
      P[r] = predict(Test[r,..],W);
      //writeln("P[",r,"]=",P[r]);
    }

    return P;
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

    var Predicted: [1..fs] real;
    var Actual: [1..fs] real;

    Predicted = perceptron(Training, Testing, lRate, nEpoch);
    Actual = Testing[1..fs, col];

    Scores[n] = accuracyMetric(Actual, Predicted);
    //writeln("RMSE = ", Scores[n]);
  }

  writeln("Scores=",Scores);
  writeln("Mean accuracy=", ((+ reduce Scores) / Scores.size),"%");

}

proc main(){

  var t: Timer;

  var (r,c) = inputSize(filename);
  var Dataset: [1..r,1..c] real;

  Dataset = load_csv(filename);

  t.start();
  evaluateAlgorithm(Dataset, nFolds, lRate, nEpoch);
  t.stop();

  if !quiet then
  writeln("Time: ", t.elapsed(TimeUnits.milliseconds), " milliseconds");

  return 0;
}
