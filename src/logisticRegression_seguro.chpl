use Math;
use IO;
use DataHandling;
use Time;

config const quiet: bool = false;
config var filename = "contrived";
config var lr = 0.1;
config var ne = 100;

/*** Estimate coefficients ***/
proc coefficientsSGD(A: [?D] real, lRate: real, nEpoch: int, rows: int){

  var col = A[1,..].size;
  var Coeff: [1..col] real;
  var yhat: real;
  var error: real;

  /*writeln("A=",A,"\n");
  writeln("col=",col);
  writeln("row=",row);*/

  for e in 1..nEpoch{
    for r in 1..rows{
      yhat = predict(A[r,..],Coeff);
      error = A[r,col] - yhat;
      Coeff[1] = Coeff[1] + lRate * error * yhat * (1.0 - yhat);
      for c in 1..col-1 {
        Coeff[c+1] = Coeff[c+1] + lRate * error * yhat * (1.0 - yhat) * A[r,c];
      }
    }
  }

  return Coeff;
}

proc evaluateAlgorithm(A: [?DA] real, B: [?DB] real, lRate: real, nEpoch: int, r_te: int) {

  var Training = A;
  var Testing = B;
  var c = Testing[..,1].size;

  var Predicted: [1..c] real;

  /*writeln("Training dataset=\n", Training,"\n");
  writeln("Test dataset=\n", Testing,"\n");*/

  Predicted = logisticRegression(Training, Testing, lRate, nEpoch, r_te);

  /* writeln("**** Predicted: ", Predicted);*/
}

proc logisticRegression(A: [?DA] real, B: [?DB] real, lRate: real, nEpoch: int, r_te: int) {

  var rows = r_te;
  var cols = A[1,..].size;
  var Predictions: [1..rows] real;
  var Coeff: [1..cols] real;
  var yhat: real;

  /*writeln("cols=",cols);
  writeln("rows=",rows);*/

  Coeff = coefficientsSGD(A, lRate, nEpoch, rows);

  for i in 1..rows {
    yhat = predict(B[i,..],Coeff);
    Predictions[i] = yhat;//nearbyint(yhat):int;
    writeln("Predicted=", Predictions[i]);
  }

  return Predictions;
}

proc predict(A: [?DA] real, C: [?DC] real): real{

  var yhat = C[1];
  var s = A.size;

  for i in 1..A.size-1 {
    yhat += C[i+1] * A[i];
  }

  writeln("yhat=",yhat);
  yhat = 1.0 / (1.0 + exp(-yhat));
  writeln("final yhat=",yhat);
  return yhat;
}

/*** MAIN ***/
proc main(): int{

  var t: Timer;

  var r_tr,c_tr,r_te,c_te: int;
  (r_tr,c_tr) = inputSize("../data/contrived_train.csv");
  (r_te,c_te) = inputSize("../data/contrived_test.csv");

  var TrainingDataset: [1..r_tr, 1..c_tr] real;
  var TrainingIDs: [1..r_tr] int;
  var TestDataset: [1..r_te, 1..c_te] real;
  var TestIDs: [1..r_te] int;

  TrainingDataset = load_csv("../data/contrived_train.csv");
  writeln("*** Training dataset loaded");
  TestDataset = load_csv("../data/contrived_test.csv");
  writeln("*** Test dataset loaded");

  writeln("Training dataset has ",r_tr," rows and ",c_tr," columns");
  writeln("Test dataset has ",r_te," rows and ",c_te," columns");

  TrainingIDs = TrainingDataset[..,1]:int;
  TestIDs=TestDataset[..,1]:int;

  var MinMax_tr: [1..c_tr,1..2] real;

  for i in 1..c_tr {
    MinMax_tr[i,..] = datasetMinMax(TrainingDataset[..,2..c_tr-1]);
  }

  writeln("MinMax=", MinMax_tr);
  normaliseDataset(TrainingDataset, MinMax_tr);

  var MinMax_te: [1..c_te,1..2] real;

  for i in 1..c_te {
    MinMax_te[i,..] = datasetMinMax(TestDataset[..,2..c_te]);
  }
  normaliseDataset(TestDataset, MinMax_te);

  t.start();
  evaluateAlgorithm(TrainingDataset[..,2..c_tr], TestDataset[..,2..c_te], lr, ne, r_te);
  t.stop();

  if !quiet then
  writeln("Time: ", t.elapsed(TimeUnits.milliseconds), " milliseconds");

  return 0;
}
