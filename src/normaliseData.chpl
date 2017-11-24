use DataHandling;

proc datasetMinMax(A: [?D] real): (real, real){

  var minVal: real;
  var maxVal: real;
  var minLoc: int;
  var maxLoc: int;

  (minVal, minLoc) = minloc reduce zip(A, D);
  (maxVal, maxLoc) = maxloc reduce zip(A, D);

  return (minVal, maxVal);
}

proc normaliseDataset(A: [?DA] real, M: [?DM] real){
  var i: int;
  var r = A[..,1].size;
  var c = A[1,..].size;

  writeln("rows = ", r," and colums = ", c);

  for j in 1..r {
    for i in 1..c{
     A[j,i] = (A[j,i] - M[i,1]) / (M[i,2] - M[i,1]);
   }
  }
}

config var filename = "pima-indians-diabetes.csv";

proc main(){

  var (r,c) = inputSize(filename);
  var Data: [1..r,1..c] real;
  var MinMax: [1..c,1..c] real;

  Data = load_csv(filename);

  // get the min and max values for each column of the datasetMinMax
  for i in 1..c {
    MinMax[i,..] = datasetMinMax(Data[..,i]);
  }

  writeln(Data[1,..]);

  // normalise the dataset
  normaliseDataset(Data, MinMax);

  writeln(Data[1,..]);
}
