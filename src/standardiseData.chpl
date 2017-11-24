use DataHandling;
use Math;

/*** Calculate the mean of a column ***/
proc columnMean(A: [?D] real): (real){

  var mean: real;
  var sum: real;
  var n = A.size;

  for i in 1..n{
    sum += A[i];
  }

  mean = sum / n;

  return (mean);
}

/*** Calcuate the standard deviation of a column ***/
proc columnStdev(A: [?D] real, m: real): real{

  var stdev: real;
  var variance: real;
  var n = A.size;

  for i in 1..n {
    variance += ((A[i] - m)**2);
  }

  stdev = sqrt(variance/(n-1));

  return stdev;

}

/*** Standardise the dataset based on the mean and stdev values ***/
proc standardiseDataset(A: [?DA] real, M: [?DM] real, S: [?DS] real){
  var i: int;
  var r = A[..,1].size;
  var c = A[1,..].size;

  for j in 1..r {
    for i in 1..c{
     A[j,i] = (A[j,i] - M[i]) / (S[i]);
   }
  }
}

config var filename = "contrived.csv";

proc main(){

  var (r,c) = inputSize(filename);
  var Data: [1..r,1..c] real;
  var ColMean: [1..c] real;
  var ColStdev: [1..c] real;

  Data = load_csv(filename);

  // get the min and max values for each column of the datasetMinMax
  for i in 1..c {
    ColMean[i] = columnMean(Data[..,i]);
    ColStdev[i] = columnStdev(Data[..,i],ColMean[i]);
  }

  writeln(Data[1,..]);
//  writeln("Mean: ", ColMean);
//  writeln("Stdev: ", ColStdev);

  // standardise the dataset
  standardiseDataset(Data, ColMean, ColStdev);

  writeln(Data[1,..]);
}
