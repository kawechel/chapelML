module DataHandling{

  use Random;

  /*** Get min and max values of a column ***/
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

    writeln("*** Dataset rows = ", r," and colums = ", c," ***");

    for j in 1..r {
      for i in 1..c{
       A[j,i] = (A[j,i] - M[i,1]) / (M[i,2] - M[i,1]);
     }
    }

  }

  /*** Size of input (i.e. number of rows & columns in CSV file) ***/
  /*** returns tuple of number of rows and columns ***/
  proc inputSize(filename: string): (int,int) {

    try!{
      var inputFile = open(filename, iomode.r);
      var readChannel = inputFile.reader();
      var row:string;
      var n = 0;
      var seps = 0;

      while (readChannel.readline(row)){
        n += 1;
        seps = row.count(",");
      }
      return (n,seps+1);
    }
  }

  /*** Load CSV input data ***/
  proc load_csv(filename: string) {

    try!{
      var inputFile = open(filename, iomode.r);
      var readChannel = inputFile.reader();
      var row: string;
      var r, c: int;

      (r, c) = inputSize(filename);

      var Dataset: [1..r, 1..c] real;

      for i in 1..r {
        var j = 1;
        readChannel.readline(row);
        for str in row.split(",", maxsplit=(c-1)){
          Dataset[i,j] = str: real;
          j += 1;
        }
      }
      return Dataset;
    }
  }

  /*** Split input data to training and testing sets ***/
  proc trainTestSplit(A: [?D] real, split: real) {

      var s = A[..,1].size;
      var c = A[1,..].size;
      var ts = ceil(split * s): int;

      var Training = A[1..ts,1..c];
      var Testing = A[ts+1..s,1..c];

      return (Training, Testing, ts, (s-ts));
  }

  /*** Split shuffled input data to training and testing sets ***/
  proc trainTestShuffleSplit(A: [?D] real, split: real, seed: int) {

      var s = A[..,1].size;
      var c = A[1,..].size;
      var ts = ceil(split * s): int;

      // random shuffle the array
      A = shuffleArray(A,seed);

      var Training = A[1..ts,1..c];
      var Testing = A[ts+1..s,1..c];

      return (Training, Testing, ts, (s-ts));
  }

  /*** Split shuffled input data to k folds ***/
  proc kFoldsShuffleSplit(A: [?D] real, f: int, seed: int) {

      var r = A[..,1].size;
      var c = A[1,..].size;
      var fs = r/f : int;
      var Folds: [1..f,1..fs,1..c] real;

      // random shuffle the array
      A = shuffleArray(A,seed);

      for i in 1..f {
        for j in 1..fs {
          Folds[i,j,..] = A[j+(i-1)*fs,..];
        }
      }

      return Folds;
  }

  /*** Shuffle input array based on random permutation of rows ***/
  proc shuffleArray(A: [?D] real, seed: int){

    var r = A[..,1].size;
    var Permute: [1..r] int;
    var Tmp = A;

    permutation(Permute,seed);

    for i in 1..r{
      Tmp[i,..] = A[Permute[i],..];
    }

    return Tmp;

  }

}
