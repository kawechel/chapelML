module ML{

use Sort;
use Math;

  proc accuracy(Actual: [?DA] real, Predicted: [?DP] real): real {

    var s = Actual.size : real;
    var correct: real;

    for i in 1..Actual.size {
      if (Actual[i] == Predicted[i]) then correct += 1;
    }

    writeln(correct, " correct out of ",s);

    return (correct / s * 100.0);
  }

  proc confusionMatrix(Actual: [?DA] real, Predicted: [?DP] real) {

    var Unique: domain(real);
    var l = Actual.size;

    // first element if always unique
    Unique.add(Actual[1]);

    for i in 2..l {
      for j in 1..i-1 {
        // if actual item is already in list of unique items
        if (Unique.member(Actual[i])) then break;
        else Unique.add(Actual[i]);
      }
    }

    var Lookup: [Unique] int;
    var count = 1;

    for u in Unique {
      Lookup[u] = count;
      count += 1;
    }

    // create Confusion Matrix
    var s = Unique.size;
    var Matrix: [1..s, 1..s] int;
    var x: int;
    var y: int;

    for i in 1..l {
      x = Lookup[Actual[i]];
      y = Lookup[Predicted[i]];
      Matrix[x,y] += 1;
    }

    writeln(Unique: int);
    writeln(" ");
    for i in 1..s {
      writeln(Matrix[i,..]);
    }
  }

  proc meanAbsoluteError(Actual: [?DA] real, Predicted: [?DP] real): real{

    var sum_error: real;
    var l = Actual.size: int;

    for i in 1..l {
      sum_error += abs(Predicted[i] - Actual[i]);
    }

    return (sum_error/l);
  }

  proc rootMeanSquaredError(Actual: [?DA] real, Predicted: [?DP] real): real{

    var sum_error: real;
    var mean_error: real;
    var prediction_error: real;
    var l = Actual.size: int;

    for i in 1..l {
      prediction_error = Predicted[i] - Actual[i];
      sum_error += prediction_error**2;
    }

    mean_error = sum_error/l;

    return sqrt(mean_error);
  }

  proc randomPrediction(Train: [?TR] real, Test: [?TE] real){

  }

}
