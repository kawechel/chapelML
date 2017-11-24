use DataHandling;
use ML;

config var filename = "contrived.csv";

var (r,c) = inputSize(filename);
var Data: [1..r,1..c] real;

Data = load_csv(filename);

var Actual = Data[..,1] :real;
var Predicted = Data[..,2] :real;

var correct = accuracy(Actual, Predicted);
writeln(correct,"%");
//confusionMatrix(Actual, Predicted);
writeln(meanAbsoluteError(Actual, Predicted));
writeln(rootMeanSquaredError(Actual, Predicted));
