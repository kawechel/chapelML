use DataHandling;
use Random;

config var filename = "contrived.csv";

var (r,c) = inputSize(filename);
var Data: [1..r,1..c] real;

Data = load_csv(filename);

writeln("Data:\n",Data);

var trainsize: int;
var testsize: int;

(_,_,trainsize,testsize) = trainTestShuffleSplit(Data, 0.6, 1);

var Training: [1..trainsize,1..2] real;
var Testing: [1..testsize,1..2] real;

(Training, Testing,_,_) = trainTestShuffleSplit(Data,0.6, 1);

writeln("Training:\n", Training);
writeln("Testing:\n", Testing);
