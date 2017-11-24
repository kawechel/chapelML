use DataHandling;

config var filename = "contrived.csv";

var (r,c) = inputSize(filename);
var Data: [1..r,1..c] real;
var f = 4: int;
var fs = r/f;
var Folds: [1..f,1..fs,1..c] real;

Data = load_csv(filename);

writeln("Data:\n",Data);
writeln("==========");


Folds = kFoldsShuffleSplit(Data, f, 2);

writeln("Fold 1: \n", Folds[1,..,..]);
writeln("==========");
writeln("Fold 2: \n", Folds[2,..,..]);
writeln("==========");
writeln("Fold 3: \n", Folds[3,..,..]);
writeln("==========");
writeln("Fold 4: \n", Folds[4,..,..]);
