use DataHandling;

config var filename = "contrived.csv";

var n: int;
var s: int;

(n,s) = inputSize(filename);
var Dataset: [1..n, 1..(s+1)] real;

Dataset = load_csv(filename);

writeln("Loaded dataset with ", n," rows and ",(s+1)," columns.");
writeln(Dataset[1..5,..]);
