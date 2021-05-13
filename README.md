# VLSICircuitFaultDetectionModel-DNN
#### [Link](https://drive.google.com/drive/folders/1DA9-0UTawWD6kTfFg_VaMzuhv6-96djB?usp=sharing) to Project Drive that has all the benchmark circuits and links to the paper.

#### Install the packages
```
pip install -r requirements.txt
```

#### Input ####
The graph inputs must be generated into three files with .x, .graph and .y as suffix to the filename for node features, circuit graph dictionary and output labels respectively.

#### Load input
`dataset = CircuitDataset(path='', circs=[]) `
Here, `path` is the path to the data folder where all circuits resides and
      `circs` name of the circuits filename excluding the suffix. For eg: if circuit file name is adder.x, adder.y and adder.graph, `circs=['adder']`
      
##### Dataset splitted Into train, test and validation
There is library function that generates the disjoint union of graphs based on the CircuitDataset created above. The syntax is:
`train_test_val_data(dataset)`
This returns training, validation and test data loader that can be passed to keras model for training and evaluation.

The detailed notebook with all the steps in present in the root directory. 

## DataSets:
We extracted all the benchmark circuits from [ISCAS](http://www.pld.ttu.ee/~maksim/benchmarks/iscas99/) and [EPL Combinational Benchmark Suite](https://www.epfl.ch/labs/lsi/page-102566-en-html/benchmarks/).

Note: The files must be in the format circuit.x, circuit.graph and circuit.y.

For our experiment, we extracted the features 'logical level', 'controlability-0', 'controlability-1' and 'observality', graph dictionary and circuit labels which are 1(Hard to Observe) and 0(Easy to observe) for each circuit nodes. We dumped input, graph and output to circuit.x, circuit.graph and circuit.y pickle file which is used above for loading the dataset. All the circuits datasets that we generated is the the drive shared above inside output folder.

## GCN references:
- [1] [Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).](https://arxiv.org/abs/1710.10370)
- [2] [Du, Jian, et al. "Topology adaptive graph convolutional networks." arXiv preprint arXiv:1710.10370 (2017)](https://arxiv.org/abs/1710.10370)
- [3] [Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." arXiv preprint arXiv:1606.09375 (2016).](https://arxiv.org/abs/1606.09375)

## Team Members
- Vedika Saravanan
- Nikita Acharya
- Saurav Predhan
