# LOSTIN: <ins>L</ins>ogic <ins>O</ins>ptimization via <ins>S</ins>patio-<ins>T</ins>emporal <ins>In</ins>formation with Hybrid Graph Models

## Benchmark

All considered designs come from [EPFL logic synthesis benchmark suite](https://github.com/lsils/benchmarks)
- 9 designs from the arithmetic benchmarks
- 2 designs from the random/control benchmarks

## Data Preprocessing
### From verilog to graph
[verilog2graph](https://github.com/lydiawunan/LOSTIN/tree/main/verilog2graph) parses verilog designs into graphs. 

Specifically,
- `verilog_cleanser.py` removes the symbols incompatible to our parser from the original verilog files
- `verilog2graph.py` converts the processed verilog files into graphs, which are saved in json files

### Ground truth (i.e. label) generation
[dataset-generation](https://github.com/lydiawunan/LOSTIN/tree/main/dataset-generation) generates the ground truth based on the [ABC logic synthesis tool](https://github.com/berkeley-abc/abc).

Specifically,
- `flow_generation.ipynb` generates random synthesis flows with different lengths.
   - flow_10.csv, flow_15.csv, flow_20.csv, flow_25.csv are the previously generated synthesis flows.
- `run_abc_syn.py` invokes ABC to produce area and delay, as the ground truth.
   - The previously generated ground truth are saved in [dataset-ground-truth](https://github.com/lydiawunan/LOSTIN/tree/main/dataset-ground-truth).

## Baselines
- CNN-based model
   - Implemented based on [Developing synthesis flows without human knowledge](https://arxiv.org/abs/1804.05714)
- LSTM-based model
   - Requirements: torchtext 0.6.0
   - Implemented based on [Decision Making in Synthesis cross Technologies using LSTMs
and Transfer Learning](https://ycunxi.github.io/cunxiyu/papers/MLCAD2020.pdf)
   - Data preprocessing: [`data_preprocessing_for_lstm.ipynb`](https://github.com/lydiawunan/LOSTIN/blob/main/data_preprocessing_for_lstm.ipynb)

## Hybrid GNN models
1. GNN with a supernode
2. GNN + LSTM
   - How to run:
   '''
   python main_gnn_customized_area.py
   '''
