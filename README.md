# LOSTIN: <ins>L</ins>ogic <ins>O</ins>ptimization via <ins>S</ins>patio-<ins>T</ins>emporal <ins>In</ins>formation with Hybrid Graph Models

## Benchmark

All considered designs come from [EPFL logic synthesis benchmark suite](https://github.com/lsils/benchmarks)
- 9 designs from the arithmetic benchmarks
- 2 designs from the random/control benchmarks

## Data Preprocessing
### From verilog to graph
[verilog2graph](https://github.com/lydiawunan/LOSTIN/tree/main/verilog2graph) parses verilog designs into graphs. 

Specifically,
- `verilog_cleanser.py` removes the symbols incompatible to our parser from the original verilog files.
- `verilog2graph.py` converts the processed verilog files into graphs, which are saved in json files.

### Ground truth (i.e. label) generation
[dataset-generation](https://github.com/lydiawunan/LOSTIN/tree/main/dataset-generation) generates the ground truth based on the [ABC logic synthesis tool](https://github.com/berkeley-abc/abc).

Specifically,
- `flow_generation.ipynb` generates random synthesis flows with different lengths.
   - flow_10.csv, flow_15.csv, flow_20.csv, flow_25.csv are the previously generated synthesis flows.
- `run_abc_syn.py` invokes ABC to produce area and delay, as the ground truth.
   - The previously generated ground truth files are saved in [dataset-ground-truth](https://github.com/lydiawunan/LOSTIN/tree/main/dataset-ground-truth).

## Baselines
- **CNN-based model**
   - Implemented based on [Developing synthesis flows without human knowledge](https://arxiv.org/abs/1804.05714)
   - Data preprocessing: [`cnn_data_gen.py`](https://github.com/lydiawunan/LOSTIN/blob/main/CNN/cnn_data_gen.py)
   - How to run:
      ```python
      python train_cnn.py
      ```
- **LSTM-based model**
   - Requirements: torchtext 0.6.0
   - Implemented based on [Decision Making in Synthesis cross Technologies using LSTMs
and Transfer Learning](https://ycunxi.github.io/cunxiyu/papers/MLCAD2020.pdf)
   - Data preprocessing: [`data_preprocessing_for_lstm.ipynb`](https://github.com/lydiawunan/LOSTIN/blob/main/LSTM/data_preprocessing_for_lstm.ipynb)
   - How to run:
      ```python
      python LSTM_area.py
      ```
   - Note that the dataset files should be unzipped before running the code.

## Hybrid GNN models
- **GNN with a supernode**
   - Requirements: Pytorch Geometric
   - How to run:
      ```python
      python main_gnn.py
      ``` 
- **GNN + LSTM**
   - Requirements: Pytorch Geometric, and torchtext 0.6.0
   - How to run:
      ```python
      python main_gnn_customized_area.py
      ```
   - Note that the dataset files related to [LSTM](https://github.com/lydiawunan/LOSTIN/tree/main/GNN-LSTM/lstm) should be unzipped before running the code.

## Contact and Citation
- If there is any question, please shoot an email to nanwu@ucsb.edu
- If you find LOSTIN useful, please cite our paper:
   ```
   @inproceedings{wu2022lostin,
      title={LOSTIN: Logic Optimization via Spatio-Temporal Information with Hybrid Graph Models},
      author={Wu, Nan and Lee, Jiwon and Xie, Yuan and Hao, Cong},
      booktitle={Proceedings of the 33rd IEEE International Conference on Application-specific Systems, Architectures and Processors},
      year={2022},
      organization={IEEE}
   }
   ```
