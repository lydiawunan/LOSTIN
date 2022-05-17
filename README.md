# LOSTIN: <ins>L</ins>ogic <ins>O</ins>ptimization via <ins>S</ins>patio-<ins>T</ins>emporal <ins>In</ins>formation with Hybrid Graph Models

## Benchmark

All considered designs come from [EPFL logic synthesis benchmark suite](https://github.com/lsils/benchmarks)
- 9 designs from the arithmetic benchmarks
- 2 designs from the random/control benchmarks

## Baselines

- CNN-based model
   - [Developing synthesis flows without human knowledge](https://arxiv.org/abs/1804.05714)
- LSTM-based model
   - Requirements: torchtext 0.6.0
   - [Decision Making in Synthesis cross Technologies using LSTMs
and Transfer Learning](https://ycunxi.github.io/cunxiyu/papers/MLCAD2020.pdf)

## Hybrid GNN models
1. GNN with a supernode
2. GNN + LSTM
