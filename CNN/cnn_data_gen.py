import utils
import pandas as pd
import numpy as np
import argparse
import pprint as pp

from os import listdir
from os.path import isfile, join

import torch
from torch.utils.data import TensorDataset, DataLoader



def main(args):
    ff_10 = pd.read_csv('flow_10.csv',header=None)
    ff_15 = pd.read_csv('flow_15.csv',header=None)
    ff_20 = pd.read_csv('flow_20.csv',header=None)
    ff_25 = pd.read_csv('flow_25.csv',header=None)

    keyword = args['key']
    label_dir = 'dataset-ground-truth'
    label_list = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
    verilog_list = ['div', 'max', 'multiplier', 'sin', 'square', 'voter', 'adder', 'arbiter', 'bar', 'log2', 'sqrt']
    stat_list = []
    dataset_x = []
    dataset_y = []

    # Collect all of the data from the abc first
    for verilog in verilog_list:
        v_file = f'epfl/{verilog}.v'
        stat = utils.run_abc(v_file, '')
        delay, area, edge, nd, lev, i, o = utils.get_cnn_metrics(stat)
        stat_list.append((delay, area, edge, nd, lev, i, o))

    print("Acquired all of the data from abc!")


    # Main loop
    for i, verilog in enumerate(verilog_list):

        print("Begin processing the data for the verilog file: ", verilog)

        delay, area, edge, nd, lev, i, o = stat_list[i]
        label_file_10, label_file_15, label_file_20, label_file_25 = '', '', '', ''

        for f in label_list:
            if (keyword in f) and (verilog in f):
                if '10' in f:
                    label_file_10 = f
                elif '15' in f:
                    label_file_15 = f
                elif '20' in f:
                    label_file_20 = f
                elif '25' in f:
                    label_file_25 = f

        print("Label 10 file: ", label_file_10)
        print("Label 15 file: ", label_file_15)
        print("Label 20 file: ", label_file_20)
        print("Label 25 file: ", label_file_25)

        label_10 = pd.read_csv(f'{label_dir}/{label_file_10}', header=None)
        label_15 = pd.read_csv(f'{label_dir}/{label_file_15}', header=None)
        label_20 = pd.read_csv(f'{label_dir}/{label_file_20}', header=None)
        label_25 = pd.read_csv(f'{label_dir}/{label_file_25}', header=None)

        # Processing Length 10 Flow
        for i in range(50000):
            commands = ff_10[0][i].split(';')
        
            data = np.zeros([1, 26, 7])
            data[0][0] = [i, o, nd, lev, edge, area, delay]

            for j in range(10):
                if commands[j] == 'b':
                    data[0][j+1][0] = 1.0
                elif commands[j] == 'rf':
                    data[0][j+1][1] = 1.0
                elif commands[j] == 'rfz':
                    data[0][j+1][2] = 1.0
                elif commands[j] == 'rw':
                    data[0][j+1][3] = 1.0
                elif commands[j] == 'rwz':
                    data[0][j+1][4] = 1.0
                elif commands[j] == 'resub':
                    data[0][j+1][5] = 1.0
                elif commands[j] == 'resub -z':
                    data[0][j+1][6] = 1.0
                else:
                    raise NotImplementedError

            dataset_x.append(data)
            dataset_y.append([label_10[0][i]])

        print("Completed processing for flow-length 10")

        # Processing Length 15 Flow
        for i in range(50000):
            commands = ff_15[0][i].split(';')
        
            data = np.zeros([1, 26, 7])
            data[0][0] = [i, o, nd, lev, edge, area, delay]

            for j in range(15):
                if commands[j] == 'b':
                    data[0][j+1][0] = 1.0
                elif commands[j] == 'rf':
                    data[0][j+1][1] = 1.0
                elif commands[j] == 'rfz':
                    data[0][j+1][2] = 1.0
                elif commands[j] == 'rw':
                    data[0][j+1][3] = 1.0
                elif commands[j] == 'rwz':
                    data[0][j+1][4] = 1.0
                elif commands[j] == 'resub':
                    data[0][j+1][5] = 1.0
                elif commands[j] == 'resub -z':
                    data[0][j+1][6] = 1.0
                else:
                    raise NotImplementedError

            dataset_x.append(data)
            dataset_y.append([label_15[0][i]])

        print("Completed processing for flow-length 15")

        # Processing Length 20 Flow
        for i in range(100000):
            commands = ff_20[0][i].split(';')
        
            data = np.zeros([1, 26, 7])
            data[0][0] = [i, o, nd, lev, edge, area, delay]

            for j in range(20):
                if commands[j] == 'b':
                    data[0][j+1][0] = 1.0
                elif commands[j] == 'rf':
                    data[0][j+1][1] = 1.0
                elif commands[j] == 'rfz':
                    data[0][j+1][2] = 1.0
                elif commands[j] == 'rw':
                    data[0][j+1][3] = 1.0
                elif commands[j] == 'rwz':
                    data[0][j+1][4] = 1.0
                elif commands[j] == 'resub':
                    data[0][j+1][5] = 1.0
                elif commands[j] == 'resub -z':
                    data[0][j+1][6] = 1.0
                else:
                    raise NotImplementedError

            dataset_x.append(data)
            dataset_y.append([label_20[0][i]])

        print("Completed processing for flow-length 20")

        # Processing Length 25 Flow
        for i in range(100000):
            commands = ff_25[0][i].split(';')
        
            data = np.zeros([1, 26, 7])
            data[0][0] = [i, o, nd, lev, edge, area, delay]

            for j in range(25):
                if commands[j] == 'b':
                    data[0][j+1][0] = 1.0
                elif commands[j] == 'rf':
                    data[0][j+1][1] = 1.0
                elif commands[j] == 'rfz':
                    data[0][j+1][2] = 1.0
                elif commands[j] == 'rw':
                    data[0][j+1][3] = 1.0
                elif commands[j] == 'rwz':
                    data[0][j+1][4] = 1.0
                elif commands[j] == 'resub':
                    data[0][j+1][5] = 1.0
                elif commands[j] == 'resub -z':
                    data[0][j+1][6] = 1.0
                else:
                    raise NotImplementedError

            dataset_x.append(data)
            dataset_y.append([label_25[0][i]])

        print("Completed processing for flow-length 25")


    tensor_x = torch.Tensor(dataset_x)
    tensor_y = torch.Tensor(dataset_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    dir_upper, dir_lower = args['dataset'], args['key']
    torch.save(my_dataset, f'{dir_upper}/{dir_lower}.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument('--key', help='Select area / delay', default='area')
    parser.add_argument('--dataset', help='the save directory of dataset', default='cnn_dataset')

    args = vars(parser.parse_args())
    main(args)
