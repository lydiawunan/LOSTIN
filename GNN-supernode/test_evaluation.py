import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import argparse
import time
import numpy as np
import json
import operator
from functools import reduce

### importing OGB
from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
#reg_criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)



def gen_batch_dat(batch, graphs):
    edge_index, edge_attr, x, bat = None, None, None, None

    for idx in range(len(batch.y)):
        if idx == 0:
            edge_index = graphs[int(batch.graph_selection[idx])].edge_index
            edge_attr = graphs[int(batch.graph_selection[idx])].edge_attr
            x = graphs[int(batch.graph_selection[idx])].x
            bat = torch.zeros(len(graphs[int(batch.graph_selection[idx])].x))
        else:
            edge_index = torch.cat((edge_index, graphs[int(batch.graph_selection[idx])].edge_index), 1)
            edge_attr = torch.cat((edge_attr, graphs[int(batch.graph_selection[idx])].edge_attr), 0)
            x = torch.cat((x, graphs[int(batch.graph_selection[idx])].x), 0)
            bat = torch.cat((bat, idx+torch.zeros(len(graphs[int(batch.graph_selection[idx])].x))), 0)

    batch.edge_index = edge_index
    batch.edge_attr = edge_attr
    batch.x = x
    batch.batch = bat.to(torch.long)

    return batch


def train(model, device, loader, optimizer, task_type, graphs):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = gen_batch_dat(batch, graphs).to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator, graphs):
    model.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = gen_batch_dat(batch, graphs).to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), y_true, y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=10,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=8,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--virtual_emb_dim', type=int, default=25,
                        help='dimensionality of hidden units of virtual node in GNNs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="pita_delay",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--ckpt', type=str, default="TBD",
                        help='checkpoint file path')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    verilog_list = ['adder', 'arbiter', 'bar', 'div', 'log2', 'max', 'multiplier', 'sin', 'sqrt', 'square', 'voter']
    graphs = dataset.graphs

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    ckpt = args.ckpt
    ckpt_path = f'model/{ckpt}.pt' # args['ckpt']
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

    valid_curve = []
    test_curve = []
    train_curve = []

    test_predict_value= []
    test_true_value= []
    valid_predict_value= []
    valid_true_value= []

    for idx, verilog in enumerate(verilog_list):
        data = DataLoader(dataset[300000*idx:(300000*idx+3000)], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_perf, t_true, t_pred = eval(model, device, data, evaluator, graphs)
        print("Done with evaluation design:", verilog)
        print(test_perf)
        np.save(f'delays/{verilog}_true.npy', t_true)
        np.save(f'delays/{verilog}_pred.npy', t_pred)
    


if __name__ == "__main__":
    main()
