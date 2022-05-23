import torch
from torch_geometric.loader import DataLoader
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
    parser.add_argument('--dataset', type=str, default="pita_area",
                        help='dataset name (default: ogbg-molhiv)')

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

    #  [‘adder’, ‘arbiter’, ‘bar’, ‘div’, ‘log2’, ‘max’] / [‘multiplier’, ‘sin’, ‘sqrt’, ‘square’, ‘voter’]
    data_0, dataset_test_0 = dataset[0:1800000], dataset[1800000:3300000]
    dataset_ratio = [660000, 165000, 975000]
    dataset_train, dataset_valid, dataset_test_1 = torch.utils.data.random_split(data_0, dataset_ratio)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # test_loader = DataLoader(dataset_test_0 + dataset_test_1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

    valid_curve = []
    test_curve = []
    train_curve = []

    test_predict_value= []
    test_true_value= []
    valid_predict_value= []
    valid_true_value= []


    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type, graphs)

        print('Evaluating...')
        # train_perf, _, _ = eval(model, device, train_loader, evaluator, graphs)
        valid_perf, v_true,  v_pred= eval(model, device, valid_loader, evaluator, graphs)
        # test_perf, t_true, t_pred = eval(model, device, test_loader, evaluator, graphs)

        print({'Validation': valid_perf})

        # train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

        # test_predict_value.append(reduce(operator.add, t_pred.tolist()))
        valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

        # test_loss=test_perf[dataset.eval_metric]
        valid_loss=valid_perf[dataset.eval_metric]
        if valid_loss<=np.min(np.array(valid_curve)):
            PATH='model/1_'+args.dataset + '_'+ args.gnn+ '_layer_'+ str(args.num_layer)+'_model.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss
                        }, PATH)
    
    # test_true_value=reduce(operator.add, t_true.tolist())
    valid_true_value=reduce(operator.add, v_true.tolist())

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))

    f = open('1_'+args.dataset + '_'+ args.gnn+ '_layer_'+ str(args.num_layer)+ '.json', 'w')
    result=dict(val=valid_curve[best_val_epoch],
                valid_pred=valid_predict_value, 
                valid_true=valid_true_value,
                valid_curve=valid_curve)
    json.dump(result, f)
    f.close()

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch]}, args.filename)


if __name__ == "__main__":
    main()
