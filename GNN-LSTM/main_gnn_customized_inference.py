### Libraries
# torchtext 0.6.0
import numpy as np
import argparse
from tqdm import tqdm
# Libraries
import matplotlib.pyplot as plts
import pandas as pd
import torch

# Preliminaries
from torchtext.data import Field, TabularDataset, BucketIterator

# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn import ReLU, Linear, BatchNorm1d, ModuleList

# Training
import torch.optim as optim

# graph loading dependency
from torch_geometric.data import DataLoader
from dataset_pyg import PygGraphPropPredDataset
from gnn import GNN



# Hybridmodel model
class Hybridmodel(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim=64, graph_emb=11, model_name='gin', num_layer=5):
        super(Hybridmodel, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=False)

        self.gmodel = GNN(gnn_type = model_name, num_tasks = 1, num_layer = num_layer, emb_dim = graph_emb, drop_ratio = 0.2, virtual_node = False)

        self.linear=ModuleList()
        self.linear.append(Linear(hidden_dim + graph_emb, 100))
        self.linear.append(Linear(100,100))
        self.linear.append(Linear(100,1))

        self.norm=ModuleList()
        self.norm.append(BatchNorm1d(100))
        self.norm.append(BatchNorm1d(100))


    def forward(self, text, text_len, graph_batch):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out = output[:, -1, :]

        g_emb = self.gmodel(graph_batch)
        combined_emb = torch.cat((out, g_emb[text[:,0]-7]),1)

        flow_fea=F.relu(self.linear[0](combined_emb))
        flow_fea=self.norm[0](flow_fea)
        flow_fea=F.dropout(flow_fea,p=0.2,training=self.training)

        flow_fea=F.relu(self.linear[1](flow_fea))
        flow_fea=self.norm[1](flow_fea)
        flow_fea=F.dropout(flow_fea,p=0.2,training=self.training)

        flow_out=self.linear[2](flow_fea)
        flow_out = torch.squeeze(flow_out, 1)

        return flow_out


def load_checkpoint(load_path, model, optimizer, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']



def main():

    # arguments
    parser = argparse.ArgumentParser(description='LSTM baseline for flow perf prediction')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--emb_dim', type=int, default=20, help='dimensionality of hidden units in GNNs')
    parser.add_argument('--graph_emb', type=int, default=32, help='dimensionality of hidden units in GNNs')
    parser.add_argument('--dest_folder', type=str, default='model_ckt/area', help='Destination folder that saves the model')
    parser.add_argument('--data_folder', type=str, default='lstm/data_area', help='The folder that saves the data')
    parser.add_argument('--model_name', type=str, default='gin', help='GNN model name')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN model name')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Load graphs
    pyg_dataset = PygGraphPropPredDataset(name = 'vgraph')
    graph_loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)

    # Fields
    area_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    flow_field = Field(lower=True, include_lengths=True, batch_first=True)
    fields = [ ('flow', flow_field), ('area', area_field)]

    print("Loading data ...")

    # TabularDataset
    train, valid, test = TabularDataset.splits(path=args.data_folder, train='train.csv', validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=True, sort_within_batch=True)
    #valid_iter = BucketIterator(valid, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=True, sort_within_batch=True)
    test_iter = BucketIterator(test, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=False, sort_within_batch=False)

    # Vocabulary
    flow_field.build_vocab(train, min_freq=1, specials_first = False)

    learning_rate=2e-3
    weight_decay=2e-6

    
    model = Hybridmodel(input_dim=len(flow_field.vocab), emb_dim=args.emb_dim, graph_emb=args.graph_emb, model_name=args.model_name, num_layer=args.num_layer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    load_checkpoint(args.dest_folder + '/model_sum_'+args.model_name+str(args.num_layer)+'_batch_32.pt', model, optimizer, device)

    y_pred = []
    y_true = []
    relative_error = []
    flow_l = []
    design = []

    for graph_batch in graph_loader:
        graph_batch = graph_batch.to(device)

    model.eval()

    with torch.no_grad():
        for ((flow, flow_len), labels), _ in tqdm(test_iter, desc="Iteration"):         
            labels = labels.to(device)
            flow = flow.to(device)
            flow_len = flow_len.to("cpu")
            output = model(flow, flow_len, graph_batch)

            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

            rmae = np.abs(np.divide(np.subtract(output.tolist(), labels.tolist()), labels.tolist()))
            relative_error.extend(rmae)

            flow_l.extend(flow_len.tolist())
            design.extend((flow[:,0]-7).tolist())

    output = pd.DataFrame({'design_name':design, 'flow_length':flow_l, 'labels': y_true, 'prediction': y_pred, 'relative error': relative_error})
    output.to_csv('inference_'+args.dest_folder.split('/')[1]+'_'+args.model_name+str(args.num_layer)+'.csv',index=False)
    print(np.mean(relative_error))

    
if __name__ == "__main__":
    main()