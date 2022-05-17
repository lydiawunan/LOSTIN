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

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

destination_folder = 'data_delay'

# LSTM model

class LSTM(nn.Module):

    def __init__(self, input_dim, emb_dim, hidden_dim=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm_1 = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=False)

        self.linear=ModuleList()
        self.linear.append(Linear(hidden_dim,30))
        self.linear.append(Linear(30,30))
        self.linear.append(Linear(30,1))

        self.norm=ModuleList()
        self.norm.append(BatchNorm1d(30))
        self.norm.append(BatchNorm1d(30))


    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, (h1, c1) = self.lstm_1(packed_input)
        packed_output, _ = self.lstm_2(packed_output, (h1, c1))
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out = output[:, -1, :]
        
        #print(output.shape)
        #print(out)

        #flow_fea = F.dropout(out,p=0.5,training=self.training)

        flow_fea=F.relu(self.linear[0](out))
        flow_fea=self.norm[0](flow_fea)
        #flow_fea=F.dropout(flow_fea,p=0.4,training=self.training)

        flow_fea=F.relu(self.linear[1](flow_fea))
        flow_fea=self.norm[1](flow_fea)
        flow_fea=F.dropout(flow_fea,p=0.2,training=self.training)

        flow_out=self.linear[2](flow_fea)
        flow_out = torch.squeeze(flow_out, 1)

        return flow_out

# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function

def training(model, device,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs,
          eval_every,
          criterion = nn.MSELoss(),
          file_path = destination_folder,
          best_valid_loss = float("Inf"),best_train_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for ((flow, flow_len), labels), _ in tqdm(train_loader, desc="Iteration"):         
            labels = labels.to(device)
            flow = flow.to(device)
            flow_len = flow_len.to("cpu")
            output = model(flow, flow_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for ((flow, flow_len), labels), _ in valid_loader:
                      labels = labels.to(device)
                      flow = flow.to(device)
                      flow_len = flow_len.to("cpu")
                      output = model(flow, flow_len)

                      loss = criterion(output, labels)
                      valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss + best_train_loss > average_valid_loss + average_train_loss:
                    best_valid_loss = average_valid_loss
                    best_train_loss = average_train_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



def main():

    # arguments
    parser = argparse.ArgumentParser(description='LSTM baseline for flow perf prediction')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 300)')
    parser.add_argument('--emb_dim', type=int, default=20, help='dimensionality of hidden units in GNNs (default: 300)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Fields
    delay_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    flow_field = Field(lower=True, include_lengths=True, batch_first=True)
    fields = [ ('flow', flow_field), ('delay', delay_field)]

    # TabularDataset
    train, valid, test = TabularDataset.splits(path=destination_folder, train='train.csv', validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=True, sort_within_batch=True)
    #test_iter = BucketIterator(test, batch_size=args.batch_size, sort_key=lambda x: len(x.flow), device=device, sort=True, sort_within_batch=True)

    # Vocabulary
    flow_field.build_vocab(train, min_freq=1, specials_first = False)

    learning_rate=2e-3
    weight_decay=2e-6

    model = LSTM(input_dim=len(flow_field.vocab),emb_dim=args.emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    training(model = model, device = device, optimizer = optimizer, \
        train_loader = train_iter, valid_loader = valid_iter, eval_every = len(train_iter), \
            num_epochs=args.epochs)

    

if __name__ == "__main__":
    main()