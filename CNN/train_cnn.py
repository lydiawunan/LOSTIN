import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import argparse


# Keras -> PyTorch Implementation
# Modification: Classification -> Regression
class CNN_Regression(nn.Module):
    def __init__(self):
        super(CNN_Regression, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, (1, 2))
        self.conv_2 = nn.Conv2d(32, 64, (1, 2))
        self.pool = nn.MaxPool2d(1, 1)
        self.dropout_1 = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(64*26*5, 64)
        self.dropout_2 = nn.Dropout(0.4)
        self.fc_2 = nn.Linear(64, 1)

    def forward(self, x):
        elu = nn.ELU()
        selu = nn.SELU()

        x1 = self.conv_1(x)
        x2 = self.conv_2(elu(x1))
        x3 = self.pool(elu(x2))
        x4 = self.dropout_1(x3)
        x5 = self.fc_1(torch.flatten(x4, start_dim=1))
        x6 = self.dropout_2(selu(x5))
        out = self.fc_2(x6)
        
        return out


def RMAE(output, target):
    diff = ((target - output) / target).abs().sum()
    rmae = diff / len(output)

    return rmae


def main(args):
    model = CNN_Regression()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    epoches = 20
    batch_size = 32

    data_key = args['data']
    data_set = torch.load(f'cnn_dataset/{data_key}.pt')
    device = torch.device("cuda:" + str(args['device'])) if torch.cuda.is_available() else torch.device("cpu")

    data_0, dataset_test_0 = TensorDataset(*data_set[0:1800000]), TensorDataset(*data_set[1800000:3300000])
    dataset_ratio = [660000, 165000, 975000]
    dataset_train, dataset_valid, dataset_test_1 = torch.utils.data.random_split(data_0, dataset_ratio)

    data_set_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_set_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    data_set_test = DataLoader(dataset_test_0 + dataset_test_1, batch_size=batch_size, shuffle=True)

    checkpoint_path = args['ckpt']
    test_accuracy_curve = []

    if torch.cuda.is_available():
        print("CUDA is available! Running on GPU")
        model = model.to(device)
    else:
        print("Running on CPU")

    # Training routine
    for epoch_idx in range(epoches):
        print("Training Epoch #: ", epoch_idx+1)
        print()
        for idx, (data_batch, label_batch) in enumerate(data_set_train):
            if torch.cuda.is_available():
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)

            out = model.forward(data_batch)
            loss = criterion(out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 200 == 0:
                batch_acc = RMAE(out, label_batch)
                print("RMAE: ", batch_acc)

        print("Training done! Now evaluating..")
        print()

        # Training accuracy
        count, train_sum = 0, 0
        for data_batch, label_batch in data_set_train:
            if torch.cuda.is_available():
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)
            
            with torch.no_grad():
                out = model.forward(data_batch)
                batch_acc = RMAE(out, label_batch)

            count += 1
            train_sum += batch_acc

        train_sum = train_sum / count
        print("Training Result: ", train_sum, "%")

        # Validation accuracy
        count, valid_sum = 0, 0
        for data_batch, label_batch in data_set_valid:
            if torch.cuda.is_available():
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)

            with torch.no_grad():
                out = model.forward(data_batch)
                batch_acc = RMAE(out, label_batch)

            count += 1
            valid_sum += batch_acc

        valid_sum = valid_sum / count
        print("Validation Result: ", valid_sum, "%")

        # Testing accuracy
        count, test_sum = 0, 0
        for data_batch, label_batch in data_set_test:
            if torch.cuda.is_available():
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)

            with torch.no_grad():
                out = model.forward(data_batch)
                batch_acc = RMAE(out, label_batch)

            count += 1
            test_sum += batch_acc

        test_sum = test_sum / count
        test_accuracy_curve.append(test_sum)
        print("Testing Result: ", test_sum, "%")

        # Save the result
        if test_sum <= min(test_accuracy_curve):
            torch.save({'model_state_dict': model.state_dict(), 
                        'train_accuracy': train_sum, 
                        'valid_accuracy': valid_sum,
                        'test_accuracy': test_sum,
                       }, f'{checkpoint_path}/{data_key}_result.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this is just parser - nothing special')

    parser.add_argument('--device', default='0')
    parser.add_argument('--data', help='dataset file', default='area')
    parser.add_argument('--ckpt', help='output checkpoint path', default='cnn_checkpoints')

    args = vars(parser.parse_args())
    main(args)

