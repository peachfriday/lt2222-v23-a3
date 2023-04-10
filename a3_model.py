import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from torch import nn
from torch import optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
# Whatever other imports you need

# You can implement classes and helper functions here too.

class Model(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, hidden_size, nonlinear):
        super().__init__()
        self.input_layer = nn.Linear(input_dimensions, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_dimensions)
        if nonlinear == "relu":
            self.nonlinear = nn.ReLU()
        elif nonlinear == "tanh":
            self.nonlinear = nn.Tanh()
        else:
            self.nonlinear = nn.Identity()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,data):
        after_input_layer = self.input_layer(data)
        after_hidden_layer = self.nonlinear(after_input_layer)
        output = self.hidden_layer(after_hidden_layer)
        output = self.logsoftmax(output)

        return output
    

def train(train, pred, epochs, batch_size):
    train = TensorDataset(train, pred)
    train = [sample for sample in train if sample[1].numel() > 0]
    data = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    mod = model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(data):
            model_input = torch.stack([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])

            output = model(model_input)
            loss = loss_function(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return mod

def confusion_matrix_print(model, testvector, testlabel):
    model.eval()
    predictions = model(testvector).argmax(dim=1)
    confusionmatrix = confusion_matrix(testlabel, predictions) 
    print(f"Confusion matrix:\n{confusionmatrix}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hidden_size", type=int, default=0, help="Preferres size of the hidden layer.")
    parser.add_argument("--nonlinear", type=str, choices=["relu", "tanh", "none"], default="none", help="Type of the nonlinearity function to use.")
    args = parser.parse_args()
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    print("Reading {}...".format(args.featurefile))

    df = pd.read_csv(args.featurefile)
    train_df = df.loc[df.iloc[:, -1].str.contains('train')]
    x_train = train_df.iloc[1:, :-2].values
    y_train = train_df.iloc[1:, -2].values
    test_df = df.loc[df.iloc[:, -1].str.contains('test')]
    x_test = test_df.iloc[1:, :-2].values
    y_test = test_df.iloc[1:, -2].values
    
    input_dimensions = x_train.shape[1]
    output_dimensions = len(np.unique(y_train))

    model = Model(input_dimensions, output_dimensions, args.hidden_size, args.nonlinear)

    labelencoder = LabelEncoder()

    y_train_numbers = labelencoder.fit_transform(y_train)
    train_x_tensed = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensed = torch.tensor(y_train_numbers, dtype=torch.long)


    print('The model is being trained...')
    
    trained_model = train(train_x_tensed,train_y_tensed,epochs=4,batch_size=10)

    print('Done training!')

    confusion_matrix_print(trained_model,train_x_tensed,train_y_tensed)
  
