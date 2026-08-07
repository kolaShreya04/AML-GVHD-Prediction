# importing packages
import tensorflow
import numpy as np # to perform calculations
import pandas as pd # to read data
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import dataloader
from torch.utils.data import Dataset, DataLoader

from Data_Cleaning import *
#TODO: if issues continue use a machine learning model instead

###########
# Pytorch #
###########
#TODO DELETE OLD COMMENTS
#TODO: https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18

data = pd.read_excel("V4_Cleaned_TRANSLATED_2020_Testing_Group.xlsx")
#PYTORCH

looking_for = 'case of survival'
df = label_encode(set_none(data),looking_for)
print("labelenc")
print(df)


#input output features, converted for tensor
x = torch.tensor(df.drop(looking_for,axis = 1).values)
print(x)
y = torch.tensor(df[looking_for].values).reshape(-1, 1)
print(y)
#


n_features = x.shape[1]


#is dataframe need to convert to tensor
print(type(data))   #Dataframe
print(type(x))      #DataFrame
print(type(y))      #series

#convert to tensor for pytorch




#train test split, 80 training, 20 testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train = normalize(X_train,p=1.0,dim=0)

print(torch.Tensor(y_train).shape)
print(torch.Tensor(y_test).shape)

print(X_train)
print(":)")

#num of input features
print(n_features)
input_dim = n_features

batch_size = 100
n_iters = n_features * 3
num_epochs = n_iters / (len(x) / batch_size)
print(num_epochs)
num_epochs = int(num_epochs)
print(num_epochs)

class extraclean(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]

  def __len__(self):
    return self.length

print("clean")
dataset = extraclean(x,y)
#TODO: remove/change dataset is not iterable in this state

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=100)

'''
STEP 3: CREATE MODEL CLASS
'''

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #print(self.fc1)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        #print(self.tanh)
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #print(self.fc2)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        #print("ou1")
        #print(out)
        # Non-linearity
        out = self.sigmoid(out)
        #print("ou2")
       # print(out)
        # Linear function (readout)
        out = self.fc2(out)
        #print("ou3")
        #print(out)
        return out
''' k
STEP 4: INSTANTIATE MODEL CLASS
'''
#TODO: fix output_dim only works when set to 1
epochs = 100
input_dim = n_features
hidden_dim = 5
output_dim = 1

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.MSELoss()
#cross entropy loss will not work

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
iter = 0
costval = []
for j in range(epochs):

    for i,(x_train,y_train) in enumerate(dataloader):
        y_pred = model(x_train)
        # calculating loss
        #look at shape of tensors + dimensions
        cost = criterion(y_pred, y_train)


        # backprop
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        costval.append(cost)

        iter += 1
        # trainset, not train_loader
        # probably x in your case
        # Clear figure



        print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, cost.item(),  ((y_pred > 0.5) == y).float().mean().item()))


        #TODO: need to output correlation between columns and other items
        #for feature importance

        # plt.clf()
        #
        # y_train.numpy()
        # # Get predictions
        # predicted = model(torch.from_numpy(x_train.numpy()).requires_grad_()).data.numpy()
        #
        # # Plot true data
        # plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
        #
        # # Plot predictions
        # plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
        #
        # # Legend and plot
        # plt.legend(loc='best')
        # plt.show()
