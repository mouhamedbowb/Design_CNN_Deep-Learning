# import libraries
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

# for getting summary info on models
from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


#Import and process the data

data = np.loadtxt(
    'mnist_train.csv',
    delimiter=','
)

labels = data[:,0]
data   = data[:,1:]



# NEW: reshape to 2D!
dataNorm = data.reshape(data.shape[0],1,28,28)


dataNorm.shape


#conver to  tensor

dataT = torch.tensor(dataNorm).float()

labelsT = torch.tensor(labels).long()


train_data , train_label , test_data , test_label = train_test_split(dataT , labels , test_size=.1)


#convert into  pytorch dataset

train_data = TensorDataset(train_data,train_label)
test_data = TensorDataset(test_data , test_label)


#translate into dataloader objects
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(test_data , batch_size=test_data.tensors[0].shape[0])


train_loader.dataset.tensors[0].shape


#create the Model



def createMnistnet(printtoggle=False):
  
  
  
  class mnistNet(nn.Module):
    def __init__(self,printtoggle):
      super().__init__()
      
      
      
      #convolution layers
      self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1,padding=1)
      # size: np.floor( (28+2*1-5)/1 )+1 = 26/2 = 13 (/2 b/c maxpool)
      self.con2 = nn.Conv2d(10,20,kernel_size=5,stride=1,padding=1)
      # size: np.floor( (13+2*1-5)/1 )+1 = 11/2 = 5 (/2 b/c maxpool)
      
      
      # compute the number of units in FClayer (number of outputs of conv2)
      #expectSize = np.floor( (5+2*0-1)/1 ) + 1 # fully connected 1 layer has no padding or kernel, so set to 0/1
      #expectSize = 20*int(expectSize**2)  
      
      
      ### fully-connected layer
      self.fc1 = nn.LazyLinear
      (50)
      
      
      #output layer
      self.out = nn.Linear(50,10)
      
      # toggle for printing out tensor sizes during forward prop
      self.print=printtoggle
      
    def forward(self,x):
      print('input {x.shape}') if self.print  else None
      
      #convolution --> maxpool -->relu
      x= F.relu(F.max_pool2d(self.conv1(x),2))
      print(f'Layer conv1/pool1 {x.shape}') if self.print else None
      
      x= F.relu(F.max_pool2d(self.conv2(x),2))
      
      
      #reshape for linear Layer
      
      nUnits = x.shape.numel() / x.shape[0]
      
      x=x.view(-1,int(nUnits)) 
      if self.print : print(f'vectorize :{x.shape}')
      
      #linear layer 
      
      x=F.relu(self.fc1(x))
      if self.print : print(f'fc1 layer : {x.shape}')
      
      x=self.out(x)
      if self.print : print(f'output  layer : {x.shape}')
  
  #create the model instance 

  net = mnistNet(printtoggle)
  
  #loss function 
  
  lossfun = nn.CrossEntropyLoss()
  
  #optimizer
  
  optimizer =torch.optim.adam(net.parameters(), lr =0.001)
  
  return net , lossfun , optimizer


#test the model with one batche

net , lossfun , optimizer = createMnistnet(True)

X,y = next(iter(train_loader))


yHat = net(X)

# check  size of model output and target

print('')

print(yHat.shape)
print(y.shape)
print(y.shape)


#now lets compute the loss

loss =lossfun(yHat,y)
print(' ')
print('Loss:')
print(loss)

#count the total parametres in the model

summary = summary(net,(1,28,28))




#create a function that can train the model


def function2trainTheModel():
  
  #number of epochs
  
  numepoch = 1000
  
  #create a new model
  
  net , lossfun , optimizer = createMnistnet()
  
  # initialises the losses
  
  
  losses = torch.zeros(numepoch)
  trainAcc = []
  testAcc = []
  
  #loop over epochs
  
  for _ in range(numepoch):
    
    #loop over training data batches
    
    net.train()
    
    batchAcc = []
    
    batchloss = []
    for X,y in  train_loader : 
      
      #forward pass and loss
      
      yHat = net(X)
      loss = lossfun(yHat,y)
      
      
      
      #backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      #loss from this batch
      batchloss.append(loss.item())
      
      
      #compute Accuracy
      
      matches = torch.argmax(yHat,axis=1) ==y   #boolean
      
      matchesNumeric = matches.float()   #convert to 0/1
      
      AccuracyAvg = 100 * (torch.mean(matchesNumeric))
      batchAcc.append(AccuracyAvg)
      
      
    # end of batch loop...
    
    #now after training all the batches lets get the average of trainAcc
    
    trainAcc.append(np.mean(batchAcc))
    
    # and get average losses across the batches
    losses[_]=np.mean(batchloss)
    
    
    #mode test
    net.eval()
    
    X,y = next(iter(test_loader))
    with torch.no_grad():
      yHat = net(X)
      
    #the list of testAccuracy
    
    testAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float())) 
    
    
    # end epochs
    
    
    
  return trainAcc, testAcc ,losses, net  





#so now we will run the model and show the result 

trainAcc, testAcc ,losses, net  = createMnistnet()


fig,ax = plt.subplots(1,2,figsize=(16,5))
ax[0].plot(losses,'s-')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model loss')

ax[1].plot(trainAcc,'s-',label='Train')
ax[1].plot(testAcc,'o-',label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
ax[1].legend()

plt.show()


    
    
    
      
      
      
      
      
      
      
      
      
    
    
  





    
      
      
      
      
      
      
      
      
      
      
      
      
        
      
      
      
      
      

    
    
    
    
    
    
    
    






