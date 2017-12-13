import os
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans

def GetFileList(data_dir):
    train_dir = os.path.join(data_dir,'train')
    test_dir = os.path.join(data_dir,'test')
    
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    train_files.sort()
    test_files.sort()
    
    N = len(train_files)
    N_train = int(N)
    
    return train_files[:N_train],test_files

def GetMinData(data_dir,file_name):
    file = os.path.join(data_dir,file_name)
    
    with open(file) as f:
        lines = f.readlines()
    
    price_list = []

    for i in range(len(lines)):
        if i%6==0:
            price = float(lines[i].split(sep='/')[0])
            price_list.append(price)
    
    return torch.FloatTensor(price_list)

def GetPastData(data_dir,files,idx):
    target_idx = [idx-10,idx-5,idx-3,idx-1]
    avg_list = []
    for t in target_idx:
        data = GetMinData(data_dir,files[t])
        avg_list.append(data.mean())
    
    return avg_list

def GetPriceData1(data_dir,filename):
    # Data with close price only
    file = os.path.join(data_dir,filename)
    csv = pd.read_csv(file)
    
    if 'YAHOO' in filename:
        close = torch.FloatTensor(csv['Adj Close'])
    
    elif 'INVESTING' in filename:
        close = torch.FloatTensor(csv.iloc[:,1])

    return close

def TrainTestSplit(data):
    N = data.shape[0]
    n_train = int(N*0.5)

    train = data[:n_train]
    test = data[n_train:]

    return train, test

def GetInput(data,t,Type):
    raw_price = data[t-45:t]
    momentum = torch.FloatTensor(GetMomentum(data,t,Type))

    input = torch.cat([raw_price,momentum])

    return input

def Normalize(self,data):
    Max = data.max()
    Min = data.min()

    Normed = ((2*data)-(Max+Min)) / (Max-Min)

    return Normed

def GetMomentum(data,t,Type):
    momentum = []
    
    if Type == 'daily':
        interval = [1,3,5,10,20]
    elif Type == 'intra':
        interval = [0.3,0.5,1,3,10]
    
    for gap in interval:
        momentum.append(data[t]-data[t-gap])

    return momentum

def GetBatchInput(data):
    # Create every possible input vectors and concatenate to tensor

    N = data.shape[0]
    input_vectors = []
    input_mean = []

    for t in range(45,N):
        input = data[t-45:t].unsqueeze(0)
        input_vectors.append(input)
        input_mean.append(torch.mean(input))

    input_tensor = torch.cat(input_vectors,0)
    
    return input_tensor, input_mean

def Clustering(input_mean):
    # Applying clustering algorithm to input mean
    # return labels for each input tenros

    X = np.array(input_mean).reshape(-1,1)

    kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
    label = kmeans.labels_
    
    return label

def GetFuzzyParams(Input_Vectors,cluster_label):
    # Calculate mean and var w.r.t each dimension
    
    FuzzyMean = {}
    FuzzyStd = {}

    for i in range(3):
        label = torch.LongTensor(np.where(cluster_label==i)[0])
        
        Cluster = Input_Vectors[label]
        
        mean = torch.mean(Cluster,0)
        std = torch.std(Cluster,0)

        FuzzyMean[i] = mean
        FuzzyStd[i] = std

    return (FuzzyMean,FuzzyStd)























