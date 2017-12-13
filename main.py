import pandas as pd
import torch
from dataset import *
from model import *
from run import *

class Config():
    def __init__(self):
        self.data_dir = './Data/Korea Data/index/_001'
        #self.data_dir = './Data'
        self.filename = 'S&P500_YAHOO.csv'
        self.type = 'daily'
     
def main():
    config = Config()
    train_dir = os.path.join(config.data_dir,'train')
    test_dir = os.path.join(config.data_dir,'test')
    #index_price_data = GetPriceData1(config.data_dir,config.filename)
    #index_train,index_test = TrainTestSplit(index_price_data)
    
    #samsung_data = GetPriceData1(config_drl.data_dir,'SAMSUNG_YAHOO.csv')
    #samsung_train,samsung_test = TrainTestSplit(samsung_data)
    
   
    DRL_model = DRL(config)
    DRL_model.cuda()
    #run_train_d(index_train,DRL_model)
    #run_test_d(index_test,DRL_model)
      
    train_files, test_files = GetFileList(config.data_dir) 
    run_train_m(train_dir,train_files,DRL_model)
    #run_test_m(train_dir,train_files,DRL_model)
    
    #run_test_m(test_dir,test_files,DRL_model)

if __name__ == '__main__':
    main()
