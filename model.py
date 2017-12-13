import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DRL(nn.Module):
    # Direct Reinforcement model to learn trading policy
    # Using shallow rnn to get policy
    def __init__(self,config):
        super(DRL,self).__init__()
        self.lr = 0.00001
        self.epoch = 5 
        self.more_train = False 
        self.load_model = 'DDRmin_H20_TP_FTabs_Trunc5_4'
        self.save_model = 'DDRmin_H30_TP_FTabs_Trunc5'
        
        self.objective_type = 'TP'
        self.model_type = 'MLP'
        self.MLP_layer = 2
        self.noise_level = 2 

        self.truncate_step = 5 
        self.lamda = 0.01
        self.cost_rate = 0.0001 
        
        self.input_size = 45
        self.hidden_size = 30 

        if self.model_type == 'RNN':
            #self.RNN_cell = nn.RNNCell(self.input_size+5,self.hidden_size)
            self.GRU_cell = nn.GRUCell(self.input_size+5,self.hidden_size)
            self.RNN_Linear = nn.Linear(self.hidden_size,1)
        
        else:
            if self.MLP_layer == 1:
                self.MLPOnly = nn.Linear(self.input_size+5,1)
            else:
                self.MLP_Linear1 = nn.Linear(self.input_size+5,self.hidden_size)
                self.MLP_Linear2 = nn.Linear(self.hidden_size,1)

        #self.Policy_u = nn.Linear(1,1,bias=False)
        self.Policy_u = nn.Parameter(torch.rand(1))
        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)
        
        for param in self.parameters():
            print(param.data.shape)
    
    def GetPolicy_RNN(self,Rt,Ht_1,Ft_1):
        # Does not explicitly using Ft_1
        # Using Hidden state Ht_1 instead
        Ft_1 = Variable(Ft_1).cuda()
        Policy_bias = self.trading_reg * Ft_1
        input_t = Rt.unsqueeze(0)
        
        Ht = self.GRU_cell(input_t,Ht_1)
        H = F.tanh(self.RNN_Linear(Ht).squeeze()+Policy_bias)
        #H =  F.tanh(self.RNN_Linear(Ht).squeeze())
        
        value = H.data[0]
        
        # Short position
        if value <= -0.25:
            strategy = -1
        # Long position
        elif value >= 0.25:
            strategy = 1
        # Neutral 
        else:
            strategy = 0
        
        diff = value - strategy
        Ft = H - diff
       
        return Ft, Ht 
    
    def GetPolicy_MLP(self,Ft_1,Rt,noise):
        # Rt [Rt_m,...,Rt_1]
        
        if self.MLP_layer ==1:
            H = self.MLPOnly(Rt)
        else:
            H1 = F.sigmoid(self.MLP_Linear1(Rt))
            H = self.MLP_Linear2(H1)
        
        #Policy_bias = self.trading_reg * Ft_1
        Policy_bias = self.Policy_u*Ft_1
        out = F.tanh(H+Policy_bias)

        value = out.data[0]
        
        # Short position
        if value <= -0.25:
            strategy = -1
        # Long position
        elif value >= 0.25:
           strategy = 1
        # Neutral 
        else:
            strategy = 0 
        
        diff = value - strategy
        Ft = out - diff
        
        return Ft

    def CheckStateChange(self,policies):
        Ft_1, Ft = policies
        
        if Ft_1.data[0] == Ft.data[0]:
            return 0 
        else:
            return 1
    
    def GetCostAmount(self,policies):
        Ft_1, Ft = policies

        amount = torch.abs(Ft-Ft_1)
        #print(Ft.data[0],Ft_1.data[0],amount.data[0]) 
        return amount

    def GetPriceChange(self,price_data):
        
        Pt = price_data[1:]
        Pt_1 = price_data[:-1]
        revenue = Pt - Pt_1
        
        return revenue
    
    def GetMomentum(self,price_data,momentum_days):
        Pt = price_data[-1]
        Momentum = []
        for avg in momentum_days:
            Momentum.append(Pt-avg)
        Momentum.append(Pt-price_data[0]) 
        return torch.cat(Momentum)


    def GetSharpeRatio(self,Rt,At_1,Bt_1):
        #print(Rt.data[0],At_1.data[0],Bt_1.data[0])

        At = At_1 + self.lamda * (Rt - At_1)
        Bt = Bt_1 + self.lamda * (Rt**2 - Bt_1)
        
        first_term = Bt_1 * (Rt-At_1) 
        second_term = (At_1 * (Rt**2 - Bt_1)) / 2
        numerator = (Bt_1 - At_1**2) ** (3/2)
        
        if numerator.data[0] != 0:
            Dt = (first_term - second_term) / numerator
        else:
            Dt = Variable(torch.zeros(1)).cuda()

        #devDt = (Bt_1 - At_1*Rt) / (Bt_1 - At_1**2)**(3/2)

        return Dt, Variable(At.data).cuda(), Variable(Bt.data).cuda()



















