import sys
import os
import torch
from torch.autograd import Variable
from dataset import *

def run_train_m(data_dir,train_files,m):
    # Model Setting
    if m.more_train:
        print('Model Loaded -- %s'%m.load_model)
        m.load_state_dict(torch.load('./model/%s'%m.load_model))
    else:
        print('Train model from scratch')

    H0 = Variable(torch.zeros(m.hidden_size)).unsqueeze(0).cuda() 
    F0 = Variable(torch.FloatTensor([0.]),requires_grad=False).cuda()
    
    # Outer For Loop (Training Epoch)
    for epoch in range(m.epoch):
        policy_count = {0:0,1:0,-1:0}
        price_count = {1:0,-1:0}
        frq = 0
        cp = 0
        if m.objective_type == 'SR':
            A0 = Variable(torch.zeros(1)).cuda()
            B0 = Variable(torch.zeros(1)).cuda()
        
        Ut = (torch.FloatTensor([0.]))
        TP = Variable(torch.zeros(1)).cuda() 
        # Inner For loop (for files, each day)
        for file_idx in range(10,len(train_files)):
            momentum_days = GetPastData(data_dir,train_files,file_idx)
            price_data = GetMinData(data_dir,train_files[file_idx])
            N = price_data.shape[0]
            
            # For day trading data, run operations  
            for t in range(m.input_size,N):
                if t == m.input_size:
                    if m.model_type == 'RNN' and file_idx==0:
                        Ht_1 = Variable(H0.data)
                    if m.objective_type=='SR':
                        At_1 = A0
                        Bt_1 = B0
                    Ft_1 = F0
                else:
                    if m.model_type=='RNN':
                        Ht_1 = Ht
                    if m.objective_type=='SR':
                        At_1 = At
                        Bt_1 = Bt
                    Ft_1 = Ft

                input = Variable(price_data[t-m.input_size:t+1]
                                                ,requires_grad=False).cuda()
                revenue = m.GetPriceChange(input).cuda()
                momentum = m.GetMomentum(input,momentum_days).cuda()
                Input = torch.cat([revenue,momentum])
                
                #Noise is added here for exploration
                noise = Variable(torch.normal(0.,torch.FloatTensor([0.2])),
                                    requires_grad=False).cuda()/(epoch+m.noise_level)
                if m.model_type=='MLP':
                    Ft = m.GetPolicy_MLP(Ft_1,Input,noise)
                else:
                    Ft,Ht = m.GetPolicy_RNN(Input,Ht_1,Ft_1.data)
                
                #Ct = m.CheckStateChange((Ft_1,Ft))
                Ct = m.GetCostAmount((Ft_1,Ft))
                Zt = revenue[-1] 
                
                trading_fee = input[-1] * m.cost_rate
                
                if m.objective_type == 'SR':
                    Rt = (Ft_1 * Zt) - (trading_fee * Ct)
                    St,At,Bt = m.GetSharpeRatio(Rt,At_1,Bt_1)
                    Ut += Rt.cpu().data
                else:
                    Rt = -((Ft_1 * Zt) - (trading_fee * Ct))
                    TP += Rt
                    Ut += -Rt.cpu().data
                
                # Parameter tuning
                if t!=m.input_size and t%m.truncate_step==0:
                    m.optimizer.zero_grad()
                    if m.objective_type=='SR': 
                        St.backward()
                    else:
                        TP.backward()
                        TP = Variable(torch.zeros(1)).cuda()
                    if m.model_type=='RNN':
                        Ht = Variable(Ht.data) 
                    Ft = Variable(Ft.data)
                    m.optimizer.step()
                
                # Aggregating and Recording for report
                frq += Ct.data[0]
                cp += (Ct.data[0]!=0)
                policy_count[Ft.data[0]]+=1
                if Zt.data[0]>0:
                    price_count[1] += 1
                else:
                    price_count[-1] += 1
            
            if file_idx%30==0:
                print('\r\n########Epoch %d train file %d Total Profit %.4f'\
                                                    %(epoch,file_idx,Ut[0]))

            ###### State reporting code
            file_N = len(train_files)
            _progress = "\r[Epoch %d]"%epoch
            _progress += "%.2f %% out of total training files"\
                                                    %((file_idx/file_N)*100)
            _progress += " ||| Total Profit : %.4f"%Ut[0]
            _progress += " ||| Trading Frequency : %d , Change Position : %d\r"\
                                                            %(frq,cp)
            sys.stdout.write(_progress)
            sys.stdout.flush()
        
        print('\r\n Trading Bias Check',m.Policy_u.data[0])
        print('\r\n',policy_count)
        print('\r\n',price_count)
        total_data = price_count[1] + price_count[-1]
        print("\r\nTrading frequency %d %.4f %%"%(frq,(frq/total_data)*100))
    
        torch.save(m.state_dict(),'./model/%s_%d'%(m.save_model,epoch))

def run_test_m(data_dir,test_files,m):
    
    m.load_state_dict(torch.load('./model/%s'%m.load_model))
    print('Testing with model -- %s'%m.load_model)
    
    H0 = Variable(torch.zeros(m.hidden_size)).unsqueeze(0).cuda() 
    F0 = Variable(torch.FloatTensor([0.]),requires_grad=False).cuda()
    
    if m.objective_type == 'SR':
        A0 = Variable(torch.zeros(1)).cuda()
        B0 = Variable(torch.zeros(1)).cuda()
    
    policy_count = {0:0,1:0,-1:0}
    price_count = {1:0,-1:0}
    trading_frq = 0
    Ut = (torch.FloatTensor([0.]))
    TP = Variable(torch.zeros(1)).cuda()
     
    TP_record = []
    Policy_record = []
    
    for file_idx,f in enumerate(test_files):
        momentum_days = GetPastData(data_dir,train_files,file_idx)
        price_data = GetMinData(data_dir,f)
        N = price_data.shape[0]
        
        for t in range(m.input_size,N):
            if t == m.input_size:
                if m.model_type=='RNN' and file_idx==0:
                    Ht_1 = Variable(H0.data)
                if m.objective_type=='SR':
                    At_1 = A0
                    Bt_1 = B0
                Ft_1 = F0
            else:
                if m.model_type=='RNN':
                    Ht_1 = Variable(Ht.data)
                if m.objective_type=='SR':
                    At_1 = At
                    Bt_1 = Bt
                Ft_1 = Ft
                
            input = Variable(price_data[t-m.input_size:t+1]
                                                    ,requires_grad=False).cuda()
            revenue = m.GetPriceChange(input)
            momentum = m.GetMomentum(input,momentum_days,).cuda()
            Input = torch.cat([revenue,momentum])

            #Noise is added here for exploration
            noise = Variable(torch.normal(0.,torch.FloatTensor([0.5])),
                                                    requires_grad=False)
            
            if m.model_type=='MLP':
                Ft = m.GetPolicy_MLP(Ft_1,Input)
            else:
                Ft,Ht = m.GetPolicy_RNN(Input,Ht_1,Ft_1.data)
            
            #Ct = m.CheckStateChange((Ft_1,Ft))
            Ct = m.GetCostAmount((Ft_1,Ft))
            Zt = revenue[-1] 
            
            trading_fee = input[-1] * m.cost_rate
            
            if m.objective_type == 'SR':
                Rt = ((Ft_1 * Zt) - (trading_fee * Ct))
                St,At,Bt = m.GetSharpeRatio(Rt,At_1,Bt_1)
                Ut += Rt.cpu().data
            else:
                Rt = -((Ft_1 * Zt) - (trading_fee * Ct))
                TP += Rt
                Ut += -Rt.cpu().data
                 
            # Parameters should tuned during test too
            if t!=m.input_size and t%m.truncate_step==0:
                m.optimizer.zero_grad()
                if m.objective_type=='SR':
                    St.backward()
                else:
                    TP.backward()
                    TP = Variable(torch.zeros(1)).cuda()
                if m.model_type=='RNN':
                    Ht = Variable(Ht.data)
                Ft = Variable(Ft.data)
                m.optimizer.step()
                
            ### Aggregating and Recording for report
            TP_record.append(Ut[0])
            Policy_record.append(Ft.data[0])
            
            trading_frq += Ct.data[0]
            cp += (Ct.data[0]!=0)
            policy_count[Ft.data[0]]+=1
            if Zt.data[0]>0:
                price_count[1] += 1
            else:
                price_count[-1] += 1

        if file_idx%10 == 0:
            print('File %d Total Profit %.4f'%(file_idx,Ut[0]))
    
    print('Final Total Profit : ',Ut[0]) 
    print(policy_count)
    print(price_count)
    print("trading frequency",trading_frq)
    
    torch.save(torch.FloatTensor(TP_record),
                                './result/TP_%s'%m.load_model)
    torch.save(torch.FloatTensor(Policy_record),
                                './result/Policy_%s'%m.load_model)

def run_train_d(price_data,m):
    if m.more_train:
        print('Model Loaded -- %s'%m.load_model)
        m.load_state_dict(torch.load('./model/%s'%m.load_model))
    else:
        print('Train model from scratch')

    N = price_data.shape[0]
    H0 = Variable(torch.zeros(m.hidden_size)).unsqueeze(0) 
    F0 = Variable(torch.FloatTensor([0.]),requires_grad=False)
    
    for epoch in range(m.epoch):
        policy_count = {0:0,1:0,-1:0}
        price_count = {1:0,-1:0}
        trading_frq = 0
        Ut = (torch.FloatTensor([0.]))
        Bt = Variable(torch.zeros(1))
        
        for t in range(m.input_size,N):
            if t == m.input_size:
                if m.model_type == 'RNN':
                    Ht_1 = Variable(H0.data)
                Ft_1 = F0
            else:
                if m.model_type=='RNN':
                    Ht_1 = Variable(Ht.data)
                Ft_1 = Ft
                
            input = Variable(price_data[t-m.input_size:t+1]
                                                    ,requires_grad=False)
            revenue = m.GetPriceChange(input)
            
            #Noise is added here for exploration
            noise = Variable(torch.normal(0.,torch.FloatTensor([0.5])),
                                                    requires_grad=False)
            
            if m.model_type=='MLP':
                Ft = m.GetPolicy_MLP(Ft_1.data,revenue)
            else:
                Ft,Ht = m.GetPolicy_RNN(revenue,Ht_1,Ft_1.data)
            
            Ct = m.CheckStateChange((Ft_1,Ft))
            Zt = input[-1] - input[-2] 
            
            trading_fee = input[-1] * m.cost_rate
            
            Rt = (Ft_1 * Zt) - (trading_fee * Ct) 
            
            Ut += Rt.data
            Bt += -Rt
            
            # Parameter tuning
            if t!=m.input_size and t%4==0:
                m.optimizer.zero_grad()
                Bt.backward()
                Bt = Variable(torch.zeros(1))
                m.optimizer.step()
            
            # Aggregating and Recording for report
            trading_frq += Ct
            policy_count[Ft.data[0]]+=1
            if Zt.data[0]>0:
                price_count[1] += 1
            else:
                price_count[-1] += 1

            if t%500 == 0:
                print('Epoch %d step %d Total Profit %.4f'%(epoch,
                                                        t,Ut[0]))
        print(policy_count)
        print(price_count)
        print("trading frequency",trading_frq)
    
    torch.save(m.state_dict(),'./model/%s'%m.save_model)

def run_test_d(price_data,m):
    
    m.load_state_dict(torch.load('./model/%s'%m.load_model))
    print('Testing with model -- %s'%m.load_model)
    
    N = price_data.shape[0]
    H0 = Variable(torch.zeros(m.hidden_size)).unsqueeze(0) 
    F0 = Variable(torch.FloatTensor([0.]),requires_grad=False)
    
    policy_count = {0:0,1:0,-1:0}
    price_count = {1:0,-1:0}
    trading_frq = 0
    Ut = (torch.FloatTensor([0.]))
    Bt = Variable(torch.zeros(1))
    
    TP_record = []
    Policy_record = []
    
    for t in range(m.input_size,N):
        if t == m.input_size:
            if m.model_type=='RNN':
                Ht_1 = Variable(H0.data)
            Ft_1 = F0
        else:
            if m.model_type=='RNN':
                Ht_1 = Variable(Ht.data)
            Ft_1 = Ft
            
        input = Variable(price_data[t-m.input_size:t+1]
                                                ,requires_grad=False)
        revenue = m.GetPriceChange(input)
        
        #Noise is added here for exploration
        noise = Variable(torch.normal(0.,torch.FloatTensor([0.5])),
                                                requires_grad=False)
        
        if m.model_type=='MLP':
            Ft = m.GetPolicy_MLP(Ft_1.data,revenue)
        else:
            Ft,Ht = m.GetPolicy_RNN(revenue,Ht_1,Ft_1.data)
        
        Ct = m.CheckStateChange((Ft_1,Ft))
        Zt = input[-1] - input[-2] 
        
        trading_fee = input[-2] * m.cost_rate
        
        Rt = (Ft_1 * Zt) - (trading_fee * Ct) 
        
        Ut += Rt.data
        Bt += -Rt
        
        # Parameters should tuned during test too
        if t!=m.input_size:
            m.optimizer.zero_grad()
            Bt.backward()
            Bt = Variable(torch.zeros(1))
            m.optimizer.step()
            
        ### Aggregating and Recording for report
        TP_record.append(Ut[0])
        Policy_record.append(Ft.data[0])
        trading_frq += Ct
        policy_count[Ft.data[0]]+=1
        if Zt.data[0]>0:
            price_count[1] += 1
        else:
            price_count[-1] += 1

        if t%500 == 0:
            print('Step %d Total Profit %.4f'%(t,Ut[0]))
    
    print(policy_count)
    print(price_count)
    print("trading frequency",trading_frq)
    
    torch.save(torch.FloatTensor(TP_record),
                                './result/TP_%s'%m.load_model)
    torch.save(torch.FloatTensor(Policy_record),
                                './result/Policy_%s'%m.load_model)

