#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:35:09 2022

@author: zhangj2
"""
# In[]
import datetime
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from scipy import signal
try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
import csv
# In[] load STEAD dataset
def STEAD_dataset(csv_file,weight,no_file=None,rm_no=None,mo='dis'):

    # get STEAD data
    df = pd.read_csv(csv_file)
    # shuffle
    df = df.sample(frac=1,random_state=0)
    # separate
    df1 = df[(df.trace_category == 'noise')] 
    df2 = df[(df.trace_category == 'earthquake_local')] 
    # train 0.85 test 0.15 validation 0.05
    w1=weight[0]
    w2=weight[0]+weight[1]
    if mo=='dis':   
        ev_list_train = df1.iloc[0:int(len(df1)*w1)]['trace_name'].tolist() + df2.iloc[0:int(len(df2)*w1)]['trace_name'].tolist()
        ev_list_validation = df1.iloc[int(len(df1)*w2):]['trace_name'].tolist()+df2.iloc[int(len(df2)*w2):]['trace_name'].tolist()
        ev_list_test = df1.iloc[int(len(df1)*w1):int(len(df1)*w2)]['trace_name'].tolist() +df2.iloc[int(len(df2)*w1):int(len(df2)*w2)]['trace_name'].tolist()
    else:
        ev_list_train =  df2.iloc[0:int(len(df2)*w1)]['trace_name'].tolist()
        ev_list_validation = df2.iloc[int(len(df2)*w2):]['trace_name'].tolist()
        ev_list_test = df2.iloc[int(len(df2)*w1):int(len(df2)*w2)]['trace_name'].tolist()
          
    # remove the detect noise by STALTA
    if rm_no:
        L=np.load(no_file) # get noise by STALTA
        pk_no=L['pk_no']   
        ev_list_train=list(set(ev_list_train).difference(set(pk_no)))
        ev_list_validation =list(set(ev_list_validation).difference(set(pk_no)))
        ev_list_test=list(set(ev_list_test).difference(set(pk_no)))
    
    return ev_list_train,ev_list_validation,ev_list_test

# In[] save test info
def save_test_info(y,test_label1,file_path,save_path,name,model_name):
    # select
    inx=[i for i in range(len(test_label1)) if test_label1[i]>0 and test_label1[i]<350]
    # cal mae,std,var
    ddis=y[inx,0]-test_label1[inx]
    mse=np.mean(abs(ddis))
    std=np.std(ddis)
    var=np.var(ddis)
    print(mse,std,var)
    # write
    f1 = open(file_path,'a+')
    f1.write('============%s Error==============='%name+'\n')
    f1.write('Model: %s'%model_name+'\n')
    f1.write('MAE: %.4f'%mse+'\n')
    f1.write('STD: %.4f'%std+'\n')
    f1.write('VAR: %.4f'%var+'\n')          
    f1.close()      
    
    # plt hist
    font={'family':'Times New Roman','weight':'normal','size':18}
    figure, ax = plt.subplots(figsize=(6,6))
    plt.hist(ddis,100,color='gray',alpha=0.9)
    plt.xlabel('Error of distance (km)',font)
    plt.ylabel('Count',font)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  
    plt.xlim([-50,50])
    plt.grid(linestyle='-.')
    plt.savefig(save_path+'DisNet_predict_hist_%s.png'%name,dpi=600) 
    plt.show() 
    
    # plot true vs predict
    font={'family':'Times New Roman','weight':'normal','size':18}
    figure, ax = plt.subplots(figsize=(6,6))
    plt.scatter(test_label1[inx],y[inx,0],s=1,c='k',alpha = 0.9)
    plt.xlabel('True distance (km)',font)
    plt.ylabel('Predict distance (km)',font)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  
    plt.xlim([-10,120])
    plt.ylim([-10,120])
    dg=20
    milocx = plt.MultipleLocator(dg)
    ax.xaxis.set_major_locator(milocx)
    ax.xaxis.set_minor_locator(milocx)
    ax.grid(axis='both',which='both',linestyle='-.')#添加网格
    plt.savefig(save_path+'DisNet_F_vs_T_%s.png'%name,dpi=600)
    plt.show()    
    
# In[]
def save_azi_info(y,test_label1,file_path,save_path,name,model_name):
    # select
    inx=[i for i in range(len(test_label1)) if test_label1[i]>0 and test_label1[i]<350]
    # cal mae,std,var
    ddis=y[inx,0]-test_label1[inx]
    mse=np.mean(abs(ddis))
    std=np.std(ddis)
    var=np.var(ddis)
    print(mse,std,var)
    # write
    f1 = open(file_path,'a+')
    f1.write('============%s Error==============='%name+'\n')
    f1.write('Model: %s'%model_name+'\n')
    f1.write('MAE: %.4f'%mse+'\n')
    f1.write('STD: %.4f'%std+'\n')
    f1.write('VAR: %.4f'%var+'\n')          
    f1.close()      
    
    # plt hist
    font={'family':'Times New Roman','weight':'normal','size':18}
    figure, ax = plt.subplots(figsize=(6,6))
    plt.hist(ddis,100,color='gray',alpha=0.9)
    plt.xlabel('Error of azimuth (degree)',font)
    plt.ylabel('Count',font)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  
    plt.xlim([-50,50])
    plt.grid(linestyle='-.')
    plt.savefig(save_path+'AziNet_predict_hist_%s.png'%name,dpi=600) 
    plt.show() 
    
    # plot true vs predict
    font={'family':'Times New Roman','weight':'normal','size':18}
    figure, ax = plt.subplots(figsize=(6,6))
    plt.scatter(test_label1[inx],y[inx,0],s=1,c='k',alpha = 0.9)
    plt.xlabel('True back-azimuth (degree)',font)
    plt.ylabel('Predicted back-azimuth (degree)',font)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  
    plt.xlim([-10,120])
    plt.ylim([-10,120])
    dg=20
    milocx = plt.MultipleLocator(dg)
    ax.xaxis.set_major_locator(milocx)
    ax.xaxis.set_minor_locator(milocx)
    ax.grid(axis='both',which='both',linestyle='-.')#添加网格
    plt.savefig(save_path+'AziNet_F_vs_T_%s.png'%name,dpi=600)
    plt.show()      
    


# In[] Save training info 
def save_tr_info(history_callback,file_path):
    history_dict=history_callback.history
    loss_value=history_dict['loss'] 
    val_loss_value=history_dict['val_loss']
    try:
        acc_value=history_dict['acc']
        val_acc_value=history_dict['val_acc']
    except:
        acc_value=history_dict['accuracy']
        val_acc_value=history_dict['val_accuracy']        
    
    best_inx=np.argmin(val_loss_value)
    
    f1 = open(file_path,'a+')
    f1.write('============training==============='+'\n')
    f1.write('loss_value\n')
    f1.write('%s\n'%(str(loss_value)))
    f1.write('val_loss_value\n')
    f1.write('%s\n'%(str(val_loss_value)))
    f1.write('acc_value\n')
    f1.write('%s\n'%(str(acc_value)))
    f1.write('val_acc_value\n')
    f1.write('%s\n'%(str(val_acc_value)))
    f1.write('Training accuracy: %.4f and loss: %.4f'%(acc_value[best_inx],loss_value[best_inx])+'\n')
    f1.write('Validation accuracy: %.4f and loss: %.4f'%(val_acc_value[best_inx],val_loss_value[best_inx])+'\n')
    f1.close()

# In[] plt loss

def plot_loss3(history_callback,save_path=None,model='model'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

    history_dict=history_callback.history
    try:
        loss_value=history_dict['loss']
        val_loss_value=history_dict['val_loss']
        
        acc_value=history_dict['acc']
        val_acc=history_dict['val_acc']
    except:
        loss_value=history_dict['loss']
        val_loss_value=history_dict['val_loss']
        
        acc_value=history_dict['accuracy']
        val_acc=history_dict['val_accuracy']    
    
  
    epochs=range(1,len(acc_value)+1)
    if not save_path is None:
        np.savez(save_path+'acc_loss_%s'%model,
                 loss=loss_value,val_loss=val_loss_value,
                 acc=acc_value,val_acc=val_acc)


    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_value,'k',label='Training acc')
    plt.plot(epochs,val_acc,'k-.',label='Validation acc') 
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_%s.png'%model,dpi=600)
    plt.show()

    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_value,'k',label='Training loss')
    plt.plot(epochs,val_loss_value,'k-.',label='Validation loss')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
#    plt.ylim([0,40])
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)    
    plt.show()
    # In[]
def read_scyn_data4(data_path,csv_path,nn,nnn):
    lines_2018=[]
    tp_2018=[]
    ts_2018=[]
    with open(csv_path) as f:
        f_csv =csv.reader(f)
        for row in f_csv:
            lines_2018.append(data_path+row[0])
            tp_2018.append(int(row[1].strip('[]')))
            ts_2018.append(int(row[3].strip('[]')))
            
    begin = datetime.datetime.now()  
    test_x2=[]
    test_y2=[]
    for i in lines_2018:
        L= np.load(i)
        data=L['data']
        label=L['label'][[0,2,10,11,8,9]]
        test_x2.append(data)
        test_y2.append(label)
    test_x2=np.array(test_x2)
    test_y2=np.array(test_y2)
    tp_2018=np.array(tp_2018)
    ts_2018=np.array(ts_2018)
    
    test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
    
    test_x2=bp_filter(test_x2,2,1,45,0.01)
    test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
    test_x2=normal3(test_x2)
    
    test_x3=[]
    test_y3=[]
    
    sindex=[]
    for i in range(nn):
        if tp_2018[i]>510:
            for rdn in range(0,nnn):
                temp=np.zeros((6000,3))
                temp1=test_x2[i,4500-rdn*100:5500,:]
                temp[:1000+rdn*100,:]=temp1
                temp[1000+rdn*100:,:]=test_x2[i,500:5500-rdn*100,:]
                for jj in range(3):
                    temp[:,jj]=taper(temp[:,jj],1,300)             
                test_x3.append(temp)
                test_y3.append(test_y2[i,:])
            sindex.append(i)

    end = datetime.datetime.now()    
    print(end-begin)
    return test_x2[sindex,:,:],test_y2[sindex,:],test_x3,test_y3,tp_2018,ts_2018     
   
# In[] azi
def read_scyn_data(data_path,csv_path,nn,nnn):
    lines_2018=[]
    tp_2018=[]
    ts_2018=[]
    with open(csv_path) as f:
        f_csv =csv.reader(f)
        for row in f_csv:
            lines_2018.append(data_path+row[0])
            tp_2018.append(int(row[1].strip('[]')))
            ts_2018.append(int(row[3].strip('[]')))
            
    begin = datetime.datetime.now()  
    test_x2=[]
    test_y2=[]
    for i in lines_2018:
        L= np.load(i)
        data=L['data']
        label=L['label'][[0,5]]
        test_x2.append(data)
        test_y2.append(label)
    test_x2=np.array(test_x2)
    test_y2=np.array(test_y2)
    tp_2018=np.array(tp_2018)
    ts_2018=np.array(ts_2018)
    
    test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
    
    test_x2=bp_filter(test_x2,2,1,45,0.01)
    test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
    test_x2=normal3(test_x2)
    test_x3=[]
    test_y3=[]
    
    sindex=[]
    for i in range(nn):
        if tp_2018[i]>510:
            for rdn in range(0,nnn):
                temp=np.zeros((6000,3))
                temp1=test_x2[i,4500-rdn*100:5500,:]
                temp[:1000+rdn*100,:]=temp1
                temp[1000+rdn*100:,:]=test_x2[i,500:5500-rdn*100,:]
                for jj in range(3):
                    temp[:,jj]=taper(temp[:,jj],1,300)             
                test_x3.append(temp)
                test_y3.append(test_y2[i,:])
            sindex.append(i)
    end = datetime.datetime.now()    
    print(end-begin)
    return test_x2[sindex,:,:],test_y2[sindex,:],test_x3,test_y3,tp_2018,ts_2018


# In[]
# taper
def taper(data,n,N):
    nn=len(data)
    if n==1:
        w=math.pi/N
        F0=0.5
        F1=0.5
    elif n==2:
        w=math.pi/N
        F0=0.54
        F1=0.46
    else:
        w=math.pi/N/2
        F0=1
        F1=1
    win=np.ones((nn,1))
    for i in range(N):
        win[i]=(F0-F1*math.cos(w*(i-1)))
    win1=np.flipud(win)
    
    data1=data*win.reshape(win.shape[0],)
    data1=data1*win1.reshape(win1.shape[0],)
    return data1
    
# bandpass        
def bp_filter(data,n,n1,n2,dt):
    wn1=n1*2*dt
    wn2=n2*2*dt
    b, a = signal.butter(n, [wn1,wn2], 'bandpass')
    filtedData = signal.filtfilt(b, a, data) 
    return filtedData 
    
# In[] dis SC data generator
class GENDATA_dis(Sequence):

    def __init__(self,data_path,csv_path,batch_size=256,fg_shift=0,shuffle=True):

        self.csv_path=csv_path     
        file_index=[]
        file_tp=[]
        with open(self.csv_path) as f:
            f_csv =csv.reader(f)
            for row in f_csv:
                file_index.append(data_path+row[0])
                file_tp.append(int(row[1].strip('[]')))  
        f.close()  
         
        self.file_index =file_index
        self.file_tp=file_tp   
        self.fg_shift=fg_shift
        self.batch_size = batch_size
        self.indexes=np.arange(len(self.file_index))
        self.shuffle = shuffle        
 
    def __len__(self):
        """return: steps num of one epoch. """
        return len(self.file_index)// self.batch_size

    def __getitem__(self, index):

        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        np.random.seed(batch_inds[0])
        # get batch data file name.
        batch_index = [self.file_index[k] for k in batch_inds]
        tp_index = [self.file_tp[k] for k in batch_inds]
        
        # read batch data
        X, Y= self._read_data(batch_index,tp_index)
        
        return ({'wave_input': X}, {'main_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2 
    
    def _read_data(self, batch_index,tp_index):
        test_x2=[]
        test_y2=[]        
        #------------------------#    
        for c, evi in enumerate(batch_index):  
            L= np.load(evi)
            data=L['data']
            label=L['label'][[0,5]]
            test_x2.append(data)
            test_y2.append(label)

        test_x2=np.array(test_x2)
        test_y2=np.array(test_y2)
        
        test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
        test_x2=self._bp_filter(test_x2,2,1,45,0.01)
        test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
        test_x2=self._normal3(test_x2)

        if self.fg_shift==0:
            return test_x2,test_y2[:,0]
        
        else:
            train_x_new=[]
            for ii in range(len(test_x2)):
                temp=np.zeros((6000,3))
                if test_y2[ii,0]<30:
                    rdn=np.random.randint(0,45)
                else:
                    rdn=np.random.randint(0,25)     
                maxinx=tp_index[ii]+1000+rdn*100
                if maxinx<=6000 and rdn>0:
                    temp=np.zeros((6000,3))
                    if rdn>=3:
                        temp1=test_x2[ii,5700-rdn*100:5700,:]
                        temp[:rdn*100,:]=temp1
                        temp[rdn*100:,:]=test_x2[ii,300:6300-rdn*100,:]
                    else:
                        temp1=test_x2[ii,5400-rdn*100:5700,:]
                        temp[:rdn*100+300,:]=temp1
                        temp[rdn*100+300:,:]=test_x2[ii,300:6000-rdn*100,:]                            
                    for jj in range(3):
                        temp[:,jj]=taper(temp[:,jj],1,300)   
                else:
                    temp=test_x2[ii,:,:]   
                train_x_new.append(temp)
            train_x_new=np.array(train_x_new) 
            train_x_new=self._normal3(train_x_new)
            #-------------------------------#   
            return train_x_new, test_y2[:,0]         

# In[] dis SC data generator
class GENDATA_azi_tp(Sequence):

    def __init__(self,data_path,csv_path,batch_size=256,shuffle=True):

        self.csv_path=csv_path     
        file_index=[]
        file_tp=[]
        with open(self.csv_path) as f:
            f_csv =csv.reader(f)
            for row in f_csv:
                file_index.append(data_path+row[0])
                file_tp.append(int(row[1].strip('[]')))  
        f.close()  
         
        self.file_index =file_index
        self.file_tp=file_tp
        self.batch_size = batch_size
        self.indexes=np.arange(len(self.file_index))
        self.shuffle = shuffle        
 
    def __len__(self):
        """return: steps num of one epoch. """
        return len(self.file_index)// self.batch_size

    def __getitem__(self, index):

        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        np.random.seed(batch_inds[0])
        # get batch data file name.
        batch_index = [self.file_index[k] for k in batch_inds]
        tp_index = [self.file_tp[k] for k in batch_inds]
        
        # read batch data
        X, Y, Z= self._read_data(batch_index,tp_index)
        
        return ({'wave_input': X}, {'main_output':Y }, {'tp':Z }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2 
    
    def _read_data(self, batch_index,tp_index):
        test_x2=[]
        test_y2=[]
        test_y22=[]
        #===================================#
        for c, evi in enumerate(batch_index):
            L= np.load(evi)
            data=L['data']
            label=L['label'][[2]]
            label1=L['label'][[0]]
            test_x2.append(data)
            test_y2.append(label)
            test_y22.append(label1)

        test_x2=np.array(test_x2)
        test_y2=np.array(test_y2)
        test_y22=np.array(test_y22)
        #===================================#
        test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
        test_x2=self._bp_filter(test_x2,2,1,45,0.01)
        test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
        
        x1=[math.sin(math.radians(x)) for x in test_y2]
        x2=[math.cos(math.radians(x)) for x in test_y2]
            
        test_y=np.vstack((x1,x2)).T
        test_y=test_y*100                    
        test_x2=self._normal3(test_x2)
        #===================================#
        return test_x2[:,:6000,:],test_y, tp_index


# In[] dis SC data generator
class GENDATA_azi(Sequence):

    def __init__(self,data_path,csv_path,batch_size=256,fg_shift=0,shuffle=True):

        self.csv_path=csv_path     
        file_index=[]
        file_tp=[]
        with open(self.csv_path) as f:
            f_csv =csv.reader(f)
            for row in f_csv:
                file_index.append(data_path+row[0])
                file_tp.append(int(row[1].strip('[]')))  
        f.close()  
         
        self.file_index =file_index
        self.file_tp=file_tp   
        self.fg_shift=fg_shift
        self.batch_size = batch_size
        self.indexes=np.arange(len(self.file_index))
        self.shuffle = shuffle        
 
    def __len__(self):
        """return: steps num of one epoch. """
        return len(self.file_index)// self.batch_size

    def __getitem__(self, index):

        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        np.random.seed(batch_inds[0])
        # get batch data file name.
        batch_index = [self.file_index[k] for k in batch_inds]
        tp_index = [self.file_tp[k] for k in batch_inds]
        
        # read batch data
        X, Y= self._read_data(batch_index,tp_index)
        
        return ({'wave_input': X}, {'main_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2 
    
    def _read_data(self, batch_index,tp_index):
        test_x2=[]
        test_y2=[]
        test_y22=[]
        #===================================#
        for c, evi in enumerate(batch_index):
            L= np.load(evi)
            data=L['data']
            label=L['label'][[2]]
            label1=L['label'][[0]]
            test_x2.append(data)
            test_y2.append(label)
            test_y22.append(label1)

        test_x2=np.array(test_x2)
        test_y2=np.array(test_y2)
        test_y22=np.array(test_y22)
        #===================================#
        test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
        test_x2=self._bp_filter(test_x2,2,1,45,0.01)
        test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
        
        x1=[math.sin(math.radians(x)) for x in test_y2]
        x2=[math.cos(math.radians(x)) for x in test_y2]
            
        test_y=np.vstack((x1,x2)).T
        test_y=test_y*100                    
        test_x2=self._normal3(test_x2)
        #===================================#
        if self.fg_shift==0:
            return test_x2[:,:6000,:],test_y  
        else:
            train_x_new=[]
            for ii in range(len(test_x2)):
                temp=np.zeros((6000,3))
                if test_y22[ii]<30:
                    rdn=np.random.randint(0,45)
                else:
                    rdn=np.random.randint(0,25)     
                maxinx=tp_index[ii]+1000+rdn*100
                if maxinx<=6000 and rdn>0:
                    temp=np.zeros((6000,3))
                    if rdn>=3:
                        temp1=test_x2[ii,5700-rdn*100:5700,:]
                        temp[:rdn*100,:]=temp1
                        temp[rdn*100:,:]=test_x2[ii,300:6300-rdn*100,:]
                    else:
                        temp1=test_x2[ii,5400-rdn*100:5700,:]
                        temp[:rdn*100+300,:]=temp1
                        temp[rdn*100+300:,:]=test_x2[ii,300:6000-rdn*100,:]                            
                    for jj in range(3):
                        temp[:,jj]=taper(temp[:,jj],1,300)   
                else:
                    temp=test_x2[ii,:,:]   
                
                train_x_new.append(temp)
            train_x_new=np.array(train_x_new) 
            train_x_new=self._normal3(train_x_new)
            return train_x_new,test_y 


# In[] dis SC data

def gen_scyn_data(data_path,csv_path,batch_size=128,fg_shift=0):
    kk=0
    lines_2018=[]
    tp_2018=[]
    
    with open(csv_path) as f:
        f_csv =csv.reader(f)
        for row in f_csv:
            lines_2018.append(data_path+row[0])
            tp_2018.append(int(row[1].strip('[]')))  
    f.close()    
    
 
    while 1:
        random = np.random.RandomState(kk)
        
        random.shuffle(lines_2018)
        random.shuffle(tp_2018)
        
        index=int(len(lines_2018)/batch_size)
        for inx in range(index):
            test_x2=[]
            test_y2=[]
            
            for i in lines_2018[inx*batch_size:(inx+1)*batch_size]:
                L= np.load(i)
                data=L['data']
                label=L['label'][[0,5]]
                test_x2.append(data)
                test_y2.append(label)

            test_x2=np.array(test_x2)
            test_y2=np.array(test_y2)
            
            test_x2=test_x2.transpose(0,2,1).reshape(len(test_x2)*3,6000)
            test_x2=bp_filter(test_x2,2,1,45,0.01)
            test_x2=test_x2.reshape(int(len(test_x2)/3),3,6000).transpose(0,2,1)
            test_x2=normal3(test_x2)
            
            if fg_shift==0:
                yield ({'wave_input': test_x2[:,:6000,:]}, {'main_output': test_y2[:,0]})  
                
            else:
                train_x_new=[]
                for ii in range(len(test_x2)):
                    temp=np.zeros((6000,3))
                    if test_y2[ii,0]<30:
                        rdn=random.randint(0,45)
                    else:
                        rdn=random.randint(0,25)     
                    maxinx=tp_2018[ii]+1000+rdn*100
                    if maxinx<=6000 and rdn>0:
                        temp=np.zeros((6000,3))
                        if rdn>=3:
                            temp1=test_x2[ii,5700-rdn*100:5700,:]
                            temp[:rdn*100,:]=temp1
                            temp[rdn*100:,:]=test_x2[ii,300:6300-rdn*100,:]
                        else:
                            temp1=test_x2[ii,5400-rdn*100:5700,:]
                            temp[:rdn*100+300,:]=temp1
                            temp[rdn*100+300:,:]=test_x2[ii,300:6000-rdn*100,:]                            
                        for jj in range(3):
                            temp[:,jj]=taper(temp[:,jj],1,300)   
                    else:
                        temp=test_x2[ii,:,:]   
                    train_x_new.append(temp)
                train_x_new=np.array(train_x_new) 
                #-------------------------------#   
                yield ({'wave_input': train_x_new}, 
                        {'main_output': test_y2[:,0]}) 

        kk=kk+1 


    
    
# In[] normalization
#def normal3(data):  
#    data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
#    for i in range(data.shape[0]):
#        #data1=data[i,:]-np.mean(data[i,:])
#        data1=data[i,:,:]
#        x_max=np.max(abs(data1))
#        if x_max!=0.0:
#            data2[i,:,:]=data1/x_max 
#    return data2
def normal3(data):  
    data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[0]):
        data1=data[i,:,:]
        # mean
        data1=data1-np.mean(data1,axis=0)
        # std
        std_data = np.std(data1, axis=0, keepdims=True)
        std_data[std_data == 0] = 1
        data1 /= std_data
        # max
        x_max=np.max(abs(data1),axis=0)
        x_max[x_max == 0]=1
        data2[i,:,:]=data1/x_max 
        
    return data2 

class DataGenerator_dis(Sequence):

    def __init__(self,file_index, file_name,batch_size=256,shuffle=True):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.file_index = file_index
        self.indexes=np.arange(len(self.file_index))
        self.shuffle = shuffle
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.file_index)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        # get batch data inds.
        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        # get batch data file name.
        batch_index = [self.file_index[k] for k in batch_inds]
        # read batch data
        X, Y= self._read_data(batch_index)
        return ({'wave_input': X}, {'main_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2 
    
    def _read_data(self, batch_index):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            images: (batch_size, (image_shape)).
            labels: (batch_size, (label_shape)). """
        batch_size=self.batch_size
        train_x=np.zeros((batch_size,6000,3))
#        pt=np.zeros((batch_size,))  
#        st=np.zeros((batch_size,))
        dis=np.zeros((batch_size,))
        dtfl = h5py.File(self.file_name, 'r')
        #------------------------#    
        for c, evi in enumerate(batch_index):  
            dataset = dtfl.get('data/'+str(evi)) 
#            pt[c] = dataset.attrs['p_arrival_sample']
            if evi.split('_')[-1] == 'NO':
                dis[c]=0
            else:    
                dis[c]= dataset.attrs['source_distance_km']
            
            train_x[c,:,:3] = np.array(dataset)

        train_x[:,:,:3]=self._normal3(train_x[:,:,:3])

        return train_x,dis
    
# In[]  
import math        
class DataGenerator_azi(Sequence):

    def __init__(self,file_index, file_name,batch_size=256,shuffle=True):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.file_name = file_name
        self.file_index = file_index
        self.indexes=np.arange(len(self.file_index))
        self.shuffle = shuffle
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.file_index)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        # get batch data inds.
        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        # get batch data file name.
        batch_index = [self.file_index[k] for k in batch_inds]
        # read batch data
        X, Y= self._read_data(batch_index)
        return ({'wave_input': X}, {'main_output':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            # mean
            data1=data1-np.mean(data1,axis=0)
            # std
            std_data = np.std(data1, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data1 /= std_data
            # max
            x_max=np.max(abs(data1),axis=0)
            x_max[x_max == 0]=1
            data2[i,:,:]=data1/x_max 
            
        return data2 
    
    def _read_data(self, batch_index):

        batch_size=self.batch_size
        dtfl = h5py.File(self.file_name, 'r')
        train_x=np.zeros((batch_size,6000,3))
        baz=np.zeros((batch_size,)) 

        #------------------------#    
        for c, evi in enumerate(batch_index):
            dataset = dtfl.get('data/'+str(evi)) 
            event_lat=dataset.attrs['source_latitude']
            event_lon=dataset.attrs['source_longitude']
            station_lat=dataset.attrs['receiver_latitude']
            station_lon=dataset.attrs['receiver_longitude']
            
            distance_m, azimuth, back_azimuth = obspy.geodetics.base.gps2dist_azimuth(
                                                                        float(event_lat), 
                                                                        float(event_lon),
                                                                        float(station_lat), 
                                                                        float(station_lon), 
                                                                        a=6378137.0, 
                                                                        f=0.0033528106647474805)

            baz[c] = back_azimuth
            train_x[c,:,:3] = np.array(dataset)
                
        baz=[ [math.sin(baz_deg*math.pi/180),math.cos(baz_deg*math.pi/180)] for baz_deg in baz ]
        baz=np.array(baz)
    
        return self._normal3(train_x), 100*baz            
            
