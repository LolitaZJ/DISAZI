#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:11:50 2022

DISAZI is a pipline to achieve single-station location.

DISAZI consists of DisNet and AziNet.

Train and Test AziNet

Fast run!!!
#==========================================#
# train instance data 
nohup python DISAZI_AZI_MASTER_INSTANCE.py --mode=train --epochs=100 --patience=5 --model_name=IN_AziNet_model02 --GPU=1 > INAZI02.txt 2>&1 &

#==========================================# 

@author: zhangj2
"""


# In[]
#==========================================#
# Import libs
#==========================================# 
# common
import os
import pandas as pd
import numpy as np
import datetime
import argparse
import h5py
import math
# neural
import tensorflow as tf
import keras 
from keras import backend as K
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model
# custom
from utils_master import GENDATA_azi,DataGenerator_azin,gen_scyn_data,plot_loss3,save_tr_info,read_scyn_data
from utils_master import save_azi_info,STEAD_dataset,read_scyn_data4,DataGenerator_azi
from model_set import build_azi_model
# plot
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
import matplotlib  
#matplotlib.use('Agg') 
plt.switch_backend('agg')
# 
def my_loss1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)
np.random.seed(0)
# In[]
#==========================================#
# Set GPU
#==========================================# 
def start_gpu(args):
    try:
        cuda_kernel=args.GPU
        os.getcwd()
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_kernel
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        session = tf.Session(config=config)   
        K.set_session(session)
    except:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        print('Physical GPU：', len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('Logical GPU：', len(logical_gpus))

        
#==========================================#
# Set Configures
#==========================================# 
def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="1",
                        help="set gpu ids") 
    
    parser.add_argument("--mode",
                        default="pre_train",
                        help="/train/test/predict/pre_train")
    
    parser.add_argument("--data_name",
                        default="STEAD",
                        help="test dataset name (SC/STEAD)") 
    
    parser.add_argument("--model_name",
                        default="IN_AziNet_model",
                        help="model name")  
    
    parser.add_argument("--pre_model_path",
                        default="./model/Azi_model_wt.h5",
                        help="pre-model path")  
    
    parser.add_argument("--epochs",
                        default=500,
                        type=int,
                        help="number of epochs (default: 100)")
    
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=20,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--clas",
                        default=3,
                        type=int,
                        help="number of class") 
    
    parser.add_argument("--monitor",
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='mse',
                        help="loss fucntion")  
    
    parser.add_argument("--use_multiprocessing",
                        default=True,
                        help="False,True")     
    
    parser.add_argument("--workers",
                        default=4,
                        type=int,
                        help="workers")     

    parser.add_argument("--model_dir",
                        default='./model/',
                        help="Checkpoint directory")

    parser.add_argument("--num_plots",
                        default=10,
                        type=int,
                        help="Plotting trainning results")
    
    parser.add_argument("--wave_input",
                        default=(6000,1,3),
                        type=int,
                        help="wave_input")
        
    parser.add_argument("--data_dir",
                        default="./INSTANCE/dataset/Instance_events_counts.hdf5'",
                        help="Input data file directory")
     
    parser.add_argument("--csv_dir",
                        default="./INSTANCE/dataset/metadata_Instance_events_v2.csv'",
                        help="Input csv file directory")  
    
    parser.add_argument("--rm_no",
                        default=True,
                        help="rm noise detected by STALTA")     
    
    parser.add_argument("--weight",
                        default=[0.85,0.10,0.05],
                        type=list,
                        help="weights of train,test,and validation") 
    
    parser.add_argument("--output_dir",
                        default='./Res/',
                        help="Output directory")
    
    parser.add_argument("--conf_dir",
                        default='./model_configure/',
                        help="Configure directory")    
    
    parser.add_argument("--acc_loss_fig",
                        default='./acc_loss_fig/',
                        help="acc&loss directory")    

    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="If plot figure for test")
    
    parser.add_argument("--save_result",
                        action="store_true",
                        help="If save result for test")
           
    args = parser.parse_args()
    return args

#==========================================#
# Save Configures
#==========================================# 
def set_configure(args):
    model_name=args.model_name
    pre_model_path=args.pre_model_path
    wave_input=args.wave_input
    epochs=args.epochs
    patience=args.patience
    monitor=args.monitor
    mode=args.monitor_mode
    batch_size=args.batch_size
    use_multiprocessing=args.use_multiprocessing
    workers=args.workers
    loss=args.loss
    
    if not os.path.exists(args.conf_dir):
        os.mkdir(args.conf_dir)  
    # save configure
    files=args.conf_dir+'Conf_%s.txt'%args.model_name
    if not os.path.exists(files):
        f1 = open(files,'w')    
        f1.write('Model: %s'%model_name+'\n')
        f1.write('Pre model path: %s'%pre_model_path+'\n')
        f1.write('epochs: %d'%epochs+'\n')
        f1.write('batch_size: %d'%batch_size+'\n')
        f1.write('monitor: %s'%monitor+'\n')
        f1.write('mode: %s'%mode+'\n')
        f1.write('patience: %d'%patience+'\n')
        f1.write('wave_input: %s'%str(wave_input)+'\n')
        f1.write('loss: %s'%loss+'\n')
        f1.write('workers: %d'%workers+'\n')
        f1.write('use_multiprocessing: %s'%use_multiprocessing+'\n')
        f1.close()
#==========================================#
# Main function
#==========================================#     
def main(args):
    
   
    if args.mode=='train' or args.mode=='pre_train':
        print('#==========================================#')
        print(args.mode)
        
        if args.mode=='train':
            file_name=args.data_dir
            evt_csv=args.csv_dir
            evt_meta = pd.read_csv(evt_csv, keep_default_na=False)            
            evt_meta['year']=evt_meta['source_origin_time'].str.split('-').str[0].astype('int')
            df=evt_meta[ evt_meta.year<=2017 ]
            
            df=df[(df.path_ep_distance_km<240) & (df.source_type=='earthquake')]
            dist=df.path_ep_distance_km.tolist()
            baz=df.path_backazimuth_deg.tolist()
            df = df.sample(frac=1,random_state=0)
            trace_nm=df['trace_name'].tolist()
            nn=len(df)
            ev_list_train = trace_nm[:int(nn*0.1)]
            ev_list_validation = trace_nm[int(nn*0.1):int(nn*0.2)]
            ev_list_test = trace_nm[int(nn*0.2):]
 
            # Mk generator
            train_generator = DataGenerator_azin(df,ev_list_train,file_name, batch_size=args.batch_size)
            validation_generator = DataGenerator_azin(df,ev_list_validation,file_name, batch_size=args.batch_size)
            test_generator = DataGenerator_azin(df,ev_list_test,file_name, batch_size=args.batch_size)
            
            # Steps/Epoches
            steps_per_epoch=int(len(ev_list_train)//args.batch_size)
            validation_steps=int(len(ev_list_validation)//args.batch_size)            
            
        else:
            csv_file= args.csv_dir
            file_name= args.data_dir
            df = pd.read_csv(csv_file)
            # shuffle
            df = df.sample(frac=1,random_state=0)
            # separate
            df2 = df[(df.trace_category == 'noise')] 
            df1 = df[(df.trace_category == 'earthquake_local')] 
            
            # train 0.85 test 0.15 validation 0.05
            ev_list_train = df1.iloc[0:int(len(df1)*0.85)]['trace_name'].tolist() #+ df2.iloc[0:int(len(df2)*0.85)]['trace_name'].tolist()
            ev_list_validation = df1.iloc[int(len(df1)*0.95):]['trace_name'].tolist() #+df2.iloc[int(len(df2)*0.95):]['trace_name'].tolist()
            ev_list_test = df1.iloc[int(len(df1)*0.85):int(len(df1)*0.95)]['trace_name'].tolist() #+df2.iloc[int(len(df2)*0.85):int(len(df2)*0.95)]['trace_name'].tolist()
            
            #====================================#
            train_generator = DataGenerator_azi(ev_list_train,file_name, batch_size=args.batch_size)
            validation_generator = DataGenerator_azi(ev_list_validation,file_name, batch_size=args.batch_size)
            test_generator = DataGenerator_azi(ev_list_test,file_name, batch_size=args.batch_size)
            
            steps_per_epoch=int(len(ev_list_train)//args.batch_size)
            validation_steps=int(len(ev_list_validation)//args.batch_size)

        # Build model
        ## train_pre_model
        if args.mode=='pre_train':
            input_shape=(6000,3)
            model1=build_azi_model(input_shape)
        else:
            input_shape=(6000,3)
            model1=build_azi_model(input_shape) 
            model1.load_weights(args.pre_model_path)

        try:
            model1.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy']) #lr=0.001
        except:
            model1.compile(loss='mae',optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])
        saveBestModel = ModelCheckpoint('./model/%s.h5'%args.model_name,monitor=args.monitor, verbose=1, 
                                        save_best_only=True, mode=args.monitor_mode)
        estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
        callbacks_list = [saveBestModel,estop]
        
        # Fit
        print('#==========================================#')
        print('Training~')        
        begin = datetime.datetime.now() 
        history_callback=model1.fit_generator(
                                              generator=train_generator, 
                                              steps_per_epoch= steps_per_epoch,                      
                                              epochs=args.epochs, 
                                              verbose=1,
                                              callbacks=callbacks_list,
                                              use_multiprocessing=args.use_multiprocessing,
                                              workers=args.workers,
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps)
                                    
        end = datetime.datetime.now()
        print(end-begin)
    
        # Plot and save acc & loss
        if not os.path.exists(args.acc_loss_fig):
            os.mkdir(args.acc_loss_fig)      
        plot_loss3(history_callback,model=args.model_name,save_path=args.acc_loss_fig)
        
        # Save training info 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name
        save_tr_info(history_callback,file_path)
        
        # Evaluate and predict
        dis_model=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf})
        
        scores = dis_model.evaluate_generator(generator=test_generator,
                                              workers=args.workers,
                                              use_multiprocessing=args.use_multiprocessing,
                                              verbose=1)   
        
        print('Testing accuracy: %.4f and loss: %.4f ' %(scores[1],scores[0]))
        f1 = open(file_path,'a+')
        f1.write('============evaluate==============='+'\n')
        f1.write('Testing accuracy: %.4f and loss: %.4f'%(scores[1],scores[0])+'\n')
        f1.close()
        # Save model weight
        dis_model.save_weights('./model/%s_wt.h5'%args.model_name)
        
    # prediect
    if args.mode=='predict':
        try:
            model1=load_model('./model/%s.nn'%args.model_name,custom_objects={'tf':tf,'my_loss1':my_loss1})
        except:
            model1=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf,'my_loss1':my_loss1})
            
        save_path='./Res/%s/'%args.model_name
        if not os.path.exists(save_path):
            os.mkdir(save_path) 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name

        # mk data
        if args.data_name=='SC':
            
            file_name=args.data_dir
            evt_csv=args.csv_dir
            evt_meta = pd.read_csv(evt_csv, keep_default_na=False)            
            evt_meta['year']=evt_meta['source_origin_time'].str.split('-').str[0].astype('int')
            df=evt_meta[ evt_meta.year<=2017 ]
            
            df=df[(df.path_ep_distance_km<240) & (df.source_type=='earthquake')]
            dist=df.path_ep_distance_km.tolist()
            baz=df.path_backazimuth_deg.tolist()
            df = df.sample(frac=1,random_state=0)
            trace_nm=df['trace_name'].tolist()
            nn=len(df)
            ev_list_train = trace_nm[:int(nn*0.1)]
            ev_list_validation = trace_nm[int(nn*0.1):int(nn*0.2)]
            ev_list_test = trace_nm[int(nn*0.2):]
            
            test_gen = DataGenerator_azin(df,ev_list_test,file_name, batch_size=len(ev_list_test))

            tmp2=iter(test_gen)
            tmp=next(tmp2)
            test_data1=tmp[0]['wave_input']
            test_label1=tmp[1]['main_output']

        else:
            csv_file=args.csv_dir
            file_name=args.data_dir         
            _,_,ev_list_test=STEAD_dataset(csv_file,weight=[0.85,0.1,0.05],mo='azi')
            ev_test=ev_list_test.copy()
            np.random.shuffle(ev_test)
            
            test_gen = DataGenerator_azi(ev_test,file_name, batch_size=len(ev_test))
            tmp2=iter(test_gen)
            tmp=next(tmp2)
            test_data1=tmp[0]['wave_input']
            test_label1=tmp[1]['main_output']
        
        # predict
        y_pred=model1.predict(test_data1,verbose=1,batch_size=1024)
        score=model1.evaluate(test_data1,test_label1,batch_size=1024)

        y_pred_du=[]
        y_true_du=[]     
        for i in range(len(y_pred)):
            y_pred_du.append(math.atan2(y_pred[i,0],y_pred[i,1]))
            y_true_du.append(math.atan2(test_label1[i,0],test_label1[i,1]))
        # true label
        a=np.zeros((len(y_true_du)))
        for i in range(len(y_true_du)):
            if y_true_du[i]<0:
                a[i]=y_true_du[i]*180/math.pi+360
            else:
                a[i]=y_true_du[i]*180/math.pi 
                
        # predict label  
        b=np.zeros((len(y_pred_du)))
        for i in range(len(y_pred_du)):
            if y_pred_du[i]<0:
                b[i]=y_pred_du[i]*180/math.pi+360
            else:
                b[i]=y_pred_du[i]*180/math.pi
                
        # cal errot        
        err_baz=(b-a)
        err_du=[]
        for i in err_baz:
            if i>180:
                err_du.append(i-360)
            elif i<-180:
                err_du.append(i+360)
            else:
                err_du.append(i)
                
        err_du=np.array(err_du) 
        
        mse=np.mean(abs(err_du))
        std=np.var(abs(err_du))
        var=np.std(abs(err_du)) 
        print(mse,std,var)
        
        # write
        f1 = open(file_path,'a+')
        f1.write('============%s Error==============='%args.data_name+'\n')
        f1.write('Model: %s'%args.model_name+'\n')
        f1.write('MAE: %.4f'%mse+'\n')
        f1.write('STD: %.4f'%std+'\n')
        f1.write('VAR: %.4f'%var+'\n')
        f1.write('Loss of test data: %.4f'%score[0]+'\n')
        f1.write('Acc of test data: %.4f'%score[1]+'\n')            
        f1.close()         
        
        # plt error hist
        font={'family':'Times New Roman','weight':'normal','size':18}   
        figure, ax = plt.subplots(figsize=(9,6)) 
        plt.hist(err_du,12,color='gray',alpha=0.9)
        plt.xlabel('Error of back-azimuth (degree)',font)
        plt.ylabel('Frequency',font)
        plt.xlim([-180,180])
        my_x_ticks = np.arange(-180, 181, 60)
        plt.xticks(my_x_ticks)
#        miloc = plt.MultipleLocator(30)
#        ax.xaxis.set_minor_locator(miloc)
#        ax.grid(axis='x', which='minor',ls='--')
#        miloc = plt.MultipleLocator(200)
#        ax.yaxis.set_minor_locator(miloc)
#        ax.grid(axis='y', which='minor',ls='--')
        plt.grid(ls='--')
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
        plt.savefig(save_path+'AziNet_Error_hist_%s.png'%args.data_name,dpi=600)
        plt.show()
        
        # F vs T       
        font={'family':'Times New Roman','weight':'normal','size':18}
        figure, ax = plt.subplots(figsize=(6,6))
#        plt.plot(a,b,k.',alpha=0.9)
        if args.data_name=='SC':
            plt.plot(a,b,'k.',alpha=0.9)
#            plt.scatter(a,b,s=1,c='k',alpha = 0.9)
        else:
            plt.scatter(a,b,s=1,c='k',alpha = 0.2)
#        plt.plot(a,b,'k.',alpha = 0.9)
        plt.xlabel('True back-azimuth',font)
        plt.ylabel('Predicted back-azimuth',font)
        my_x_ticks = np.arange(0, 360, 50)
        plt.xticks(my_x_ticks)
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
        plt.xlim([0,360])
        plt.ylim([0,360])
        plt.plot([0,360],[0,360],'gray',lw=1)
        miloc = plt.MultipleLocator(50)
        ax.xaxis.set_minor_locator(miloc)
        ax.grid(axis='x', which='minor',ls='--')
        ax.yaxis.set_minor_locator(miloc)
        ax.grid(axis='y', which='minor',ls='--')
        plt.savefig(save_path+'AziNet_F_vs_T_%s.png'%args.data_name,dpi=600)
        plt.show()
        
        #  plt predict hist
        font={'family':'Times New Roman','weight':'normal','size':18}
        figure, ax = plt.subplots(figsize=(9,6))
        plt.hist(a,9,color='gray',alpha=0.9)
        plt.xlabel('Back-azimuth (degree)',font)
        plt.ylabel('Count',font)
        my_x_ticks = np.arange(0, 361, 40)
        plt.xticks(my_x_ticks)
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
        plt.xlim([0,360])
        plt.grid(ls='--')#添加网格
        plt.savefig(save_path+'AziNet_predict_hist_%s.png'%args.data_name,dpi=600)
        plt.show()
        
        #  plt true hist
        font={'family':'Times New Roman','weight':'normal','size':18}
        figure, ax = plt.subplots(figsize=(9,6))
        plt.hist(b,9,color='gray',alpha=0.9)
        plt.xlabel('Back-azimuth (degree)',font)
        plt.ylabel('Count',font)
        my_x_ticks = np.arange(0, 361, 40)
        plt.xticks(my_x_ticks)
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
        plt.xlim([0,360])
        plt.grid(ls='--')#添加网格
        plt.savefig(save_path+'AziNet_true_hist_%s.png'%args.data_name,dpi=600)
        plt.show()

        
    if args.mode=='test':
        print('#==========================================#')
        print(args.mode)      
        try:
            azi_model=load_model('./model/%s.nn'%args.model_name,custom_objects={'tf':tf,'my_loss1':my_loss1})
        except:
            azi_model=load_model('./model/%s.h5'%args.model_name,custom_objects={'tf':tf})
        
        save_path='./Res/%s/'%args.model_name
        if not os.path.exists(save_path):
            os.mkdir(save_path) 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name
          
        # mk shift dataset
        data111,label111,data222,label222,tp111,ts111=read_scyn_data4(data_path3,csv_path3,1000,11)
        data222=np.array(data222)
        label222=np.array(label222)
        y22=azi_model.predict(data222[:,:6000,:],verbose=1,batch_size=1024)
        y11=azi_model.predict(data111[:,:6000,:],verbose=1,batch_size=1024)
        
        #cal degree
        a=label222[:,1]   #true
        y_pred_du=[]   
        for i in range(len(y22)):
            y_pred_du.append(math.atan2(y22[i,0],y22[i,1]))        
        b=np.zeros((len(y_pred_du),1))
        for i in range(len(y_pred_du)):
            if y_pred_du[i]<0:
                b[i]=y_pred_du[i]*180/math.pi+360
            else:
                b[i]=y_pred_du[i]*180/math.pi
        #no shift        
        y_pred_du1=[]   
        for i in range(len(y11)):
            y_pred_du1.append(math.atan2(y11[i,0],y11[i,1]))        
        b1=np.zeros((len(y_pred_du1),1))
        for i in range(len(y_pred_du1)):
            if y_pred_du1[i]<0:
                b1[i]=y_pred_du1[i]*180/math.pi+360
            else:
                b1[i]=y_pred_du1[i]*180/math.pi            
        # plt shift data test 
        font={'family':'Times New Roman','weight':'normal','size':18}
        font1={'family':'Times New Roman','weight':'normal','size':15}
        # plt
        t=np.arange(6000)*0.01
        for kk in range(11,20,10):
            k=0
            figure, ax = plt.subplots(figsize=(10,8))
            for i in range(10):
                plt.plot(t,data222[i+kk,:6000,2]+k,'k')
                plt.text(61,i,round(b[i+kk][0],2),font1)
                plt.xlim([0,60])
                k=k+1
            plt.title('Back-azimuth: %d '%a[kk],font)
            plt.xlabel('Time (s)',font)
            plt.tick_params(labelsize=15)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels] 
            plt.savefig(save_path+'SC_shift_azi_wave_%03dk.png'%kk,dpi=600)
            plt.show() 
            

        save_azi_info(b,label222[:,1],file_path,save_path,'shifted',args.model_name)
        save_azi_info(b1,label111[:,1],file_path,save_path,'no_shift',args.model_name)
       
            
# In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    set_configure(args)
    main(args)
    




