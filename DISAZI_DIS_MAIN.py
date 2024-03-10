#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 21:02:47 2022

DISAZI is a pipline to achieve single-station location.

DISAZI consists of DisNet and AziNet.

Train and Test DisNet

Fast run!!!
#==========================================#

python DISAZI_DIS_MAIN.py --mode=pre_train --epochs=100 --patience=5 --model_name=DisNet_model

python DISAZI_DIS_MAIN.py --mode=predict --model_name=DisNet_model


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
# neural
import tensorflow as tf
import keras 
from keras import backend as K
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model
# custom
from utils_master import GENDATA_dis,DataGenerator_dis,gen_scyn_data,plot_loss3,save_tr_info,read_scyn_data
from utils_master import save_test_info,STEAD_dataset
from model_set import build_dis_model
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
                        default="2",
                        help="set gpu ids") 
    
    parser.add_argument("--mode",
                        default="train",
                        help="/train/test/predict/")
    
    parser.add_argument("--data_name",
                        default="STEAD",
                        help="test dataset name (SC/STEAD)") 
    
    parser.add_argument("--model_name",
                        default="DisNet_model",
                        help="model name")  
    
    parser.add_argument("--pre_model_path",
                        default="./model/PRE_DIS_MODEL.nn",
                        help="pre-model path")
    
    parser.add_argument("--epochs",
                        default=100,
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
                        default=10,
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
                        default='mae',
                        help="loss fucntion")  
    
    parser.add_argument("--use_multiprocessing",
                        default=True,
                        help="False,True")     
    
    parser.add_argument("--workers",
                        default=16,
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
                        default="./data/merged.hdf5",
                        help="Input data file directory")
    
    parser.add_argument("--csv_dir",
                        default="./data/merged.csv",
                        help="Input csv file directory")     
    
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
        if not os.path.exists('./model/'):
            os.mkdir('./model/')        
        
        if args.mode=='train':
            print('set your own generator')
            # Get data list
            # your own dataset
            
            # Mk generator
            # your generator
            
            # Steps/Epoches
            # your steps_per_epoch
            # your validation_steps             
            
        else:
            csv_file=("/data2/share/STEAD/merged.csv")
            file_name="/data2/share/STEAD/merged.hdf5"
            df = pd.read_csv(csv_file)
            # shuffle
            df = df.sample(frac=1,random_state=0)
            # separate
            df1 = df[(df.trace_category == 'noise')] 
            df2 = df[(df.trace_category == 'earthquake_local')] 
            # train 0.85 test 0.15 validation 0.05
            ev_list_train = df1.iloc[0:int(len(df1)*0.85)]['trace_name'].tolist() + df2.iloc[0:int(len(df2)*0.85)]['trace_name'].tolist()
            ev_list_validation = df1.iloc[int(len(df1)*0.95):]['trace_name'].tolist()+df2.iloc[int(len(df2)*0.95):]['trace_name'].tolist()
            ev_list_test = df1.iloc[int(len(df1)*0.85):int(len(df1)*0.95)]['trace_name'].tolist() +df2.iloc[int(len(df2)*0.85):int(len(df2)*0.95)]['trace_name'].tolist()
            
            #====================================#
            train_generator = DataGenerator_dis(ev_list_train,file_name, batch_size=args.batch_size)
            validation_generator = DataGenerator_dis(ev_list_validation,file_name, batch_size=args.batch_size)
            test_generator = DataGenerator_dis(ev_list_test,file_name, batch_size=args.batch_size)
            
            steps_per_epoch=int(len(ev_list_train)//args.batch_size)
            validation_steps=int(len(ev_list_validation)//args.batch_size)
        
        # Build model
        ## train_pre_model
        if args.mode=='pre_train':
            input_shape=(6000,3)
            model1=build_dis_model(input_shape)
        else:
            model1=load_model(args.pre_model_path,custom_objects={'tf':tf})
#            model1=load_model(args.pre_model_path,custom_objects={'my_loss1':my_loss1})
        
        try:
            model1.compile(loss=args.loss,optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy']) #lr=0.001
        except:
            model1.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])
        
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
        dis_model=load_model('./model/%s.h5'%args.model_name,custom_objects={'my_loss1':my_loss1})
        
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
            os.makedirs(save_path) 
        file_path=args.conf_dir+'Conf_%s.txt'%args.model_name

        # mk data
        if args.data_name=='SC':
            # your generator
            # your own data
            print('set your own generator')
        else:
            csv_file=("/data2/share/STEAD/merged.csv")
            file_name="/data2/share/STEAD/merged.hdf5"          
            _,_,ev_list_test=STEAD_dataset(csv_file,weight=[0.85,0.1,0.05])
            ev_test=ev_list_test.copy()
            np.random.shuffle(ev_test)
            
            test_gen = DataGenerator_dis(ev_test,file_name, batch_size=len(ev_test))
            tmp2=iter(test_gen)
            tmp=next(tmp2)
            test_data1=tmp[0]['wave_input']
            test_label1=tmp[1]['main_output']
        
        # predict
        y=model1.predict(test_data1,verbose=1,batch_size=1024)
        score=model1.evaluate(test_data1,test_label1,batch_size=1024)

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
        f1.write('============%s Error==============='%args.data_name+'\n')
        f1.write('Model: %s'%args.model_name+'\n')
        f1.write('MAE: %.4f'%mse+'\n')
        f1.write('STD: %.4f'%std+'\n')
        f1.write('VAR: %.4f'%var+'\n')
        f1.write('Loss of test data: %.4f'%score[0]+'\n')
        f1.write('Acc of test data: %.4f'%score[1]+'\n')            
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
        plt.savefig(save_path+'DisNet_predict_hist_%s.png'%args.data_name,dpi=600) 
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
        plt.savefig(save_path+'DisNet_F_vs_T_%s.png'%args.data_name,dpi=600)
        plt.show()
        
        #  plt hist
        inx1=[i for i in range(len(test_label1)) if test_label1[i]>0 and test_label1[i]<120]
        font={'family':'Times New Roman','weight':'normal','size':18}
        figure, ax = plt.subplots(figsize=(7,5))
        plt.hist(test_label1[inx1],12,color='gray',alpha=0.9)
        plt.xlabel('Distance (km)',font)
        plt.ylabel('Count',font)
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.xlim([0,120])
        plt.grid(linestyle='-.')
        plt.savefig(save_path+'DisNet_data_hist_%s.png'%args.data_name,dpi=600)
        plt.show()
        
        #  plt relative curve            
        inx = [i for i in range(len(test_label1)) if test_label1[i]>0  ]
        y_true1 = test_label1[inx]
        y_pred1 = y[inx,0]
        
        font={'family':'Times New Roman','weight':'normal','size':18}
        cer_dis=(y_pred1[:]-y_true1[:])/y_true1[:]
        print('Relative Error: %.2f'%np.median(cer_dis))
        er_al=[]
        for i in range(10,120,10):
            index=[j for j,dis in enumerate(y_true1[:]) if dis <i]
            er_d=abs(y_pred1[index]-y_true1[index])/y_true1[index]
            er_al.append(np.median(er_d))
        dis_rg=np.arange(10,120,10)
        
        figure, ax = plt.subplots(figsize=(7,5))
        plt.plot(dis_rg,er_al,'k',linewidth=2)
        plt.xlabel('Distance (km)',font)
        plt.ylabel('Relative error ',font)
        plt.xlim([9.5,110])
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
        plt.grid(ls='--')#添加网格
        plt.savefig(save_path+'DisNet_Pre_Test_Rela_%s.png'%args.data_name,dpi=600)
        plt.show()            
            
       
            
# In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    set_configure(args)
    main(args)
    




