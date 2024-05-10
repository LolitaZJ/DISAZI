#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:27:42 2024

Use the well trained modell to INSTANCE dataset.

single-station location
Multi-station location

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
from utils_master import DataGenerator_disin,DataGenerator_azin,gen_scyn_data,plot_loss3,save_tr_info,read_scyn_data
from utils_master import save_azi_info,STEAD_dataset,read_scyn_data4
from model_set import build_azi_model
# plot
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.style.use('default')
import matplotlib  
# matplotlib.use('Agg') 

def my_loss1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)
np.random.seed(0)

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
# import libs
def ls_loc(x,y,d,err0=0.01,k0=1000,a=0.001):
    err_al=[]
    px=0
    py=0
    x0=np.mean(x)
    y0=np.mean(y)
    k=0
    err1=0
    m=len(x)
    while(1):
        err=0
        for i in range(m):
            r=((x0-x[i])**2+(y0-y[i])**2)**(0.5)*111   
            if r<=0:  
                r=0.001
            px=px+(r-d[i])*(x0-x[i])*111/r/m
            py=py+(r-d[i])*(y0-y[i])*111/r/m               
            err=err+0.5/m*(r-d[i])**2
                   
        x0=x0-a*px
        y0=y0-a*py
        k+=1
        
        if k>1:
            if err1>err:
                err1=err
                x1=x0
                y1=y0 
                k1=k
            
        else:
            err1=err
            x1=x0
            y1=y0
            k1=k
    #    print('iter:%d x:%.2f y:%.2f err:%.2f' %(k,x0,y0,err))
        err_al.append(err)
        
        if err<err0 or k>k0:
#            print('iter:%d x:%.2f y:%.2f err:%.2f' %(k,x0,y0,err))
            break
    return x0,y0,err_al,x1,y1,err1,k1 

def cal_min_dis(x,y,std_loc):
    mis_dis=((std_loc[0][0]-x)**2+(std_loc[0][1]-y)**2)**0.5+((std_loc[1][0]-x)**2+(std_loc[1][1]-y)**2)**0.5
    return mis_dis

def plot_cir(c,r):
    theta = np.arange(0, 2*np.pi, 0.01)
    x = c[0] + r * np.cos(theta)
    y = c[1] + r * np.sin(theta)
    plt.plot(x,y)  

def plot_circle_map3(result,dis_data,lon1,lat1,use_name1,
x_lon,y_lat,x_true,y_true,x1,y1,flag=1,path=None,ss=-1,title=None):
    font2={'family':'Times New Roman','weight':'bold','size':15}
    # px_max=math.ceil(np.max([np.max(np.array(lon1)),x1,x_true] ) +0.5)
    # px_min=math.floor(np.min([np.min(np.array(lon1)),x1,x_true] ) -0.5)
    # py_max=math.ceil(np.max([np.max(np.array(lat1)),y1,y_true] ) +0.5)
    # py_min=math.floor(np.min([np.min(np.array(lat1)),y1,y_true] ) -0.5)    
    
    # scale_y=(px_max-px_min)/(py_max-py_min)
    # py=int(6/scale_y)
    py=6
    cc=[ i+1 for i in result]
    # figure, ax =plt.subplots(figsize=(6,py)) 
    figure, ax =plt.subplots()
    # plt.scatter(lon1, lat1, s=50,marker='^',c=cc,linewidths=0)
    plt.scatter(lon1, lat1, s=50,marker='^',color='k',linewidths=0)
    # for i in range(len(use_name1)):
    #     plt.text(lon1[i]+0.2,lat1[i]-0.2,use_name1[i],font2,color='k')
    if ss==-1:    
        for i in range(len(dis_data)):
            plot_cir([x_lon[i],y_lat[i]],dis_data[i]/111)
    else:
        
        plot_cir([x_lon[ss],y_lat[ss]],dis_data[ss]/111)

    if flag==1:
        plt.plot(x_true,y_true,'y*',markersize=15)

    plt.plot(x1,y1,'r*',markersize=15)
    # plt.xlim([px_min,px_max])
    # plt.ylim([py_min,py_max])
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]    
    plt.xlabel('Longitude',font2)
    plt.ylabel('Latitude',font2) 
    if title is not None:
        plt.title('Event id: %s'%title)

    ax = plt.gca()
    ax.set_aspect(1)
    if path is not None:
        plt.savefig(path,dpi=600) 
    plt.show()
    
# In[] set gpu
cuda_kernel="1"
os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_kernel
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)   
K.set_session(session)   
plt.style.use('default') 

# In[] load data
file_name='/data2/zhangj2/INSTANCE/dataset/Instance_events_counts.hdf5'
evt_csv='/data2/zhangj2/INSTANCE/dataset/metadata_Instance_events_v2.csv'
evt_meta = pd.read_csv(evt_csv, keep_default_na=False)            
evt_meta['year']=evt_meta['source_origin_time'].str.split('-').str[0].astype('int')
df=evt_meta[ evt_meta.year>2017 ]
df=df[(df.path_ep_distance_km<240) & (df.source_type=='earthquake')]

tmp=list(set(df.source_id.tolist()))
mag=[]
for eid in tmp:
    mag.append(df[df.source_id==eid].source_magnitude.tolist()[0])
plt.hist(mag)

# In[] get three stations 
df2=[]
for eid in tmp[:]: 
    df1=df[ (df.source_id==eid) & (df.station_channels=='HH') ]
    if len(df1)>3:
        df2.append(df1.sort_values(by="path_ep_distance_km").iloc[:3]) # nn=0
df3=pd.concat(df2,axis=0)

trace_nm=df3['trace_name'].tolist()
dist=df3.path_ep_distance_km.tolist()
baz=df3.path_backazimuth_deg.tolist()
x=df3.station_longitude_deg.tolist()
y=df3.station_latitude_deg.tolist()
sx=df3.source_longitude_deg.tolist()
sy=df3.source_latitude_deg.tolist()
sta_nm=df3.station_code.tolist()
evt_nm=df3.source_id.tolist()

# In[] get generator
dis_generator = DataGenerator_disin(df3,trace_nm,file_name, batch_size=512)
azi_generator = DataGenerator_azin(df3,trace_nm,file_name, batch_size=512) 

# In[]  transfer learning model
azi_name = 'IN_AziNet_model02'
dis_name = 'IN_DisNet_model02'
azi_model=load_model('./model/%s.h5'%azi_name,custom_objects={'tf':tf})
dis_model=load_model('./model/%s.h5'%dis_name,custom_objects={'tf':tf})

# In[] predict the dis and azi
azi_pred=azi_model.predict_generator(azi_generator)
dis_pred=dis_model.predict_generator(dis_generator)

# In[] calculate azi
azi_pred_du=[]
for i in range(len(azi_pred)):
    azi_pred_du.append(math.atan2(azi_pred[i,0],azi_pred[i,1]))

b=np.zeros((len(azi_pred_du)))
for i in range(len(azi_pred_du)):
    if azi_pred_du[i]<0:
        b[i]=azi_pred_du[i]*180/math.pi+360
    else:
        b[i]=azi_pred_du[i]*180/math.pi
# In[] single location
b1=[]
b2=[]
b3=[]
b4=[]
for i in range(len(azi_pred)):
    b1.append(dis_pred[i,0]/110*math.sin(azi_pred_du[i]))
    b2.append(dis_pred[i,0]/110*math.cos(azi_pred_du[i]))
     
b3=np.array(x[:len(b1)])+np.array(b1)
b4=np.array(y[:len(b2)])+np.array(b2)

# In[] location error
err=[sx[:len(b3)]-b3,sy[:len(b4)]-b4]
err_loc=np.linalg.norm(np.array(err),ord=2, axis=0)

font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(8,6))
plt.hist(err_loc,24)
plt.xlabel('Location error (degree)',font)
plt.ylabel('Count',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([0,2])
plt.grid()#添加网格
plt.savefig('./instance_fig/ins_location_1_org.png',dpi=600)
plt.show()

print(np.mean(err_loc))
print(np.std(err_loc))


# In[] plot dis
font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(6,6))
#plt.plot(label2[:,0],y,'o')
plt.plot([0,110],[0,110],'gray',lw=1)
# plt.scatter(label1[:,0],sedn_pred[:,0],c='b',alpha = 0.2)
plt.scatter(dist[:len(dis_pred)],dis_pred[:,0],s=1,c='k',alpha = 0.9)
plt.xlabel('True distance (km)',font)
plt.ylabel('Predict distance (km)',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([0,110])
plt.ylim([0,110])
plt.grid() 
plt.savefig('./instance_fig/ins_dis_1_org.png',dpi=600)
plt.show()

err_dis=dist[:len(dis_pred)]-dis_pred[:,0]

print('Mean:%.2f'%np.mean(err_dis) )
print('MAE:%.2f'%np.mean(abs(err_dis)))
print('Std:%.2f'%np.std(err_dis) )

font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(8,6))
plt.hist(err_dis,49)
plt.xlabel('Distance error (km)',font)
plt.ylabel('Count',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([-50,50])
plt.grid() 
plt.savefig('./instance_fig/ins_dis_2_org.png',dpi=600)
plt.show()

# In[]  plot azi  
font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(6,6))
#plt.plot(label2[:,0],y,'o')
plt.plot([0,360],[0,360],'gray',lw=1)
# plt.scatter(label1[:,1],b,c='b',alpha = 0.2)
plt.scatter(baz[:len(b)],b,s=1,c='k',alpha = 0.9)
plt.xlabel('True back-azimuth',font)
plt.ylabel('Predict back-azimuth',font)

plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([0,360])
plt.ylim([0,360])
plt.grid() 
plt.savefig('./instance_fig/ins_azi_1_org.png',dpi=600)
plt.show()

err_azi=[]
for i in baz[:len(b)]-b:
    if i>180:
        i=-i+360
    if i<-180:
        i=i+180
    err_azi.append(i)

print('Mean:%.2f'%np.mean(err_azi) )
print('MAE:%.2f'%np.mean(np.abs(err_azi)))
print('Std:%.2f'%np.std(err_azi) )


font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(8,6))
plt.hist(err_azi)
plt.xlabel('Back-azimuth error (degree)',font)
plt.ylabel('Count',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([-180,180])
# plt.ylim([0,360])
plt.grid() 
plt.savefig('./instance_fig/ins_azi_2_org.png',dpi=600)
plt.show()

# In[] location with 2 or 3 stations
np.random.seed(7)
stas=3
mun=1
evt=5802
err4=[]
loc4=[]
for i in range(evt): 
    stan=np.random.randint(2,4)
    if stan==3:
        cx,cy,er,cx1,cy1,err1,k1=ls_loc(x[i*stas:(i+1)*stas],y[i*stas:(i+1)*stas],dis_pred[i*stas:(i+1)*stas],err0=0.01,k0=2000,a=0.00001)

        err4.append(np.sqrt((sx[i*stas]-cx1)**2+(sy[i*stas]-cy1)**2))
        loc4.append([sx[i*stas],sy[i*stas],cx1,cy1])
    if stan==2:
        stai=[0,1,2]
        np.random.shuffle(stai)
        a=i*stas
        stai2 = list(map(lambda x:x+a,stai))[:2]
        dis=np.sum(dis_pred[stai2])
        dis1=np.abs(dis_pred[stai2[0]]-dis_pred[stai2[1]])
        sta_dist=((x[stai2[0]]-x[stai2[1]])**2+(y[stai2[0]]-y[stai2[1]])**2)**0.5*111

        if dis<sta_dist or  dis1 > sta_dist:
            cx,cy,er,cx1,cy1,err1,k1=ls_loc(np.array(x)[stai2],np.array(y)[stai2],np.array(dis_pred)[stai2],err0=0.01,k0=2000,a=0.00001)
            x0=cx1
            y0=cy1    
        else:
            cx,cy,er,cx1,cy1,err1,k1=ls_loc(np.array(x)[stai2],np.array(y)[stai2],np.array(dis_pred)[stai2],err0=0.01,k0=2000,a=0.00001)
            kk=(y[stai2[0]]-y[stai2[1]])/(x[stai2[0]]-x[stai2[1]])
            c=y[stai2[0]]-kk*x[stai2[0]]

            cx2=cx1-(2*kk*(kk*cx1-cy1+c))/(kk*kk+1)
            cy2=cy1-(-2*(kk*cx1-cy1+c))/(kk*kk+1)
            std_loc=[]
            for ii in range(2):
                std_loc.append([b3[stai2[ii]],b4[stai2[ii]]])
            tmp1=cal_min_dis(cx1,cy1,std_loc)
            tmp2=cal_min_dis(cx2,cy2,std_loc)
            if tmp1<tmp2:
                x0=cx1
                y0=cy1
            else:
                x0=cx2
                y0=cy2
        err4.append(np.sqrt((sx[i*stas]-x0)**2+(sy[i*stas]-y0)**2))
        loc4.append([sx[i*stas],sy[i*stas],x0,y0])

err4=np.array(err4)
print(np.mean(err4))
print(np.std(err4))

font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(8,6))
plt.hist(err4,49)
plt.xlabel('Location error (degree)',font)
plt.ylabel('Count',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([0,1])
plt.grid() 
plt.savefig('./instance_fig/ins_location_rand_err.png',dpi=600)
plt.show()

# location
loc4=np.array(loc4)
font={'family':'Times New Roman','weight':'normal','size':15}
figure, ax =plt.subplots(1,2)
ax[0].scatter(loc4[:,0],loc4[:,1],s=1,marker='*',color='k',label="True")
# ax[0].scatter(sx[:len(b3)],sy[:len(b4)],s=1,marker='*',color='k',label="True")
labels = ax[0].get_xticklabels() + ax[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax[0].set_xlabel('Longitude',font)
ax[0].set_ylabel('Latitude',font)  
ax[0].set_xlim([5,20])
ax[0].set_ylim([35,48])
ax[0].legend()
ax[0].grid()
ax[1].scatter(loc4[:,2],loc4[:,3],s=1,marker='*',color='r',label="Predicted")
labels = ax[1].get_xticklabels() + ax[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels] 
ax[1].set_xlabel('Longitude',font)
ax[1].set_xlim([5,20])
ax[1].set_ylim([35,48])
ax[1].legend()
ax[1].grid()
plt.savefig('./instance_fig/ins_location_rand_loc.png',dpi=600)
plt.show()



# In[] location one station
np.random.seed(7)
stas=3
err1=[]
loc1=[]
for i in range(5802):
    stan=np.random.randint(3)
    err1.append(err_loc[i*stas+stan])
    loc1.append([sx[i*stas+stan],sy[i*stas+stan],b3[i*stas+stan],b4[i*stas+stan]])


font={'family':'Times New Roman','weight':'normal','size':18}
figure, ax = plt.subplots(figsize=(8,6))
plt.hist(err1,24)
plt.xlabel('Location error (degree)',font)
plt.ylabel('Count',font)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]  
plt.xlim([0,2])
plt.grid() 
plt.savefig('./instance_fig/ins_location_one_error_rand.png',dpi=600)
plt.show()

print(np.mean(err1))
print(np.std(err1))

loc1=np.array(loc1)
font={'family':'Times New Roman','weight':'normal','size':15}
figure, ax =plt.subplots(1,2)
ax[0].scatter(loc1[:,0],loc1[:,1],s=1,marker='*',color='k',label="True")
# ax[0].scatter(sx[:len(b3)],sy[:len(b4)],s=1,marker='*',color='k',label="True")
labels = ax[0].get_xticklabels() + ax[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax[0].set_xlabel('Longitude',font)
ax[0].set_ylabel('Latitude',font)  
ax[0].set_xlim([5,20])
ax[0].set_ylim([35,48])
ax[0].legend()
ax[0].grid()
# ax[0].
# ax[0].set_aspect(1) 
# plt.subplot(122)
ax[1].scatter(loc1[:,2],loc1[:,3],s=1,marker='*',color='r',label="Predicted")
# ax[1].scatter(b3,b4,s=1,marker='*',color='r',label="Predicted")
labels = ax[1].get_xticklabels() + ax[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels] 
ax[1].set_xlabel('Longitude',font)
# plt.ylabel('Latitude',font) 
# ax = plt.gca()
# ax[1].set_aspect(1)
ax[1].set_xlim([5,20])
ax[1].set_ylim([35,48])
# plt.style.use('default')
ax[1].legend()
ax[1].grid()
plt.savefig('./instance_fig/ins_location_one_loc_rand.png',dpi=600)
plt.show()

# In[] plot all location
loc4=np.array(loc4)
font={'family':'Times New Roman','weight':'normal','size':15}
figure, ax =plt.subplots(1,3,figsize=(8,4),dpi=600)
ax[0].scatter(loc4[:,0],loc4[:,1],s=1,marker='*',color='k',label="True")
ax[0].set_ylabel('Latitude',font)  
ax[1].scatter(loc1[:,2],loc1[:,3],s=1,marker='*',color='g',label="One")
ax[2].scatter(loc4[:,2],loc4[:,3],s=1,marker='*',color='r',label="Multi")

for i in range(3):
    labels = ax[1].get_xticklabels() + ax[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    ax[i].set_xlabel('Longitude',font)
    ax[i].set_xlim([5,20])
    ax[i].set_ylim([35,48])
    ax[i].legend()
    ax[i].grid()
plt.savefig('./instance_fig/ins_location_rand_all.png',dpi=600)
plt.show()

# In[]




