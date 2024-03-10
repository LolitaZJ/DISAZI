DISAZI is a pipline to achieve single-station location.
DISAZI consists of DisNet and AziNet.
Author: Ji Zhang  
Date: 2024.03.08  
Version 1.0.0 

## Install
>      
>      python==3.8.8
>      tensorflow-gpu==2.7.0
>      keras==2.7.0

## DATA
> you need change the file path   
--data_dir: STEAD.h5 dataset  
--cvr_dir: STEAD.csv file

! you can load <code>[STEAD Data](https://github.com/smousavi05/STEAD)</code> or use `https://github.com/smousavi05/STEAD` to get dataset.

## Fast run!!!

> train DisNet
---   
    python DISAZI_DIS_MAIN.py --mode=pre_train --epochs=100 --patience=5 --model_name=DisNet_model

> predict DisNet
--- 
    python DISAZI_DIS_MAIN.py --mode=predict --model_name=DisNet_model

> train AziNet
---
    python DISAZI_AZI_MAIN.py --mode=pre_train --model_name=AziNet_model --epochs=100 --patience=5

> predict AziNet
--- 
   	python DISAZI_AZI_MAIN.py --mode=predict --model_name=AziNet_model 