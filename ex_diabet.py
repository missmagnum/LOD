from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from numpy import *

from gather_sda import Gather_sda
from knn import knn
import time


dat=np.loadtxt('diabetes.csv',skiprows=1,delimiter=',',usecols=(0,1,2,3,4,5,6,7))
#dat=np.loadtxt('E-GEOD-72658.txt',skiprows=1,delimiter='\t',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))


np.random.shuffle(dat)
print(dat.shape)

#dataset=dat


#################NORMALIZATION#############################
dataset=np.zeros_like(dat)
#dataset=dat/np.linalg.norm(dat)
#dataset=dat-np.mean(dat,axis=0)/np.std(dat)
mea=np.mean(dat,axis=1)
st=np.std(dat,axis=1)
for i in range(dat.shape[1]):
    dataset[:,i]=(dat[:,i]-mea)/st


############################################################


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.9)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


sda_error=[]
mean_error=[]
knn_error=[]
sdaw=[]
missing_percent=np.linspace(0.,0.9,10)
#missing_percent=[0.6,.7,.8]

def MAE(x,xr,mas):
    return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))

def MSE(x,xr,mas):
    return np.mean(np.sum((1-mas) * (x-xr)**2,axis=1))


for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
    ### without corruption in training
    train_mask =  np.random.binomial(n=1, p = 1, size = train_set.shape) #rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   
    
    
    #### SDA with test set for outputinitialization###################
    # method =  'rmsprop'  'adam'   'nes_mom'  'adadelta'  
    gather_sda=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 200,
                      pretrain_lr = 0.00001,
                      training_epochs = 200,
                      finetune_lr = 0.00001,
                      batch_size = 2,
                      hidden_size = [780,400,100,50,10,2],
                      corruption_da = [0.1,0.1,0.2,0.1, 0.1,0.1],
                      dA_initiall = True ,
                      error_known = True )    
    gather.finetuning()
    
  
    sda_error.append(MAE(test_set, gather_sda.gather_out(), test_mask))

    ########### SDA without initialization ######################
    gather_sdaw=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 200,
                      pretrain_lr = 0.00001,
                      training_epochs = 200,
                      finetune_lr = 0.00001,
                      batch_size = 2,
                      hidden_size = [780,400,100,50,10,2],
                      corruption_da = [0.1,0.1,0.2,0.1, 0.1,0.1],
                      dA_initiall = False ,
                      error_known = True )

    gather.finetuning()
  
    sdaw.append((MAE(test_set, gather_sdaw.gather_out(), test_mask))

  
    ############# KNN  & MEAN #########################
    knn_result = knn(dataset,available_mask,k=1000)
    knn_error.append(MAE(dataset,knn_result,available_mask))
 
    mean_error.append(MAE(dataset,dataset.mean(axis=0),available_mask))
    
    
   


day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result_{}_{}.dat'.format(day,tim),'w')
result.write("mean_error %s\n\nsda_error %s\n\nknn_error %s\n\n2sda %s" % (str(mean_error), str(sda_error),str(knn_error),str(sdaw)))
result.close()    

plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda')
plt.plot(missing_percent,sdaw,'--mo',label='sda_No_initial')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: diabetes')
plt.legend(loc=4,prop={'size':9})
plt.show()



print(sda_error)
print(knn_error)
print(mean_error)
print(sdaw)
