from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from numpy import *

from gather_sda import Gather_sda
from knn import knn
import time



#dat=np.loadtxt('E-GEOD-72658.txt',skiprows=1,delimiter='\t',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
#dat=np.loadtxt('diabet',skiprows=1,delimiter=',',usecols=(0,1,2,3,4,5,6,7))

dat=np.loadtxt('rheumatoid.txt',skiprows=1,delimiter='\t',usecols=range(1,9))#(1388, 8) 
np.random.shuffle(dat)
print(dat.shape)

#dataset=dat


#################NORMALIZATION#############################

## standard score
dataset = (dat-dat.mean(axis=0))/dat.std(axis=0)
### PCA--> 3


"""    
## feature scaling
dataset=np.zeros_like(dat)
for i in range(dat.shape[1]):
    dataset[:,i]=-1+ 2*(dat[:,i]-min(dat[:,i]))/(max(dat[:,i])-min(dat[:,i]))
"""
############################################################


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]



sda_error=[]
mean_error=[]
knn_error=[]


missing_percent=np.linspace(0.,0.9,10)
#missing_percent=[0.6,.7,.8]


cross_vali = 10

for kfold in range(cross_vali):
    print('...k= {} out of {} crossvalidation'.format(kfold,cross_vali))
    np.random.shuffle(train)
    percent_valid = int(train.shape[0] * 0.9)
    train_set, valid_set = train[:percent_valid] , train[percent_valid:]


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
                          pretraining_epochs = 10,
                          pretrain_lr = 0.0001,
                          training_epochs = 100,
                          finetune_lr = 0.0001,
                          batch_size = 20,
                          hidden_size = [600,100,50,4],
                          corruption_da = [ 0.1,.1,0.1,.1],
                          dA_initiall = True ,
                          error_known = True )    
        gather_sda.finetuning()


        sda_error.append(MAE(test_set, gather_sda.gather_out(), test_mask))

   
        ############# KNN  & MEAN #########################
        knn_result = knn(dataset,available_mask,k=5)
        knn_error.append(MAE(dataset,knn_result,available_mask))
 
        mean_error.append(MAE(dataset,dataset.mean(axis=0),available_mask))
    
    
 
print(sda_error)
print(knn_error)
print(mean_error)  


day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result_{}_{}.dat'.format(day,tim),'w')
result.write("mean_error %s\n\nsda_error %s\n\nknn_error %s" % (str(mean_error), str(sda_error),str(knn_error)))
result.close()    
"""
plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: diabetes')
plt.legend(loc=4,prop={'size':9})
plt.show()


##############################################


missing_percent=[ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]*10


mis=np.linspace(0.,0.9,10)
for i in range(10):
    k=i+10
    
    plt.plot(mis,mean_error[i:k],'--bo')
    plt.plot(mis,knn_error[i:k], '--go')
    plt.plot(mis,sda_error[i:k],'--ro' )
plt.show()

"""

