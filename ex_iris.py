from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from numpy import *

from gather_sda import Gather_sda
from knn import knn
import time

 
dat=np.loadtxt('Iris.csv',skiprows=1,delimiter=',',usecols=(1,2,3,4,5))

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
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


sda_error=[]
mean_error=[]
knn_error=[]
sdaw=[]
missing_percent=np.linspace(0.,0.9,10)
#missing_percent=[0.6,.7,.8]

for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
    ### without corruption in training
    train_mask =  np.random.binomial(n=1, p = 1, size = train_set.shape) #rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   
    
    
    #### SDA with test set for output
    # method =  'rmsprop'  'adam'   'nes_mom'  'adadelta'  
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 100,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [160,50,2],
                      corruption_da = [0.1,0.2,  0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
      
    knn_result = knn(dataset,available_mask)
    #########run the result for test
    dd_mask=test_mask
    dd = test_set
    
    sda_error.append(sum((1-dd_mask)*(np.abs(dd-gather.gather_out())), axis=1).mean())
    mean_error.append(sum((1-available_mask)*(np.abs(dataset-dataset.mean(axis=0))), axis=1).mean())
    knn_error.append(sum((1-available_mask)*(np.abs(dataset-knn_result)), axis=1).mean())
    #plot(mis,b_error[-1],'ro')
    #plot(mis,mean_error[-1],'bo')
    #plot(mis,knn_error[-1],'g*')

    #### SDA with corruption in training
 
   
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'nes_mom',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 100,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [160,50,1],
                      corruption_da = [0.1, 0.1,0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()

    sdaw.append(sum((1-dd_mask)*(np.abs(dd-gather.gather_out())), axis=1).mean())
    #plt.plot(mis,sdaw[-1],'m+')


day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result_{}_{}.dat'.format(day,tim),'w')
result.write("mean_error %s\n\nsda_error %s\n\nknn_error %s\n\n2sda %s" % (str(mean_error), str(sda_error),str(knn_error),str(sdaw)))
result.close()    

plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda[800,200,8]')
plt.plot(missing_percent,sdaw,'--mo',label='sda[1000,200,8]')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: shifted sin + noise')
plt.legend(loc=4,prop={'size':9})
plt.show()



print(sda_error)
print(knn_error)
print(mean_error)
print(sdaw)
