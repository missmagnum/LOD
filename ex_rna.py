from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from gather_sda import Gather_sda
from knn import knn
import time

dat=np.loadtxt('rna_data.txt').T    #(172, 5000)
#dataset=(X-np.mean(X,axis=0))/np.std(X,axis=0)

print(dat.shape)

#dataset=dat


#################NORMALIZATION#############################

## standard score
dataset = (dat-dat.mean(axis=0))/dat.std(axis=0)  ### PCA--> 22 out of 134 over 1 var


"""    
## feature scaling
dataset=np.zeros_like(dat)
for i in range(dat.shape[1]):
    dataset[:,i]=-1+ 2*(dat[:,i]-min(dat[:,i]))/(max(dat[:,i])-min(dat[:,i]))
"""
############################################################



sda_error=[]
mean_error=[]
knn_error=[]
sdaw=[]
missing_percent=np.linspace(0.1,0.9,9)
missing_percent=[0.1,0.5,0.8]


def MAE(x,xr,mas):
    return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))
    #return np.sum((1-mas) * np.abs(x-xr))/np.sum(1-mas)

def MSE(x,xr,mas):
    return np.mean(np.sum((1-mas) * (x-xr)**2,axis=1))


np.random.shuffle(dataset)
percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
    

cross_vali = 1

for kfold in range(cross_vali):

    np.random.shuffle(train)
    percent_valid = int(train.shape[0] * 0.8)
    train_set, valid_set = train[:percent_valid] , train[percent_valid:]


    print('...kfold= {} out of {} crossvalidation'.format(kfold+1,cross_vali))
    for mis in missing_percent:
        print('missing percentage: ',mis)


        available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
        rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
       
        train_mask =  np.random.binomial(n=1, p = 1-mis, size = train_set.shape) #rest_mask[:percent_valid]
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
                          pretraining_epochs = 200,
                          pretrain_lr = 0.0001,
                          training_epochs = 300,
                          finetune_lr = 0.0001,
                          batch_size = 10,  #10
                          hidden_size = [5000,1000], #[3000,1000,450,168],
                          corruption_da = [0.1, 0.1, .1, 0.1],
                          drop = [0.1 ,0., 0.,0.,0.,0.],
                          dA_initiall = True ,
                          error_known = True ,
                          activ_fun = T.tanh)  #T.nnet.sigmoid)

        gather.finetuning()
        ###########define nof K ###############
        k_neib = 30
        print('... Knn calculation with {} neighbor'.format(k_neib))
        knn_result = knn(dataset,available_mask,k=k_neib)

        #########run the result for test

        sda_error.append(MAE(test_set, gather.gather_out(), test_mask))
        mean_error.append(MAE(dataset,dataset.mean(axis=0),available_mask))
        knn_error.append(MAE(dataset,knn_result,available_mask))

        print('sda_error= ',sda_error[-1])
        print('knn_error= ',knn_error[-1])
        print('mean_error= ',mean_error[-1])  

    

print('sda_error= ',sda_error)
print('knn_error= ',knn_error)
print('mean_error= ',mean_error)  


"""    
day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result/result_{}_{}.dat'.format(day,tim),'w')
result.write('name of the data: {} with k={} for knn\n\n'.format(data_name,k_neib))
result.write("mean_error= %s\n\nsda_error= %s\n\nknn_error= %s" % (str(mean_error), str(sda_error),str(knn_error)))
result.close()
"""

