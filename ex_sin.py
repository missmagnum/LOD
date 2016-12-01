from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from gather_sda import Gather_sda
from knn import knn
import time


def syn_ph(nsamp,nfeat,doplot=False):
    """
    t = np.linspace(0,2*np.pi,100)
    uni=np.random.uniform(0,2*np.pi)
    uniform= np.sin(t+uni)
    plt.plot(t,np.sin(t),'r',t+uni,uniform,'b',t+uni,uniform+np.random.normal(0,.5,100),'bo')
    """
    
    X = np.zeros((nsamp,nfeat))
    t = np.linspace(0,2*np.pi,nfeat)
    if doplot:
        figure(1)
        clf()
    for i in range(nsamp):
        ph = np.random.uniform(0,2*np.pi)
        #ph = np.random.uniform(-3,3)
        X[i,:] = np.sin(t+ph) + np.random.normal(0,.3,nfeat) 
        if doplot:           
            plot(t,X[i,:],'r.')
    if doplot:
         plot(t,np.sin(t+ph),'b')         
 
    return X

dat=syn_ph(1000,200)
print(dat.shape)
data_name = str('sine')

dataset=dat


#################NORMALIZATION#############################

"""

## standard score
dataset = (dat-dat.mean(axis=0))/dat.std(axis=0)  ### PCA--> 22 out of 134 over 1 var

   
## feature scaling
dataset=np.zeros_like(dat)
for i in range(dat.shape[1]):
    dataset[:,i]=-1+ 2*(dat[:,i]-min(dat[:,i]))/(max(dat[:,i])-min(dat[:,i]))
"""
############################################################


sda2_error=[]
sda_error=[]
mean_error=[]
knn_error=[]
sdaw=[]
missing_percent=np.linspace(0.1,0.9,9)
#missing_percent=[0.1,0.5,0.6]




def MAE(x,xr,mas):
    return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))
    #return np.sum((1-mas) * np.abs(x-xr))/np.sum(1-mas)

def MSE(x,xr,mas):
    return np.mean(np.sum((1-mas) * (x-xr)**2,axis=1))

np.random.shuffle(dataset)
percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
    

cross_vali = 50

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
                          pretraining_epochs =200,
                          pretrain_lr = 0.0001,
                          training_epochs = 300,
                          finetune_lr = 0.0001,
                          batch_size = 50,
                          hidden_size = [250,100,2],#,10,2],  # 1st lay_unit: 4/3*input_size[400,100,20,3
                          corruption_da = [0.2,0.1, 0.1,0.1,.1,.2,.1],
                          drop = [0. ,0.3, 0.,0.0,0.,0.],
                          dA_initiall = True ,
                          error_known = True ,
                          activ_fun = T.tanh)  #T.nnet.sigmoid)

        gather.finetuning()

        ###########define nof K ###############
       
        k_neib = 50
        print('... Knn calculation with {} neighbor'.format(k_neib))
        knn_result = knn(test_set,test_mask,k=k_neib)
        
        #########run the result for test
        gather2=Gather_sda(dataset = test_set*test_mask,
                          portion_data = data,
                          problem = 'regression',
                          available_mask = mask,
                          method = 'adam',
                          pretraining_epochs =300,
                          pretrain_lr = 0.0001,
                          training_epochs = 300,
                          finetune_lr = 0.0001,
                          batch_size = 100,
                          hidden_size = [250,100,2],  # 1st lay_unit: 4/3*input_size[400,100,20,3
                          corruption_da = [0.2,0.1, 0.1,0.1,.1,.2,.1],
                          drop = [0. ,0.3, 0.,0.0,0.,0.],
                          dA_initiall = False ,
                          error_known = True ,
                          activ_fun = T.tanh)  #T.nnet.sigmoid)

        gather2.finetuning()

 

        sda2_error.append(MSE(test_set, gather2.gather_out(), test_mask))
        sda_error.append(MSE(test_set, gather.gather_out(), test_mask))
        mean_error.append(MSE(dataset,dataset.mean(axis=0),available_mask))
        knn_error.append(MSE(test_set,knn_result,test_mask))

        print('sda_error= ',sda_error[-1])
        print('knn_error= ',knn_error[-1])
        print('mean_error= ',mean_error[-1])  

    
print('sda2_error= ',sda2_error)
print('sda_error= ',sda_error)
print('knn_error= ',knn_error)
print('mean_error= ',mean_error)  


 
  

day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result/result_{}_{}.dat'.format(day,tim),'w')
result.write('name of the data-without regularization: {} with k={} for knn\n\n'.format(data_name,k_neib))
result.write("mean_error= %s\n\nsda_error= %s\n\nknn_error= %s\n\nsda2_error= %s\n" % (str(mean_error), str(sda_error),str(knn_error),str(sda2_error)))
result.close()

""" 

sda2_error=np.array(sda2_error)
sda_error=np.array(sda_error)
knn_error=np.array(knn_error)
mean_error=np.array(mean_error)
mean_error=mean_error.reshape(50,9)
sda_error=sda_error.reshape(50,9)
knn_error=knn_error.reshape(50,9)
sda2_error=sda2_error.reshape(50,9)
missing_percent=np.linspace(0.1,0.9,9)


mean_con=np.std(mean_error)
knn_con=np.std(knn_error)
sda_con=np.std(sda_error)
sda2_con=np.std(sda2_error)


plt.figure()
plt.errorbar(missing_percent,np.mean(mean_error,axis=0),yerr=mean_con,fmt='--o',label='mean_row')
plt.errorbar(missing_percent,np.mean(knn_error,axis=0),yerr=knn_con,fmt='--o',label='knn' )
plt.errorbar(missing_percent,np.mean(sda_error,axis=0),yerr=sda_con,fmt='--o',label='sda')
plt.errorbar(missing_percent,np.mean(sda2_error,axis=0),yerr=sda2_con,fmt='--o',label='sda_NOinitial')

plt.xlabel('Corruption Fraction')
plt.ylabel('Mean Squared Error')
plt.title('dataset: Synthetic')
plt.legend(loc=4,prop={'size':9})
plt.show()


"""

