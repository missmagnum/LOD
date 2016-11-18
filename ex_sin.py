import numpy as np
from pylab import *
import datetime

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
        X[i,:] = np.sin(t+ph) + np.random.normal(0,0.5,nfeat) 
        if doplot:           
            plot(t,X[i,:],'r.')
    if doplot:
         plot(t,np.sin(t+ph),'b')         
 
    return X

dataset=syn_ph(4000,1000)
np.random.shuffle(dataset)
print(dataset.shape)


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


b_error=[]
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
                      method = 'nes_mom',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [100,20,2],
                      corruption_da = [0.1, 0.1,  0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
      
    knn_result = knn(dataset,available_mask)
    #########run the result for test
    dd_mask=test_mask
    dd = test_set
    
    b_error.append(sum((1-dd_mask)*((dd-gather.gather_out())**2), axis=1).mean())
    mean_error.append(sum((1-available_mask)*((dataset-dataset.mean(axis=0))**2), axis=1).mean())
    knn_error.append(sum((1-available_mask)*((dataset-knn_result)**2), axis=1).mean())
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
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [100,2],
                      corruption_da = [0.1, 0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()

    sdaw.append(sum((1-dd_mask)*((dd-gather.gather_out())**2), axis=1).mean())
    #plot(mis,sdaw[-1],'m+')


day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result_{}_{}.dat'.format(day,tim),'w')
result.write("mean_error= %s\n\nsda_error= %s\n\nknn_error= %s\n\n2sda= %s" % (str(mean_error), str(b_error),str(knn_error),str(sdaw)))
result.close()    
"""    
plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda[800,200,8]')
plt.plot(missing_percent,sdaw,'--mo',label='sda[1000,200,8]')
plt.xlabel('corruption percentage')
plt.ylabel('MSE')
plt.title('dataset: shifted sin + noise')
plt.legend(loc=4,prop={'size':9})
plt.show()

"""
print(b_error)
print(knn_error)
print(mean_error)

    
