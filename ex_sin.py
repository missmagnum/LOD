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

def syn_cos(nsamp,nfeat,doplot=False):
    
    X = np.zeros((nsamp,nfeat))
    t = np.linspace(0,2*np.pi,nfeat)
    
    u1 = np.sin(t)*np.random.uniform(0,2*np.pi)
    u2 = np.sin(3*linspace(2*np.pi,3*np.pi,nfeat))*np.random.uniform(0,2*np.pi)
    
    if doplot:
        figure(1)
        clf()
    for i in range(nsamp):
        ph = np.random.uniform(0,2*np.pi)
        X[i,:] = u1+u2 + np.random.normal(0,0.5,nfeat) 
        if doplot:           
            plot(t,X[i,:],'r.')
    if doplot:
         plot(t,np.sin(t+ph),'b')         
 
    return X


"""
nfeat=5
nsamp=100
x=syn_ph(nsamp,nfeat)


for i in range(nfeat):
    plot([i]*nsamp,x[:,i],'r.')
#plot(x[:,1],x[:,3],'r.')
#################

x=syn_ph(1000,10)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2])

#######################
########ino neshoon bede


t = np.linspace(0,2*np.pi,100)
uniform1= np.sin(t+np.pi/2)
uniform2=np.sin(t+np.pi)
f,ax = plt.subplots(2, sharex=True)
sam =uniform1+np.random.normal(0,.5,100)
ax[0].plot(t,np.sin(t+0.1),'r',t,uniform1,'b',t,sam,'bo')
ax[1].plot( np.linspace(2.9,3.1,100),sam,'ro' )

f1,ax1 = plt.subplots(2, sharex=True)
ax1[0].plot(t,np.sin(t),'r',t,uniform2,'b',t,uniform2+np.random.normal(0,.5,100),'bo')
ax1[1].plot( np.linspace(2.9,3.1,100),uniform2+np.random.normal(0,.5,100),'ro'  )


plt.show()



######################
t = np.linspace(0,2*np.pi,100)
uniform1 = np.sin(t)*np.random.uniform(0,2*np.pi)
uniform2 = np.cos(t)*np.random.uniform(0,2*np.pi)
figure(1)
plot(t,np.sin(t),'r',t,uniform1,'b',t,uniform1+np.random.normal(0,.5,100),'bo')
figure(2)
plot(t,np.sin(t),'r',t,uniform2,'b',t,uniform2+np.random.normal(0,.5,100),'bo')
show()

"""

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
                      method = 'adam',
                      pretraining_epochs = 10,
                      pretrain_lr = 0.0001,
                      training_epochs = 100,
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
    plot(mis,b_error[-1],'ro')
    plot(mis,mean_error[-1],'bo')
    plot(mis,knn_error[-1],'g*')

    #### SDA with corruption in training
 
   
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 10,
                      pretrain_lr = 0.0001,
                      training_epochs = 100,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [100,2],
                      corruption_da = [0.1, 0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()

    sdaw.append(sum((1-dd_mask)*((dd-gather.gather_out())**2), axis=1).mean())
    plot(mis,sdaw[-1],'m+')


tim=time.strftime("%d/%m/%Y")
result=open('result_{}.dat'.format(tim),'w')
result.write("mean_error %s\nsda_error %s\nknn_error %s\n2sda %s" % (str(mean_error), str(b_error),str(knn_error),str(sdaw)))
    
plot(missing_percent,mean_error,'b',label='mean_row')
plot(missing_percent,knn_error,'g',label='knn' )
plot(missing_percent,b_error,'r',label='sda[100,20,2]')
plot(missing_percent,sdaw,'m',label='sda[100,2]')
xlabel('corruption percentage')
ylabel('MSE')
title('dataset: shifted sin + noise')
legend(loc=4,prop={'size':9})
print(b_error)
print(knn_error)
print(mean_error)
show()
    
