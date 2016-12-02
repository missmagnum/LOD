from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from gather_sda import Gather_sda
from knn import knn
import time
import gzip, pickle

import copy

def MAE(x,xr,mas):
    return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))

def MSE(x,xr,mas):
    return np.mean(np.sum((1-mas) * (x-xr)**2,axis=1))


def block_mask(datashape,mask,mis):
    n=int(mis*28)
    block=[0]*28
    for row in range(datashape):
        ran=np.random.randint(100,700,size=n)
        for r in ran:
            mask[row,r:r+28]=block
    return mask


def mnist_block(mean_data,knn_data,train_set, valid_set, test_set, mis,k_neib):

  
    train_mask=block_mask(train_set.shape[0], np.ones_like(train_set), mis)
    valid_mask=block_mask(valid_set.shape[0], np.ones_like(valid_set), mis)
    test_mask=block_mask(test_set.shape[0], np.ones_like(valid_set), mis)

    data = (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask

    ###knn
    t0=time.time()
    print('... Knn calculation with {} neighbor'.format(k_neib))
    knn_result = knn(knn_data , test_mask ,k = k_neib)
    tknn=time.time()-t0
   

    ###sda
    t0=time.time()    
    gather=Gather_sda(dataset = test_set*test_mask,
                          portion_data = data,
                          problem = 'regression',
                          available_mask = mask,
                          method = 'adam',
                          pretraining_epochs = 200,
                          pretrain_lr = 0.0001,
                          training_epochs = 300,
                          finetune_lr = 0.0001,
                          batch_size = 200,
                          hidden_size = [1000, 500,300,100, 10],  #4/3*input_siz(784)
                          drop = [0. ,0., 0.,0., 0., 0.],
                          corruption_da = [0.1,0.2,.1,0.2,.1,.2,.1],
                          dA_initiall = True ,
                          error_known = True ,
                          activ_fun = T.tanh)  #T.tanh
    gather.finetuning()
    tsda=time.time()-t0
       

    #print('time_knn',tknn,'time_sda',tsda)

    sda_er = MSE(test_set, gather.gather_out(), test_mask)
    kn_er = MSE(test_set,knn_result,test_mask)
    mean_er = MSE(mean_data,train_set.mean(axis=0),train_mask)
    
    return(sda_er,kn_er,mean_er)



    
    """
    ###plot
    subplot(141)
    imshow(train_set[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    title('sample')
    subplot(142)
    corrup=train_set[200:210]*train_mask[200:210]
    imshow(corrup.reshape((280, 28)), cmap = cm.Greys_r)
    subplot(143)
    imshow(gather.gather_out()[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    subplot(144)
    imshow(knn_result[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    show()
    """

   
  
if __name__ == "__main__":
    ###data
    f = gzip.open('mnist.pkl.gz', 'rb')
    (train_set, _ ), (valid_set, _ ), (test_set, _ )= pickle.load(f, encoding='latin1')
    f.close()
    train_set, valid_set, test_set = 1-train_set, 1-valid_set, 1-test_set  ####Black to white
    #knn_data = np.split(train_set, 10)[0]
    data_name=str('mnist')

    sda_error=[]   
    knn_error=[]
    mean_error=[]

    mean_data = copy.copy(train_set)
    knn_data = copy.copy(test_set)
    
    ###k-neigbor####
    k_neib = 50
    
    missing_percent=np.linspace(0.1,0.9,9)  
    missing_percent=[0.1,0.5,0.7]
    
    for mis in missing_percent:
        print('missing percentage: ',mis)       
        np.random.shuffle(train_set)
        sd,knn,mean = mnist_block(mean_data,knn_data,train_set, valid_set, test_set, mis , k_neib)
        sda_error.append(sd)
        knn_error.append(knn)
        mean_error.append(mean)
        
        print('sda_error= ',sda_error[-1])
        print('knn_error= ',knn_error[-1])
        print('mean_error= ',mean_error[-1])  



print('sda_error= ',sda_error)
print('knn_error= ',knn_error)
print('mean_error= ',mean_error)  


    
day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result/result_{}_{}_{}.dat'.format(data_name,day,tim),'w')
result.write('name of the data: {} with k={} for knn\n\n'.format(data_name,k_neib))
result.write("mean_error= %s\n\nsda_error= %s\n\nknn_error= %s" % (str(mean_error), str(sda_error),str(knn_error)))
result.close()


        

"""
moheeeeeeeeeem
plot(missing_percent,mean,marker='d',color='b',label='mean_row')
plot(missing_percent,knn_error,marker='p',color='g',label='knn')
plot(missing_percent,sda_error,marker='o',color='r',label='sda')
xlabel('corruption percentage')
ylabel('MSE')
title('dataset: MNIST')
legend(loc=4,prop={'size':9})
show()

"""

        

"""
missing_percent=np.linspace(0.1,0.9,9)
sda_error = [ 0.736476 , 1.85685, 3.00893, 4.12659 , 5.60219 , 6.30348 , 8.02204 , 9.71651 ,11.1634]
knn_error = [1.96055 , 4.74457 , 7.46039,  10.5575 , 13.9904 , 16.8772 ,  21.7978, 26.9399,31.4934 ]

plot(missing_percent,sda_error,'r',label='sda')
plot(missing_percent,knn_error,'g',label='knn')
xlabel('corruption percentage')
ylabel('MSE')
title('dataset: mnist')
legend(loc=4,prop={'size':9})
show()



#################################  new############

missing percentage:  0.1
... Knn calculation with 50 neighbor
... pre-training the model
... getting the finetuning functions
sda_error=  6.01451
knn_error=  4.88066
mean_error=  10.505



"""

