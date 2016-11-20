from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt


from gather_sda import Gather_sda
from knn import knn
import time
import glob


#############################################################################################
############################################################################################
#dat=np.loadtxt('wdbc.data',skiprows=0,delimiter=',',usecols=range(2,32))  #(569, 30)  


e_name=glob.glob('E-GEOD-80233/*.txt')


prot_breac=[]    ######### Protein_breastcancer E-GEOD-80233  (945, 134) 
for i in e_name:
    prot_breac.append(np.loadtxt(i,skiprows=1,delimiter='\t',usecols=[1]))

dat=np.array(prot_breac)
np.random.shuffle(dat)
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


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


sda_error=[]
mean_error=[]
knn_error=[]
sdaw=[]
missing_percent=np.linspace(0.1,0.9,9)
#missing_percent=[0.6,.7,.8]

for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]

    #######################################
    ######## without corruption in training
    ########################################
    train_mask =  np.random.binomial(n=1, p = 1-mis, size = train_set.shape) #rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   
     ###############activation_function =T.tanh
    
    #### SDA with test set for output
    # method =  'rmsprop'  'adam'   'nes_mom'  'adadelta'  
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 10,
                      hidden_size = [800,200,100,18],  #19 was good for >80%corrup
                      corruption_da = [0.1,0.1,.1,.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
    ###########define nof K ###############
    k_neib = 10
    print('... Knn calculation with {} neighbor'.format(k_neib))
    knn_result = knn(dataset,available_mask,k=k_neib)

    #########run the result for test


    def MAE(x,xr,mas):
        return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))

    
    sda_error.append(MAE(test_set, gather.gather_out(), test_mask))
    mean_error.append(MAE(dataset,dataset.mean(axis=0),available_mask))
    knn_error.append(MAE(dataset,knn_result,available_mask))

    print('sda_error= ',sda_error[-1])
    print('knn_error= ',knn_error[-1])
    print('mean_error= ',mean_error[-1])  

    

print('sda_error= ',sda_error)
print('knn_error= ',knn_error)
print('mean_error= ',mean_error)  


    
day=time.strftime("%d-%m-%Y")
tim=time.strftime("%H-%M")
result=open('result_{}_{}.dat'.format(day,tim),'w')
result.write("mean_error %s\n\nsda_error %s\n\nknn_error %s" % (str(mean_error), str(sda_error),str(knn_error)))
result.close()


"""
plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda[800,200,8]')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: breastCancer')
plt.legend(loc=4,prop={'size':9})
plt.show()



sda_error=  [3.871869681867294, 8.2436541341801259, 12.964913741228191, 18.675575989245448, 25.131510073080488, 34.622592099472961, 46.613367289922948, 63.890278885043216, 88.559940110878642]
knn_error=  [3.20499688381648, 6.4108534236094155, 9.8306643617146943, 13.36512634122391, 17.311326069823874, 21.927842680853594, 30.761750633469379, 63.610188952641785, 82.84037353999733]
mean_error=  [9.9338877005167667, 19.601172492159652, 29.131744163946042, 39.047887549364873, 48.633334717173184, 58.842152171117533, 68.442723810961837, 78.347584073602746, 88.232847722551384]

                   pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 10,
                      hidden_size = [600,200,100,21],  #19 was good for >80%corrup
                      corruption_da = [0.1,0.1,.1,.1],
"""

