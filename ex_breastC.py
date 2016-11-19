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
                      finetune_lr = 0.00001,
                      batch_size = 30,
                      hidden_size = [400,100,21],  #19 was good for >80%corrup
                      corruption_da = [0.1,0.1,.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
    ###########define nof K ###############
    print('... Knn calculation')
    knn_result = knn(dataset,available_mask,k=20)

    #########run the result for test
    #dd_mask=test_mask
    #dd = test_set

    def MAE(x,xr,mas):
        return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))

    
    sda_error.append(MAE(test_set, gather.gather_out(), test_mask))
    mean_error.append(MAE(dataset,dataset.mean(axis=0),available_mask))
    knn_error.append(MAE(dataset,knn_result,available_mask))
        
    #sda_error.append(sum((1-dd_mask)*(np.abs(dd-gather.gather_out())), axis=1).mean())
    #mean_error.append(sum((1-available_mask)*(np.abs(dataset-dataset.mean(axis=0))), axis=1).mean())
    #knn_error.append(sum((1-available_mask)*(np.abs(dataset-knn_result)), axis=1).mean())
  


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
"""

