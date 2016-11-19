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
                      batch_size = 5,
                      hidden_size = [400,100,21],  #19 was good for >80%corrup
                      corruption_da = [0.1,0.1,.1],
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



sda_error=  [0.0, 4.0373230473166686, 8.6391201303436187, 13.987815029050715, 19.316061324443002, 27.401109964367784, 34.576769459871421, 45.848975034057055, 64.912280949745977, 85.867797854139724]
knn_error=  [0.0, 3.2239342553787758, 6.3587991700998661, 9.9424136683517457, 13.383944546058032, 17.322183179812015, 21.691424523815666, 30.697269379329331, 63.725304693900412, 82.948171246346362]
mean_error=  [0.0, 9.8420046178497458, 19.515621979481068, 29.247118318942814, 39.149222483825014, 49.072122138920648, 58.793330123512035, 68.755998416924584, 78.553190438712463, 87.990986727414281]

pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 10,
                      hidden_size = [400,100,21],  #19 was good for >80%corrup
                      corruption_da = [0.1,0.1,.1],





"""

