from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from gather_sda import Gather_sda
from knn import knn
import time
import glob


#############################################################################################
############################################################################################
#dat=np.loadtxt('wdbc.data',skiprows=0,delimiter=',',usecols=range(2,32))  #(569, 30)  


e_name=glob.glob('E-GEOD-80233/*.txt')

data_name=str('Protein_breastcancer E-GEOD-80233')


prot_breac=[]    ######### Protein_breastcancer E-GEOD-80233  (945, 134) 
for i in e_name:
    prot_breac.append(np.loadtxt(i,skiprows=1,delimiter='\t',usecols=[1]))

dat=np.array(prot_breac)
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
#missing_percent=[0.6,.7,.8]




cross_vali = 30

for kfold in range(cross_vali):
    np.random.shuffle(dataset)
    percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
    train, test_set = dataset[:percent] ,dataset[percent:]
    

    np.random.shuffle(train)
    percent_valid = int(train.shape[0] * 0.8)
    train_set, valid_set = train[:percent_valid] , train[percent_valid:]


    def MAE(x,xr,mas):
        return np.mean(np.sum((1-mas) * np.abs(x-xr),axis=1))

    def MSE(x,xr,mas):
        return np.mean(np.sum((1-mas) * (x-xr)**2,axis=1))

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
                          batch_size = 12,
                          hidden_size = [600,200,100,60,40,21],  #19 was good for >80%corrup
                          corruption_da = [0.1,0.2,.1,0.2,.1,.2,.1],
                          dA_initiall = True ,
                          error_known = True ,
                          activ_fun = T.tanh)  #T.nnet.sigmoid)

        gather.finetuning()
        ###########define nof K ###############
        k_neib = 60
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
result=open('result/result_{}_{}.dat'.format(day,tim),'w')
result.write('name of the data: {} with k={} for knn\n\n'.format(data_name,k_neib))
result.write("mean_error= %s\n\nsda_error= %s\n\nknn_error= %s" % (str(mean_error), str(sda_error),str(knn_error)))
result.close()


"""
plt.plot(missing_percent,mean_error,'--bo',label='mean_row')
plt.plot(missing_percent,knn_error,'--go',label='knn' )
plt.plot(missing_percent,sda_error,'--ro',label='sda')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: breastCancer')
plt.legend(loc=4,prop={'size':9})
plt.show()




sda_error=  [3.8260220448509372, 8.3134063113508372, 13.301469426012245, 18.534558816726975, 25.507519918750134, 34.345746144852747, 43.907880553354687, 65.711575619830185, 88.206401253949323]
knn_error=  [4.133471439051136, 8.71459097334583, 13.255826924199463, 18.118129399653526, 23.603173355960664, 30.873652227755972, 41.707055851469555, 63.523198813225022, 82.904008821343837]
mean_error=  [9.7225906322632252, 19.710458600496462, 29.408958287533988, 39.221673034352115, 48.674593199351477, 58.789162736627119, 68.300097992049515, 78.483836561944045, 88.256928348874325]

                     method = 'adam',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 10,
                      hidden_size = [600,200,100,60,40,21],  #19 was good for >80%corrup
                      corruption_da = [0.2,0.1,0.1,.1,.2,.1],
                      dA_initiall = True ,
                      error_known = True ,
                      activ_fun = T.tanh)




behhhhhhtariiiiiiiin

sda_error=  [3.8037614867461857, 7.8602512425349325, 12.516868974637243, 17.8771785506013, 25.107349726799882, 34.43905740485728, 45.409295524454201, 63.063975343204433, 87.790687521439]
knn_error=  [4.2961102325030769, 8.7259405347851438, 13.233383029598356, 18.199898697545482, 23.520422607234046, 30.964698101688594, 42.496468044341697, 63.474155393451007, 82.996476076]
mean_error=  [9.8257947820992175, 19.852977538494812, 29.286181719811889, 39.264182885314241, 48.835079455949547, 58.858003643711626, 68.75078881886111, 78.322282592449042, 88.277710211]


sda2_error=np.array(sda2_error)
sda_error=np.array(sda_error)
knn_error=np.array(knn_error)
mean_error=np.array(mean_error)
mean_error=mean_error.reshape(30,9)
sda_error=sda_error.reshape(30,9)
knn_error=knn_error.reshape(30,9)
sda2_error=sda2_error.reshape(30,9)
missing_percent=np.linspace(0.1,0.9,9)

mean_con=np.max(mean_error,axis=0)-np.min(mean_error,axis=0)
knn_con=np.max(knn_error,axis=0)-np.min(knn_error,axis=0)
sda_con=np.max(sda_error,axis=0)-np.min(sda_error,axis=0)
 
mean_con=np.std(mean_error)
knn_con=np.std(knn_error)
sda_con=np.std(sda_error)
sda2_con=np.std(sda2_error)


plt.figure()
plt.errorbar(missing_percent,np.mean(mean_error,axis=0),yerr=mean_con,fmt='--o',label='mean_row')
plt.errorbar(missing_percent,np.mean(knn_error,axis=0),yerr=knn_con,fmt='--o',label='knn' )
plt.errorbar(missing_percent,np.mean(sda_error,axis=0),yerr=sda_con,fmt='--o',label='sda')
plt.xlabel('corruption percentage')
plt.ylabel('Mean absolute error')
plt.title('dataset: breast cancer')
plt.legend(loc=4,prop={'size':9})
plt.show()





"""

