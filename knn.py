import numpy as np
import time
import copy

def knn(data, mask, k = 3, lamb=.7 ,norm=2 ): #for sine lamb=0.5
    data=copy.copy(data)
    t0=time.time()
    n = data.shape[0]
    missing_index=np.where(mask==0)
    miss_rows = list(set(missing_index[0])) #without replicate
    
    #distance
    D=np.zeros((len(miss_rows),n)) 
  
        
    for i,mis in enumerate(miss_rows):
        for j in range(n):
            m = mask[mis]*mask[j]
            if mis==j :
                D[i,j]= 1e5
            else:
                D[i,j] = (1./(np.sum(m)) * np.sum( (abs(data[mis]-data[j])**norm)*m))**(1/norm)
                #D[i,j] = np.sqrt(np.sum((data[mis]-data[j])**2 ))
    
    sort_d=np.argsort(D,axis=1)[:,:k]

    #gaussian kernel function 
    k=lambda u: 1./(2*np.pi) * np.exp(-.5*u**2)
    
    for i,mis in enumerate(miss_rows):
        indx = np.where(mask[mis]==0)[0]
        
        for j in indx:
            kernel = k(D[i,sort_d[i]]/lamb)
            
            weight = kernel / np.sum( kernel )
            data[mis,j] = np.sum( weight * data[sort_d[i],j] )
    

    #print(time.time()-t0)
    return data
            
    




if __name__ == "__main__":

    
    z=np.array([[9,1,2],[0,0,4],[5,9,0],[0,1,8],[7,3,7],[2,9,9]])
    a=np.array([[1, 1, 1], [0, 0, 1],[1, 1, 0], [0, 1, 1],[1, 1, 1], [1, 1, 1]])
    d=knn(z,a)



