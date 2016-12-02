import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np



class perceptron():
    def __init__(self,input=None, n_in=None , n_out=None, W=None,
               b=None, activation=None, decoder=False,
               first_layer_corrup=False, drop = None ):
     
        self.srng = RandomStreams()
        shape = (n_in, n_out)
       
        if not W:            
            W = theano.shared(self.floatX(np.random.randn(*shape) * 0.1), name='W', borrow=True)
          
        if not b:
            b = theano.shared(self.floatX( np.zeros((n_out,))), name='b', borrow=True)
      
        
        if drop is not None:
            input = self.dropout (input,drop)
            
        self.W = W
        self.b = b
        
        if decoder:
            lin_output=T.dot(input, self.W.T) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]



    def floatX(self,X):
        return np.asarray(X, dtype=theano.config.floatX)

    def dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X



       
