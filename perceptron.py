import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

"""
class perceptron(object):
    def __init__(self, rng=None,theano_rng=None,
                 input=None, n_in=None , n_out=None,
                 W=None, b=None,
                 activation=None,
                 decoder=False,
                 first_layer_corrup=False,
                 drop = None):

        self.srng = RandomStreams()
        self.input = input       
        if not rng:
            rng = np.random.RandomState(123)
        if not theano_rng:
            theano_rng = RandomStreams(rng.randint(2 ** 30))       


        if W is None:
            
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)


        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

   
        if first_layer_corrup:
            corruption_level = 0.1        
            input = theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
       
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

    def dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X

       

"""
class perceptron():
    def __init__(self,input=None, n_in=None , n_out=None, W=None,
               b=None, activation=None, decoder=False,
               first_layer_corrup=False, drop = None ):
     
        self.srng = RandomStreams()
        shape = (n_in, n_out)
       
        if W is None:            
            W = theano.shared(self.floatX(np.random.randn(*shape) * 0.1), name='W', borrow=True)
          
        if b is None:
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



       
