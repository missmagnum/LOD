from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import lasagne

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams

from update import Update


class dA(object):    

    def __init__( self, numpy_rng, theano_rng=None, input=None, n_visible=None, n_hidden=None, W=None, bhid=None,
                  bvis=None , method=None , problem = None, activation = None, regu_l1=0, regu_l2=0):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.method=method
        self.activ = activation

        self.srng = RandomStreams()
        self.regu_l1 = regu_l1
        self.regu_l2 = regu_l2
        if not W:            
            W = theano.shared(self.floatX(np.random.randn(*shape) * 0.1), name='W', borrow=True)
          
        if not bvis:
            bvis = theano.shared(self.floatX( np.zeros((n_visible,))), name='b_prime', borrow=True)
      
        if not bhid:
            bhid = theano.shared(self.floatX( np.zeros((n_hidden,))), name='b', borrow=True)
    


        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.problem = problem
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        self.params = [self.W, self.b, self.b_prime]
        self.main_params=[self.W,self.b]
        

    def get_corrupted_input(self, X, p=0.):

        retain_prob = 1 - p
        X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X


    def get_hidden_values(self, input):
        if self.activ is None:
            return T.dot(input, self.W) + self.b
        else:
            return self.activ(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        if self.activ is None:
            return T.dot(hidden, self.W_prime) + self.b_prime
        else:
            return self.activ(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)        
        L =T.sum(T.sqr(self.x-z))
        
        ################## add l2 regularization #################
        lamb1 = self.regu_l1  #0.001 #1e-5
        lamb2 = self.regu_l2   #1e-5
        #L2 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l2)
        #L1 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l1) * lambda1

        regu_l2 = T.sum(T.sqr(self.W)+T.sqr(self.b))
        regu_l1 = abs(T.sum(self.W)+T.sum(self.b))
        
        cost = L  + lamb2 * regu_l2 + lamb1 * regu_l1
        
        updates = Update(method = self.method,
                         cost = cost,
                         params = self.params,
                         learning_rate= learning_rate)

        return (cost, updates)

    def get_prediction(self):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        return z

    def get_latent_representation(self):
        return self.get_hidden_values(self.x)


    def floatX(self,X):
        return np.asarray(X, dtype=theano.config.floatX)

