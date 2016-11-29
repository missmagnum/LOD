import numpy
import lasagne

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams

from update import Update


class dA(object):    

    def __init__( self, numpy_rng, theano_rng=None, input=None, n_visible=None, n_hidden=None, W=None, bhid=None,
                  bvis=None , method=None , problem = None, activation = None):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.method=method
        self.activ = activation
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='b_prime',
                borrow=True
            )
            
        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
            
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
        

    def get_corrupted_input(self, input, corruption_level):
        
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

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
        lamb1 = 0.001 #1e-5
        lamb2 = 1e-4
        #L2 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l2)
        #L1 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l1) * lambda1

        regu_l2 = T.sum(T.sqr(self.W)+T.sqr(self.b))
        regu_l1 = abs(T.sum(self.W)+T.sum(self.b))
        
        cost = L   + lamb2 * regu_l2
        
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




