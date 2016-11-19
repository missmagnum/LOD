import numpy

import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

class perceptron(object):
    def __init__(self, rng=None,theano_rng=None, input=None, n_in=None , n_out=None, W=None, b=None,
                 activation=None, decoder=False, first_layer_corrup=False):
        self.input = input
        

        if not rng:
            rng = numpy.random.RandomState(123)
        if not theano_rng:
            theano_rng = RandomStreams(rng.randint(2 ** 30))       


        if W is None:
            
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)


        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

   
       
        self.W = W
        self.b = b
       

        if first_layer_corrup:
            corruption_level = 0.1        
            input = theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
       

        if decoder:
            lin_output=T.dot(input, self.W.T) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]

       
