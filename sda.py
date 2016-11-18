import numpy as np
import numpy
import theano
import lasagne

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from perceptron import perceptron
from update import Update


class Sda(object):


    def __init__( self, numpy_rng=None, theano_rng=None,n_inputs=None,
                  hidden_layers_sizes = None,
                  corruption_levels=[0.1, 0.1],
                  dA_initiall=True,
                  error_known=True,
                  method=None,
                  problem = None):         

        activation_function =T.tanh  #T.arctan  #T.tanh 
        
        self.n_layers = len(hidden_layers_sizes)
        self.n_inputs=n_inputs
        self.hidden_layers_sizes=hidden_layers_sizes
        self.error_known = error_known
        self.method=method
        self.problem = problem
        
        assert self.n_layers >= 2

        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(123)
               
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')       
        self.mask = T.matrix('mask')


        ### encoder_layers ####
        
        self.encoder_layers = []
        self.encoder_params = []
        self.dA_layers=[]
        for i in range(self.n_layers):
            
            if i == 0:
                input_size = self.n_inputs
                corruption=True
                
            else:
                input_size = self. hidden_layers_sizes[i-1]
                corruption=False
          
            if i == 0:
                layer_input = self.x
            else:
                layer_input=self.encoder_layers[-1].output
                
            act_func=activation_function
                
            self.encoder_layer=perceptron(rng = numpy_rng,
                                          theano_rng=theano_rng,
                                          input = layer_input,
                                          n_in = input_size,
                                          n_out = self.hidden_layers_sizes[i],
                                          activation = act_func,
                                          first_layer_corrup=corruption)

            if dA_initiall :
                dA_layer = dA(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=self.encoder_layer.W,
                              bhid=self.encoder_layer.b,
                              method = self.method)
                
                self.dA_layers.append(dA_layer)
            
            self.encoder_layers.append(self.encoder_layer)
            self.encoder_params.extend(self.encoder_layer.params)

 


        ### decoder_layers ####

        self.decoder_layers = []
        self.decoder_params = []
        
        self.reverse_layers=self.encoder_layers[::-1]
        #self.reverse_da=self.dA_layers[::-1]
        
        decode_hidden_sizes=list(reversed(self.hidden_layers_sizes))

        for i,j in enumerate(decode_hidden_sizes):
            
            
            input_size=j
            if i == 0:
                layer_input=self.reverse_layers[i].output
            else:
                layer_input=self.decoder_layers[-1].output
            
            if i==len(decode_hidden_sizes)-1:
                n_out= self.n_inputs
            else:
                n_out=decode_hidden_sizes[i+1]


            if i==len(decode_hidden_sizes)-1:
                if self.problem == 'regression':
                    act_func = None 
                else:
                    act_func = T.nnet.sigmoid
            else:
            act_func=activation_function
            
            self.decoder_layer=perceptron(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=n_out,
                                        W= self.reverse_layers[i].W,
                                        b= None,
                                        activation=act_func,
                                        decoder=True
            )

            
            self.decoder_layers.append(self.decoder_layer)
            
            self.decoder_params.append(self.decoder_layer.b)
            
            
        self.network_layers=  self.encoder_layers + self.decoder_layers
        self.params = self.encoder_params + self.decoder_params
        print(self.params)
        

 

    def pretraining_functions(self, train_set_x, batch_size):

       
        index = T.lscalar('index') 
        corruption_level = T.scalar('corruption')  
        learning_rate = T.scalar('lr')  
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.1),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            
            pretrain_fns.append(fn)

        return pretrain_fns



    

    def finetune_cost(self):     
                                
      
        ## cost over known data
        x = self.x * self.mask
        z = self.decoder_layer.output* self.mask

        if self.problem == 'regression':
            print('regression')
            cost = T.mean(T.sum((x - z )**2 , axis=1))
        else:
            cost = T.mean(T.sum((x - z )**2 , axis=1))#T.mean(T.sum( x* T.log(z) + (1-x)*T.log(1-z) ,axis=1))
        
        ## add regularization
        regularizationl2=lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l2)
        regu_l2 = T.sum([ T.sum(layer.W**2) for layer in self.network_layers] )
        regu_l1 = T.sum([ T.sum(abs(layer.W)) for layer in self.network_layers] )
        lambda1 = 1e-4

        cost_regu=cost + lambda1 * regu_l2

        return cost_regu ,cost

        
    
    def build_finetune_functions(self,dataset, method, train_set_x, valid_set_x, test_set_x,
                                 train_mask, test_mask, valid_mask,
                                 batch_size, learning_rate):
        


        
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index') 

        
        finetune_cost, validation_test=self.finetune_cost()
        
        updates = Update(method = method,
                         cost = finetune_cost,
                         params = self.params,
                         learning_rate= learning_rate)

        train_fn = theano.function(
            inputs=[index],
            outputs = finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.mask: train_mask[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            outputs = validation_test,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.mask: test_mask[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            outputs = validation_test,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.mask: valid_mask[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        
 
            
        #self.output=lasagne.layers.get_output(self.network_layers,inputs=dataset)

        
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score



