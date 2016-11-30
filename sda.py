import numpy as np
import theano
import lasagne

import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
                  problem = None,
                  activ_fun = None,
                  drop = None):         

        self.activ_fun = activ_fun  #T.arctan  #T.tanh 
        
        self.n_layers = len(hidden_layers_sizes)
        self.n_inputs=n_inputs
        self.hidden_layers_sizes=hidden_layers_sizes
        self.error_known = error_known
        self.method=method
        self.problem = problem
        self.drop = drop
        
        #assert self.n_layers >= 2

 
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
                
            
                
            self.encoder_layer=perceptron(input = layer_input,
                                          n_in = input_size,
                                          n_out = self.hidden_layers_sizes[i],
                                          activation = activ_fun,
                                          first_layer_corrup=corruption,
                                          drop = self.drop[i])

            if dA_initiall :
                dA_layer = dA(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=self.encoder_layer.W,
                              bhid=self.encoder_layer.b,
                              method = self.method,
                              activation=activ_fun)
                
                self.dA_layers.append(dA_layer)
            
            self.encoder_layers.append(self.encoder_layer)
            self.encoder_params.extend(self.encoder_layer.params)

 


        ### decoder_layers ####

        self.decoder_layers = []
        self.decoder_params = []
        self.drop.reverse()
        
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
                act_func=activ_fun
            
            self.decoder_layer=perceptron(input=layer_input,
                                          n_in=input_size,
                                          n_out=n_out,
                                          W= self.reverse_layers[i].W,
                                          b= None,
                                          activation=act_func,
                                          decoder=True,
                                          drop = None)#self.drop[i])
    

            
            self.decoder_layers.append(self.decoder_layer)
            
            self.decoder_params.append(self.decoder_layer.b)
            
            
        self.network_layers=  self.encoder_layers + self.decoder_layers
        self.params = self.encoder_params + self.decoder_params
        #print(self.params)
        

 

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
            #print('regression')
            #cost = T.sum(T.sum((x - z )**2 , axis=1))
            cost = T.sum(T.sqr(x-z)) 
        else:
            T.mean(T.sum( x* T.log(z) + (1-x)*T.log(1-z) ,axis=1))
        
        ################### add regularization ###################

        lamb1 = 1e-8
        lamb2 = 1e-5
        #L2 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l2)
        #L1 = lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l1)
        
        regu_l2 = T.sum([T.sum(T.sqr(layer.W)) for layer in self.network_layers] )
        regu_l1 = T.sum([ np.abs(T.sum(layer.W)) for layer in self.network_layers] ) 

        cost_regu=cost   + lamb2 * regu_l2 # + lamb1 * regu_l1

        return cost_regu ,cost

        
    
    def build_finetune_functions(self,dataset, method, train_set_x, valid_set_x,
                                 train_mask, valid_mask,
                                 batch_size, learning_rate):
        


        
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
 
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

 
        valid_score_i = theano.function(
            [index],
            outputs = finetune_cost,
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

        
   

        return train_fn, valid_score



