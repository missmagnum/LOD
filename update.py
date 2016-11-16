import numpy as np
import numpy
import theano
import lasagne
import theano.tensor as T


def Update(method=None , cost= None , params= None, learning_rate= None):        
    if method == None :
        gparams = T.grad(cost, params)
            
        
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(params, gparams)
        ]

            
    elif method == 'nes_mom' :
            
            
        updates = lasagne.updates.nesterov_momentum(cost,
                                                    params,
                                                    learning_rate = learning_rate)

            
            
    elif method == 'adadelta' :
        updates=lasagne.updates.adadelta(cost,
                                         params,
                                         learning_rate = learning_rate,
                                         rho = 0.95,
                                         epsilon = 1e-6)
    elif method=='adam':
        updates=lasagne.updates.adam(cost,
                                     params,
                                     learning_rate=learning_rate,
                                     beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-08)

    elif method== 'rmsprop':
        """
        learning_rate=learning_rate
        rho=0.9
        epsilon=1e-6
        one = T.constant(1)
            
        gparams = T.grad(cost, params)
            
        for param, grad in zip(params, gparams):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates=[(accu , accu_new) ,(param , param - (learning_rate * grad /
                                          T.sqrt(accu_new + epsilon)))]

        """
        updates=lasagne.updates.rmsprop(cost, params, learning_rate=learning_rate, rho=0.9, epsilon=1e-06)
        
            
    else:
        raise ValueError(" check the given UPDATES ")

    return updates
