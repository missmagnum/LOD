import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from perceptron import perceptron
from sda import Sda

class Gather_sda(object):

    def __init__(self,
                 dataset=None,
                 portion_data=None,
                 problem = 'regression' ,
                 available_mask = None,
                 method = None,
                 pretraining_epochs = 100,
                 pretrain_lr = 0.005,
                 training_epochs = 100,
                 finetune_lr = 0.0005,
                 batch_size = 50,
                 hidden_size = [100,20,2],
                 corruption_da = [0.1, 0.1, 0.1],
                 dA_initiall = True,
                 error_known = True ):
        self.problem = problem
        self.method = method
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.training_epochs = training_epochs
        self.finetune_lr = finetune_lr     
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.corruption_da = corruption_da
        self.dA_initiall = dA_initiall
        self.error_known = error_known
       
        def load_data(X):
            try:
                matrix = X.as_matrix()
            except AttributeError:
                matrix = X
                shared_x = theano.shared(numpy.asarray(matrix, dtype=theano.config.floatX), borrow=True)
            return shared_x

        self.train_set,self.valid_set,self.test_set = [load_data(i) for i in portion_data]
        
        if error_known:
            self.train_mask,self.valid_mask,self.test_mask = [load_data(i) for i in available_mask]

        else:
            self.train_mask,self.valid_mask,self.test_mask = [load_data(numpy.ones_like(i)) for i in available_mask]
        
       
        self.dataset=load_data(dataset)
        self.n_visible = dataset.shape[1]
        self.n_train_batches = self.train_set.get_value(borrow=True).shape[0] // batch_size        
        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
        print('input size', self.n_visible)
        
    def pretraining(self):

        self.sda=Sda(
            numpy_rng = self.numpy_rng,
            theano_rng= self.theano_rng,
            n_inputs = self.n_visible,
            hidden_layers_sizes = self.hidden_size,
            corruption_levels = self.corruption_da,
            dA_initiall = self.dA_initiall,
            error_known = self.error_known,
            method=self.method,
            problem = self.problem)
                 
   
        pretraining_fns = self.sda.pretraining_functions(train_set_x = self.train_set,
                                                         batch_size = self.batch_size)

        print('... pre-training the model')
        corruption_levels = self.corruption_da

        for i in range(self.sda.n_layers):
        
            for epoch in range(self.pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(self.n_train_batches):
                    c.append(pretraining_fns[i](index = batch_index,
                                                corruption = corruption_levels[i],
                                                lr = self.pretrain_lr))
                #print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

   
    
    def finetuning(self):

        if self.dA_initiall:
            self.pretraining()
        else:
            self.sda=Sda(
            numpy_rng = self.numpy_rng,
            theano_rng= self.theano_rng,
            n_inputs = self.n_visible,
            hidden_layers_sizes = self.hidden_size,
            corruption_levels = self.corruption_da,
            dA_initiall = self.dA_initiall,
            error_known = self.error_known,
            method=self.method,
            problem = self.problem)

        self.gather_out=theano.function(
            [],
            outputs=self.sda.decoder_layer.output,
            givens={
                self.sda.x : self.dataset}
        )

        

        print('... getting the finetuning functions')
        train_fn, validate_model, test_model =self.sda.build_finetune_functions(
            dataset = self.dataset,
            method = self.method,
            train_set_x = self.train_set,
            valid_set_x = self.valid_set,
            test_set_x = self.test_set,
            train_mask = self.train_mask,
            test_mask = self.test_mask,
            valid_mask = self.valid_mask,
            batch_size = self.batch_size,
            learning_rate = self.finetune_lr)

      

        patience = 10 * self.n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                            # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
        validation_frequency = min( self.n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
     
        done_looping = False
        epoch = 0
        ### hold out cross validation
        while (epoch < self.training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):
                
                
                minibatch_avg_cost = train_fn(minibatch_index)
                
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f ' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        
                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        #print(('     epoch %i, minibatch %i/%i, validation error of '
                               'best model %f ') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               best_validation_loss))
                        #print('W',self.sda.decoder_layer.W.get_value()[-1,-1])
                        
                if patience <= iter:
                    done_looping = True
                    break
      
        
        print(
            (
                'Optimization complete with best validation score of %f , '
                'on iteration %i, '
                'with test performance %f '
            )
            % (best_validation_loss , best_iter + 1, test_score )
        )
        
        return self.sda

        
