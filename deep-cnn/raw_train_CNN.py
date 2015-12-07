"""
    This program implements a Deep Convolutional Neural Network
    on a small extract of the magnatagatune dataset for demonstration
    purposes for a masters final project.
    
    "DEEP CONVOLUTIONAL NETWORKS FOR AUDIO FEATURE LEARNING"
    
    Benjamin Stanley (Dec-2014).
    
    Queen Mary University of London
    
    ATTN: Theano must be installed to run this program.
    
    Please run this file on the computer server: exeter.eecs.qmul.ac.uk
    As this has the necessary Theano program files to run the code.
    The code must be run from the provided folder as it contains functions from the nntoolkit.
    
    nntools is available at: https://github.com/benanne/nntools
"""

from __future__ import print_function

import numpy
import pickle
import gzip
import time
import scipy.io

import nntools
import itertools
import theano
import theano.tensor as T
from nntools import init

###################################################################################################

"""
Training Parameters
"""

BATCH_SIZE = 10             # Number of example for each mini-batch
LEARNING_RATE = 0.01        # Value to adjust the weights of each parameter
MOMENTUM = 0.9              # Parameter that controls the trajectory of the learning rate
PATIENCE = 0.0001           # Difference in validation error between iterations before traing stops

###################################################################################################

def loadData(filename):

    """
    Loads a dataset from a specified filepath
    """
    
    # Load the dataset
    f = gzip.open(filename, 'rb')
    trainData, validData = pickle.load(f)
    f.close()
    
    def normalise_data(x, rmax, rmin):
        """
        A function that performs normalisation on the dataset
        so that example values fall between a minimum and maximum
        """
        N = numpy.zeros(numpy.shape(x))
        rows = len(x)
        ra = rmax
        rb = rmin
        
        for i in range(0,rows):
            a = numpy.min(x[i])
            b = numpy.max(x[i])
            N[i] = (((ra-rb) * (x[i]-a)) / (b-a)) + rb
        
        return N
    
    
    def shuffle_in_unison(a, b):
        """
        A function that shuffles the X,Y pairs in unison
        so to prevent the succession of similar songs
        """
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(a)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(b)
    
    
    def shared_dataset(data_xy, borrow=True):
        """ 
        A function that imports the data and returns
        input / ouput data sets for training and validation.
        The data is returned as Theano shared variables so that
        batches can be passed to the GPU.
        """
        
        data_x, data_y = data_xy
        
        # Normalise the data to fall in a specific range
        data_x = normalise_data(data_x, 0.995, -0.995)
        
        # Shuffle new datasets in union to avoid correlations
        shuffle_in_unison(data_x, data_y)
        
        # Determine the amount of examples per set
        num_examples = len(data_x)
        
        # Convert data to Theano shared variables
        data_x = numpy.reshape(data_x, (data_x.shape[0], 1, 48000))
        
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
                                 
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that).
        # However the target data is used as an index throughout
        # the program so they must be converted to integer values
        
        return shared_x, T.cast(shared_y, 'int32'), num_examples
    
    # Perform the dataset organisation
    validDataX, validDataY, num_examples_valid = shared_dataset(validData)
    trainDataX, trainDataY, num_examples_train = shared_dataset(trainData)
    
    # Construct a dictionary containing the dataset and
    # the shape information associated with it.
    
    return dict(
                X_train = trainDataX,
                y_train = trainDataY,
                X_valid = validDataX,
                y_valid = validDataY,
                num_examples_train = num_examples_train,
                num_examples_valid = num_examples_valid,
                input_width  = 48000,
                input_height = 1,
                output_dim = 50,
                )

###################################################################################################

def conv1d_sc(input, filters, image_shape=None, filter_shape=None, border_mode='valid', subsample=(1,)):
    """
    using Theano 'conv2d' function to perform a 1 dimensional convolution accross the time axis
    """
    if border_mode != 'valid':
        raise RuntimeError("Unsupported border_mode for conv1d_sc: %s" % border_mode)
    
    # Re-arrange the shape of the input by inserting a broacastable dimension
    # 'conv2d' accepts only 4D tensors as input
    if image_shape is None:
        image_shape_sc = None
    else:
        image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2]) # (b, c, i0) to (b, 1, c, i0)
    
    # Re-arrange the shape of the filters by inserting a broacastable dimension
    # 'conv2d' accepts only 4D tensors to describe filters
    if filter_shape is None:
        filter_shape_sc = None
    else:
        filter_shape_sc = (filter_shape[0], 1, filter_shape[1], filter_shape[2])
    
    # Arrange the input to match the input shape
    input_sc = input.dimshuffle(0, 'x', 1, 2)

    # We need to flip the channels dimension because it will be convolved over.
    filters_sc = filters.dimshuffle(0, 'x', 1, 2)[:, :, ::-1, :]

    # Perform the convolution
    conved = T.nnet.conv2d(input_sc, filters_sc, image_shape=image_shape_sc, filter_shape=filter_shape_sc, subsample=(1, subsample[0]))
    return conved[:, 0, :, :] # drop the unused dimension

###################################################################################################

def conv1d_unstrided(input, filters, image_shape, filter_shape, border_mode='valid', subsample=(1,), implementation=conv1d_sc):
    """
    perform a strided 1D convolution by reshaping input and filters so that the stride becomes 1.
    This function requires that the filter length is a multiple of the stride.
    It also truncates the input to have a length that is a multiple of the stride.
    """
    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]
    
    if filter_length % stride > 0:
        raise RuntimeError("Filter length (%d) is not a multiple of the stride (%d)" % (filter_length, stride))
    
    assert border_mode == 'valid'

    num_steps = filter_length // stride
    
    # input sizes need to be multiples of the strides, truncate to correct sizes.
    truncated_length = (input_length // stride) * stride
    input_truncated = input[:, :, :truncated_length]
    
    r_input_shape = (batch_size, num_input_channels, truncated_length // stride, stride)
    r_input = input_truncated.reshape(r_input_shape)
    
    # fold strides into the feature maps dimension (input)
    r_input_folded_shape = (batch_size, num_input_channels * stride, truncated_length // stride)
    r_input_folded = r_input.dimshuffle(0, 1, 3, 2).reshape(r_input_folded_shape)
    
    r_filter_shape = (num_filters, num_input_channels, num_steps, stride)
    r_filters_flipped = filters[:, :, ::-1].reshape(r_filter_shape)
    
    # fold strides into the feature maps dimension (filters)
    r_filter_folded_shape = (num_filters, num_input_channels * stride, num_steps)
    r_filters_flipped_folded = r_filters_flipped.dimshuffle(0, 1, 3, 2).reshape(r_filter_folded_shape)
    r_filters_folded = r_filters_flipped_folded[:, :, ::-1] # unflip
    
    return implementation(r_input_folded, r_filters_folded, r_input_folded_shape, r_filter_folded_shape, border_mode, subsample=(1,))

###################################################################################################


def buildModel(input_width, input_height, output_dim,
               batch_size = BATCH_SIZE):
    """
    Build the CNN model based on the specified hyperparameters & data shape.
    """
    
    # Define the input layer to the network
    inputLayer = nntools.layers.InputLayer(
        shape=(BATCH_SIZE, input_height, input_width),
        )
    
    # Perform a strided convolution of the input
    # with uniform weight initialisation, recitified
    # linear units. This forms 128 learnt filters with
    # a length and stride of 256 samples.
    # Biases initialised to 1.
    stridedConvolutionLayer = nntools.layers.Conv1DLayer(
        inputLayer,
        num_filters=128,
        filter_length=(256),
        stride=256,
        W=init.Uniform(),
        b=init.Constant(1.0),
        nonlinearity=nntools.nonlinearities.rectify,
        convolution=conv1d_unstrided
        )
    
    # Construct the first convolutional layer that
    # perfoms convolutions on the 128 feature maps
    # to extract higher level features with uniform
    # weight initialisation & recitified linear units.
    # Biases initialised to 1.
    convLayer1 = nntools.layers.Conv1DLayer(
        stridedConvolutionLayer,
        num_filters=32,
        filter_length=(8),
        stride=1,
        W=init.Uniform(),
        b=init.Constant(1.0),
        nonlinearity=nntools.nonlinearities.rectify,
        convolution=conv1d_sc
        )
    
    # Perform max pooling of feature maps accross the time dimension.
    poolLayer1 = nntools.layers.MaxPool1DLayer(convLayer1, ds=(1, 4))
    
    # Construct the second convolutional layer that
    # perfoms convolutions on the 32 feature maps from
    # the previous layer to extract higher level features
    # with uniform weight initialisation & recitified linear units.
    # Biases initialised to 1.
    convLayer2 = nntools.layers.Conv1DLayer(
        poolLayer1,
        num_filters=32,
        filter_length=(8),
        stride=1,
        W=init.Uniform(),
        b=init.Uniform(1.0),
        nonlinearity=nntools.nonlinearities.rectify,
        convolution=conv1d_sc
        )
    
    # Perform max pooling of feature maps accross the time dimension.
    poolLayer2 = nntools.layers.MaxPool1DLayer(convLayer2, ds=(1, 4))
    
    # Build the first dense layer to classify the extracted
    # features. This layer is initialised with wieghts sampled from
    # a uniform distribution. Biases initialsed to 0.1
    hiddenLayer1 = nntools.layers.DenseLayer(
        poolLayer2,
        num_units=100,
        nonlinearity=nntools.nonlinearities.rectify,
        W=init.Uniform(),
        b=init.Constant(0.1),
        )
    
    # Ouput layer with a sigmoid output determines the scores
    # for each tag. This layer is initialised with wieghts sampled from
    # a uniform distribution. Biases initialsed to 0.1
    outputLayer = nntools.layers.DenseLayer(
        hiddenLayer1,
        num_units=output_dim,
        nonlinearity=nntools.nonlinearities.sigmoid,
        W=init.Uniform(),
        b=init.Constant(0.1),
        )
                                           
    return outputLayer

###################################################################################################

def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    
    """
    Create the functions used for training. Here 
    the gradients are calculated accross the model 
    for each batch and the updates are passed to 
    the weights.
    """
    # Take a slice of the data and corresponding targets
    #to pass to the GPU.
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.imatrix('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)
    
    # Compute the basis loss term for use with each dataset.
    def loss(output):
        return T.nnet.binary_crossentropy(output, y_batch).mean()
    
    # Compute the training loss
    loss_train = loss(output_layer.get_output(X_batch))
    loss_eval = loss(output_layer.get_output(X_batch, deterministic=True))
    
    # Generate the predictions
    pred_out = output_layer.get_output(X_batch, deterministic=True)
    
    # Calucluate the accuracy of the predictions agains the model
    # by using the sum of squares to calculated the euchlidean distance
    accuracy = T.sqrt(T.sum(T.sqr(pred_out - y_batch)))
    
    # Retreive every parameters at each layer of the network
    all_params = nntools.layers.get_all_params(output_layer)
    
    # Compute the updates based on the loss, current parameter,
    # learning rate and momentum
    updates = nntools.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum, weight_decay= 0.01)
                                                
    # Train the model iteratively using mini-batch
    # gradient descent.
    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
            },
        )
        
    # Compute the validation loss at regular intervals
    # to avoid overfitting.
    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            },

        )
    
    # Output functions for use with training
    return dict(
        train=iter_train,
        valid=iter_valid,
        )

###################################################################################################

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
   
    
    """
    This functions implements the mini-batch gradient descent
    algorithm for each batch in the dataset using the functions
    created in 'create_iter_funcs'.
    """
    
    # Determine the number of batches for the trainig and validation
    # sets.
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)
        
        # Give the average training loss for the epoch
        avg_train_loss = numpy.mean(batch_train_losses)
        
        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)
        
        # Give the average validation loss for the epoch
        avg_valid_loss = numpy.mean(batch_valid_losses)
        avg_valid_accuracy = numpy.mean(batch_valid_accuracies)
        
        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            'training_losses': batch_train_losses
            }

###################################################################################################


def main():
    
    start_time = time.clock()
    
    ###########
    #Load data#
    ###########
    print("loading data...")
    
    dataset = loadData('train_valid_set.pkl.gz')
    
    #############
    #Build Model#            
    #############
    print ("building the model...")
    
    output_layer = buildModel(
        input_width=dataset['input_width'],
        input_height=dataset['input_height'],
        output_dim=dataset['output_dim'],
        )
                              
    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor3,
        )
                              
    #############
    #Train Model#
    #############
    print("starting training...")
    prev_valid_loss = 1
    loss_characteristics = []
                              
    for epoch in train(iter_funcs, dataset):
        print("Epoch %d" % (epoch['number']))
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("             RMSE:\t\t%.2f " %
              (epoch['valid_accuracy']))
         
        # Generate the loss characteristic for analysis
        loss_characteristics.append(epoch['training_losses'])
                                                      
        
        # Perform early stopping by breaking the loop
        # when the validation loss stops decresing or
        # only decreses marginally (determined by patience).
        
        curr_valid_loss = (epoch['valid_loss'])
                                                          
        if abs(curr_valid_loss - prev_valid_loss) < PATIENCE:
            break
                                                                  
        elif curr_valid_loss > prev_valid_loss:
            break
                                                                          
        prev_valid_loss = curr_valid_loss
                                                                      
    end_time = time.clock()
    print('Optimization complete.')
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    
    # Save losses to show learning characteristics
    scipy.io.savemat('predictions/model_losses.mat', mdict={'loss_characteristics': loss_characteristics})
    
    # Save model parameters to file
    # Save parameter values including biases for each layer
    param_values = []
    all_ps = nntools.layers.get_all_params(output_layer)
    
    for p in range(len(all_ps)):
        param = all_ps[p].get_value()
        param_values.append(param)
    
    save_params=open('model/model_param_values.pkl', 'wb')
    pickle.dump(param_values, save_params)
    save_params.close()

if __name__ == '__main__':
    main()
