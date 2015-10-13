import pickle

import numpy as np
import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import Pool2DDNNLayer as Pool2DLayer

def load_vgg(params_filename):
    cnn = {}
    X = theano.shared(np.zeros((1,3,1,1), theano.config.floatX), name="X")
    cnn["in"] = InputLayer(shape=(None, 3, None, None), input_var = X)
    cnn["conv1_1"] = Conv2DLayer(cnn["in"], num_filters=64, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv1_2"] = Conv2DLayer(cnn["conv1_1"], num_filters=64, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool1"] = Pool2DLayer(cnn["conv1_2"], pool_size=(2,2), stride=2, mode="average_inc_pad")

    cnn["conv2_1"] = Conv2DLayer(cnn["pool1"], num_filters=128, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv2_2"] = Conv2DLayer(cnn["conv2_1"], num_filters=128, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool2"] = Pool2DLayer(cnn["conv2_2"], pool_size=(2,2), stride=2, mode="average_inc_pad")

    cnn["conv3_1"] = Conv2DLayer(cnn["pool2"], num_filters=256, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv3_2"] = Conv2DLayer(cnn["conv3_1"], num_filters=256, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv3_3"] = Conv2DLayer(cnn["conv3_2"], num_filters=256, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool3"] = Pool2DLayer(cnn["conv3_3"], pool_size=(2,2), stride=2, mode="average_inc_pad")

    cnn["conv4_1"] = Conv2DLayer(cnn["pool3"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv4_2"] = Conv2DLayer(cnn["conv4_1"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv4_3"] = Conv2DLayer(cnn["conv4_2"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool4"] = Pool2DLayer(cnn["conv4_3"], pool_size=(2,2), stride=2, mode="average_inc_pad")

    cnn["conv5_1"] = Conv2DLayer(cnn["pool4"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv5_2"] = Conv2DLayer(cnn["conv5_1"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv5_3"] = Conv2DLayer(cnn["conv5_2"], num_filters=512, pad=1,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool5"] = Pool2DLayer(cnn["conv5_3"], pool_size=(2,2), stride=2, mode="average_inc_pad")

    # DenseLayers break when connectd to a net with variable input shapes on
    # the last two dimensions. Luckily we don't need them for the style
    # transfer.

    #cnn["fc6"] = DenseLayer(cnn["pool5"], 4096, nonlinearity=lasagne.nonlinearities.rectify)
    #cnn["fc6_drop"] = lasagne.layers.DropoutLayer(cnn["fc6"], p=0.5)
    #cnn["fc7"] = DenseLayer(cnn["fc6_drop"], 4096, nonlinearity=lasagne.nonlinearities.rectify)
    #cnn["fc7_drop"] = lasagne.layers.DropoutLayer(cnn["fc7"], p=0.5)

    #cnn["prob"] = DenseLayer(cnn["fc7_drop"], 1000, nonlinearity=lasagne.nonlinearities.softmax)

    params = pickle.load(open(params_filename, "rb"), encoding="bytes")
    for layer in cnn:
        if layer not in params:
            continue

        if params[layer][0].ndim == 4:
            cnn[layer].W.set_value(params[layer][0].astype(theano.config.floatX))
            cnn[layer].b.set_value(params[layer][1].astype(theano.config.floatX))
        else:
            assert(params[layer][0].ndim == 2)
            cnn[layer].W.set_value(params[layer][0].T.astype(theano.config.floatX))
            cnn[layer].b.set_value(params[layer][1].astype(theano.config.floatX))

    #cnn["prob"].W.set_value(params["fc8"][0].T.astype(theano.config.floatX))
    #cnn["prob"].b.set_value(params["fc8"][1].astype(theano.config.floatX))

    return cnn
