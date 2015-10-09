import sys
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.transform

import theano
import theano.tensor as tt

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import Pool2DDNNLayer as Pool2DLayer

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from data import load_imgs, category_map

n_classes = len(category_map)

def build_cnn(input_var, input_shape):
    cnn = {}
    cnn["in"] = InputLayer(shape=(None, 3, input_shape[0], input_shape[1]), input_var=input_var)

    cnn["conv1"] = Conv2DLayer(cnn["in"], num_filters=64, stride=2,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv2"] = Conv2DLayer(cnn["conv1"], num_filters=64, stride=2,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool2"] = Pool2DLayer(cnn["conv2"], pool_size=(2,2))

    cnn["conv3"] = Conv2DLayer(cnn["pool2"], num_filters=128,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv4"] = Conv2DLayer(cnn["conv3"], num_filters=128,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool4"] = Pool2DLayer(cnn["conv4"], pool_size=(2,2))

    cnn["conv5"] = Conv2DLayer(cnn["pool4"], num_filters=256,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["conv6"] = Conv2DLayer(cnn["conv5"], num_filters=256,
            filter_size=(3,3), nonlinearity = lasagne.nonlinearities.rectify)
    cnn["pool6"] = Pool2DLayer(cnn["conv6"], pool_size=(2,2))

    cnn["fc7"] = DenseLayer(cnn["pool6"], 4096, nonlinearity=lasagne.nonlinearities.rectify)
    cnn["fc7_drop"] = lasagne.layers.DropoutLayer(cnn["fc7"], p=0.5)

    cnn["fc8"] = DenseLayer(cnn["fc7_drop"], 4096, nonlinearity=lasagne.nonlinearities.rectify)
    cnn["fc8_drop"] = lasagne.layers.DropoutLayer(cnn["fc8"], p=0.5)

    cnn["out"] = DenseLayer(cnn["fc8_drop"], n_classes, nonlinearity=lasagne.nonlinearities.softmax)

    cnn["fc_mid"] = DenseLayer(cnn["pool4"], 4096, nonlinearity=lasagne.nonlinearities.rectify)
    cnn["out_mid"] = DenseLayer(cnn["fc_mid"], n_classes, nonlinearity=lasagne.nonlinearities.softmax)

    return cnn

def warp_fast(img, T, out_shape=None):
    if out_shape is None:
        out_shape = img.shape

    return skimage.transform._warps_cy._warp_fast(img, T.params, output_shape=out_shape, mode="constant", order=1)

def approx_rescale_transform(scale, in_shape, out_shape=None):
    if out_shape is None:
        out_shape = in_shape

    r, c = in_shape
    tr, tc = out_shape
    ds = skimage.transform.AffineTransform(scale=(scale, scale))

    sx = c / (2.0 * scale) - tc / 2.0
    sy = r / (2.0 * scale) - tr / 2.0
    shift_ds = skimage.transform.SimilarityTransform(translation=(sx, sy))

    return shift_ds + ds

def resize(img, scale):
    scale = 1.0 / scale

    if img.ndim == 3:
        r = warp_fast(img[:,:,0], approx_rescale_transform(scale, img.shape[:2]))
        g = warp_fast(img[:,:,1], approx_rescale_transform(scale, img.shape[:2]))
        b = warp_fast(img[:,:,2], approx_rescale_transform(scale, img.shape[:2]))
        return np.dstack([r,g,b])
    else:
        return warp_fast(img, approx_rescale_transform(scale, img.shape))

def perturb(img, scale, flip_x, flip_y):
    img = resize(img, scale)
    if flip_x:
        img = img[:, ::-1]
    if flip_y:
        img = img[::-1]
    return img

def batch_iter(X, y, batch_size=32, do_perturb=False):
    n_batches = X.shape[0] // batch_size
    batch_shape = (batch_size, X.shape[3], X.shape[1], X.shape[2])

    for i in range(n_batches):
        if do_perturb:
            batch = np.zeros(batch_shape, dtype=np.float32)

            for j in range(batch_size):
                scale = 1.0 + np.random.rand()
                flip_x = np.random.rand() < 0.5
                flip_y = np.random.rand() < 0.5

                batch[j] = perturb(X[i*batch_size + j], scale, flip_x, flip_y).swapaxes(2,1).swapaxes(1,0)
            yield batch, y[i*batch_size:(i+1)*batch_size]
        else:
            yield X[i*batch_size:(i+1)*batch_size].swapaxes(3,2).swapaxes(2,1), y[i*batch_size:(i+1)*batch_size]

    rem = X.shape[0] % batch_size
    if rem != 0:
        if do_perturb:
            batch = np.zeros((rem, batch_shape[1], batch_shape[2], batch_shape[3]), dtype=np.float32)
            for j in range(rem):
                scale = 1.0 + np.random.rand()
                flip_x = np.random.rand() < 0.5
                flip_y = np.random.rand() < 0.5

                batch[j] = perturb(X[n_batches*batch_size + j], scale, flip_x, flip_y).swapaxes(2,1).swapaxes(1,0)
            yield batch, y[n_batches * batch_size:]
        else:
            yield X[n_batches*batch_size:].swapaxes(3,2).swapaxes(2,1), y[n_batches*batch_size:]

def make_loss_updates(net, X, y, learning_rate, momentum, only_layers=None):
    out_layer = net["out"]
    out = lasagne.layers.get_output(out_layer)
    out_det = lasagne.layers.get_output(out_layer, deterministic=True)
    wd = 1e-3

    mid_out = lasagne.layers.get_output(net["out_mid"])

    #train_loss = lasagne.objectives.squared_error(out[:,0], y)
    train_loss = 1.0 * lasagne.objectives.categorical_crossentropy(out, y)# + 0.5 * lasagne.objectives.categorical_crossentropy(mid_out, y)
    #train_loss = abs(out[:,0] - y)
    #train_loss = 0.5*train_loss.mean() + 0.5*abs(mid_out[:,0] - y).mean()
    train_loss = train_loss.mean()
    train_loss += wd*lasagne.regularization.regularize_network_params(out_layer, lasagne.regularization.l2)

    #val_loss = lasagne.objectives.squared_error(out_det[:,0], y)
    val_loss = lasagne.objectives.categorical_crossentropy(out_det, y)
    #val_loss = abs(out_det[:,0] - y)
    val_loss = val_loss.mean()
    val_loss += wd*lasagne.regularization.regularize_network_params(out_layer, lasagne.regularization.l2)

    #val_acc = abs(out_det[:,0] - y).mean()
    #val_acc = tt.mean(tt.eq(tt.argmax(out_det, axis=1), y), dtype=theano.config.floatX)
    val_acc = tt.argmax(out_det, axis=1)

    if only_layers is not None:
        params = []
        for layer in only_layers:
            params.extend(net[layer].get_params(trainable=True))
    else:
        params = lasagne.layers.get_all_params(out_layer, trainable=True)

    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate = learning_rate, momentum = momentum)
    train_fn = theano.function([X, y], train_loss, updates=updates)
    val_fn = theano.function([X, y], [val_loss, val_acc])

    return train_fn, val_fn

def train_net(net, X, y, data, batch_size, learning_rate, momentum, num_epochs,
        verbose=False, save_models=False, only_layers=None, do_perturb=False):
    X_train, X_val, y_train, y_val = data

    if verbose:
        print("Defining / compiling loss and updates...")
    train_fn, val_fn = make_loss_updates(net, X, y, learning_rate, momentum, only_layers)

    train_losses = []
    val_losses = []
    last_lr_drop = 0

    if verbose:
        print("SGD...")

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        val_acc = 0
        n_train_batches = 0
        n_val_batches = 0
        vp = []

        start = time.time()
        for batch in batch_iter(X_train, y_train, batch_size=batch_size, do_perturb=do_perturb):
            train_loss += train_fn(*batch)
            n_train_batches += 1
        train_time = time.time()-start

        start = time.time()
        for batch in batch_iter(X_val, y_val, batch_size=batch_size):
            l, a = val_fn(*batch)
            val_loss += l
            val_acc += np.mean(a == batch[1])
            vp.extend(list(a))
            n_val_batches += 1
        val_time = time.time()-start

        train_losses.append(train_loss / n_train_batches)
        val_losses.append(val_loss / n_val_batches)

        if verbose:
            print("==[ Epoch %d of %d ]==" % (epoch, num_epochs))
            print("Train loss:\t\t{:.6f}".format(train_loss / n_train_batches))
            print("Val loss:\t\t{:.6f}".format(val_loss / n_val_batches))
            print("Val acc:\t\t{:.2f}%".format(100.*val_acc / n_val_batches))
            print("Val preds:", np.bincount(vp, minlength=19))
            print("Time (train, val):\t\t{:.2f}, {:.2f}".format(train_time, val_time))
            print()

        if (epoch+1) % 10 == 0 and save_models:
            plt.plot(range(len(train_losses)), train_losses)
            plt.plot(range(len(val_losses)), val_losses)
            plt.savefig("models/lossplot_%d.png" % epoch)
            plt.close()

            if save_models:
                pickle.dump(lasagne.layers.get_all_param_values(net["out"]), open("models/params_%d.pkl" % epoch, "wb"), protocol=-1)

        if len(val_losses) >= 6 and np.mean(val_losses[-3:]) > np.mean(val_losses[-6:-3]) and (epoch - last_lr_drop) > 3:
            learning_rate.set_value((learning_rate.get_value() / 10.).astype(np.float32))
            last_lr_drop = epoch
            print("LR change:\t\t{:.3e}".format(learning_rate.get_value().item()))
            print()
            if (1e-5 - learning_rate.get_value().item()) > 1e-8:
                break

    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.savefig("lossplot.png")
    plt.close()

if __name__ == "__main__":
    num_epochs = 200
    learning_rate = theano.shared(np.float32(1e-2), "lr")
    momentum = 0.90
    batch_size = 128
    input_shape = (320, 240)

    if sys.argv[1].endswith(".pkl"):
        print("Loading data...")
        images, categories = pickle.load(open(sys.argv[1], "rb"))
    else:
        print("Processing images...")
        images, categories = load_imgs(sys.argv[1:])
        pickle.dump((images, categories), open("warped_imgs.pkl", "wb"))

    train_images = []; val_images = []
    train_categories = []; val_categories = []

    imgs, cats = np.array(images), np.array(categories)
    cc = np.array([c[0] for c in categories])
    for cat in np.unique(cats):
        ind = (cc == cat)
        ti, vi, tc, vc = train_test_split(imgs[ind], cats[ind], test_size = 0.30)
        train_images.extend(ti); val_images.extend(vi)
        train_categories.extend(tc); val_categories.extend(vc)

    train_images = np.vstack(train_images); val_images = np.vstack(val_images)
    train_categories = np.hstack(train_categories); val_categories = np.hstack(val_categories)
    train_categories = train_categories.astype(np.int32); val_categories = val_categories.astype(np.int32)

    train_images, train_categories = shuffle(train_images, train_categories)
    val_images, val_categories = shuffle(val_images, val_categories)

    print("N train:", train_images.shape[0])
    print("N val:", val_images.shape[0])

    mean_img = train_images.mean(axis=0).mean(axis=0).mean(axis=0)
    train_images -= mean_img[np.newaxis, np.newaxis, np.newaxis]
    val_images -= mean_img[np.newaxis, np.newaxis, np.newaxis]

    data = [train_images, val_images, train_categories, val_categories]

    print("Mean color:", mean_img)

    X = tt.tensor4("X")
    y = tt.ivector("y")

    print("Building model...")
    cnn = build_cnn(X, input_shape)

    print("Training...")
    train_net(cnn, X, y, data, batch_size, learning_rate, momentum, num_epochs,
            verbose=True,
            do_perturb=False,
            save_models=True)
    params = lasagne.layers.get_all_param_values(cnn["out"])

    print("Saving...")
    pickle.dump(params, open("params.pkl", "wb"), protocol=-1)
    print("Done")
