import os
import sys

import numpy as np
from scipy.misc import imread, imsave
from scipy.optimize import fmin_l_bfgs_b
import theano
import theano.tensor as tt
import lasagne

from vgg import load_vgg

content_layer = "conv4_2"
style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]#, "conv5_1"]

# Training mean from the VGG16 model page.
vgg_mean = np.array([103.939, 116.779, 123.68], np.float32)[::-1]

def get_covar_expr(net, layer):
    out = lasagne.layers.get_output(net[layer])
    C = tt.batched_dot(out.flatten(3), out.flatten(3).dimshuffle(0,2,1))
    return C / (4*out.shape[2]*out.shape[3])

def covar_matching_loss(net, layers, weights=None):
    if weights is None:
        weights = np.ones((len(layers),), theano.config.floatX) / len(layers)

    C_exprs = [get_covar_expr(net, layer) for layer in layers]
    C_targets = [theano.shared(ce.eval()) for ce in C_exprs]
    C_diffs = tt.concatenate([ tt.mean((ce - ct)**2, axis=[1,2])[:, np.newaxis] \
            for ce, ct in zip(C_exprs, C_targets) ], axis=1)
    batch_loss = tt.dot(C_diffs, weights)
    return batch_loss

def natural_img_prior(X):
    dx = X[:, :, :, 1:] - X[:, :, :, :-1]
    dy = X[:, :, 1:] - X[:, :, :-1]
    return 0.5 * (abs(dx).mean() + abs(dy).mean())

def make_grad(net, imgs, content_layer, style_layers, alpha):
    content_imgs, style_imgs = imgs
    X = net["in"].input_var

    X.set_value(content_imgs)
    content_layer_out = lasagne.layers.get_output(net[content_layer])
    content_target = theano.shared(content_layer_out.eval(), name="content_target")

    content_loss = ((content_layer_out - content_target)**2).mean()

    X.set_value(style_imgs)
    style_loss = covar_matching_loss(net, style_layers).mean()

    X.set_value(np.random.rand(*content_imgs.shape).astype(np.float32))
    cl = content_loss.eval()
    nl = natural_img_prior(X).eval()
    X.set_value(np.random.rand(*style_imgs.shape).astype(np.float32))
    sl = style_loss.eval()

    # Scale style loss so that it matches the magnitude of the content loss.
    style_loss *= cl / sl

    loss = (alpha            * content_loss +
                               style_loss +
            1e-4 * (cl / nl) * natural_img_prior(X))

    # X needs to have the correct shape (the content images shape) here before
    # the loss/update function is compiled.
    X.set_value(content_imgs)

    return theano.function([], [loss, tt.grad(loss, X)])

def transfer_style(content_img, style_img, net, alpha, verbose=False):
    X = net["in"].input_var

    content_imgs = (content_img - vgg_mean[np.newaxis, np.newaxis]).swapaxes(2,1).swapaxes(1,0)[np.newaxis]
    style_imgs = (style_img - vgg_mean[np.newaxis, np.newaxis]).swapaxes(2,1).swapaxes(1,0)[np.newaxis]

    if verbose:
        print("Compiling loss and gradient expression...")

    fg = make_grad(net,
            [content_imgs, style_imgs],
            content_layer, style_layers,
            alpha)

    if verbose:
        print("Optimizing...")

    def f_and_fprime(x):
        X.set_value(x.reshape(content_imgs.shape).astype(np.float32))
        f, g = fg()
        return f.item(), g.ravel().astype(np.float64)

    def save_img(x, it):
        if it[0] % 10 == 0:
            imsave("out/%d.png" % it[0],
                    (x.reshape(content_imgs.shape)[0] +
                        vgg_mean[:, np.newaxis, np.newaxis]).swapaxes(0,1).swapaxes(1,2)[:,:,::-1])
        it[0] += 1

    it = [0]
    cb = lambda x : save_img(x, it)

    x0 = (255 * np.random.rand(*content_imgs.shape)) - vgg_mean[np.newaxis, :, np.newaxis, np.newaxis]
    x0 = x0.ravel()

    x_opt = fmin_l_bfgs_b(
            func     = f_and_fprime,
            x0       = x0,
            maxfun   = 300,
            iprint   = 10 if verbose else -1,
            bounds   = [(-vgg_mean.max(),255-vgg_mean.max())] * len(x0),
            callback = cb if verbose else None,
            )[0]

    return x_opt.reshape(content_imgs.shape)[0].swapaxes(0,1).swapaxes(1,2) + vgg_mean[np.newaxis, np.newaxis]

if __name__ == "__main__":
    from scipy.misc import imresize
    content_img = imread(sys.argv[1]).astype(np.float32)[:,:,::-1]
    style_img = imresize(imread(sys.argv[2]), 0.5).astype(np.float32)[:,:,::-1]

    if not os.path.exists("out"):
        os.mkdir("out")

    net = load_vgg("vgg16_params.pkl")
    print("Loaded VGG net.")

    result = transfer_style(content_img, style_img, net, alpha = 1e-2, verbose=True)
    imsave("result.png", result[:,:,::-1])
