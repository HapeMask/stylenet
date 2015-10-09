import sys
import pickle

import caffe

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python2 convert_caffe_params.py PROTOTOTXT_FILE CAFFEMODEL_FILE")
        exit(1)

    prototxt_fn, model_fn = sys.argv[1:3]
    net = caffe.Net(prototxt_fn, model_fn, caffe.TEST)
    params = {}
    for layer in net.params.keys():
        # unicode() here lets python3 load and access the dictionary like this:
        # ```params = pickle.load(open(params_filename, "rb"), encoding="bytes")```
        params[unicode(layer)] = [p.data for p in net.params[layer]]

    pickle.dump(params, open("vgg16_params.pkl", "wb"), protocol=-1)
