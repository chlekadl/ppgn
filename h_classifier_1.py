import sys

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

if settings.gpu:
    caffe.set_mode_gpu() # sampling on GPU 
  
alexNet = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)
W_fc7 = alexNet.params["fc7"][0].data[...]
b_fc7 = alexNet.params["fc7"][1].data[...]
W_fc8 = alexNet.params["fc8"][0].data[...]
b_fc8 = alexNet.params["fc8"][1].data[...]

h_classifier = caffe.Net(settings.h_classifier_definition, caffe.TEST)
h_classifier.params["fc7"][0].data[...] = W_fc7
h_classifier.params["fc7"][1].data[...] = b_fc7
h_classifier.params["fc8"][0].data[...] = W_fc8
h_classifier.params["fc8"][1].data[...] = b_fc8

h_classifier.save('/home/choidami/ml/ppgn/nets/h_classifier/h_classifier.caffemodel')

