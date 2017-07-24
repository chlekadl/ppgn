# This script shows how to reconstruct from Caffenet features
#
# Alexey Dosovitskiy, 2015

import caffe
import numpy as np
import os
import sys
import patchShow
import scipy.misc
import scipy.io

# choose the net
if len(sys.argv) == 2:
  net_name = sys.argv[1]
else:
  raise Exception('Usage: recon_input.py NET_NAME')

# set up the inputs for the net: 
batch_size = 64
image_size = (3,227,227)
images = np.zeros((batch_size,) + image_size, dtype='float32')

# use crops of the cat image as an example 
in_image = scipy.misc.imread('Cat.jpg')
for ni in range(images.shape[0]):
  images[ni] = np.transpose(in_image[ni:ni+image_size[1], ni:ni+image_size[2]], (2,0,1))
# mirror some images to make it a bit more diverse and interesting
images[::2,:] = images[::2,:,:,::-1]
  
# RGB to BGR, because this is what the net wants as input
data = images[:,::-1] 

# subtract the ImageNet mean
matfile = scipy.io.loadmat('caffenet/ilsvrc_2012_mean.mat')
image_mean = matfile['image_mean']
topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
del matfile
data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

#initialize the caffenet to extract the features
caffe.set_mode_cpu() # replace by caffe.set_mode_gpu() to run on a GPU
caffenet = caffe.Net('caffenet/caffenet.prototxt', 'caffenet/caffenet.caffemodel', caffe.TEST)

# run caffenet and extract the features
caffenet.forward(data=data)
feat = np.copy(caffenet.blobs[net_name.split('_')[0]].data)
del caffenet

# run the reconstruction net
net = caffe.Net(net_name + '/generator.prototxt', net_name + '/generator.caffemodel', caffe.TEST)
generated = net.forward(feat=feat)
topleft = ((generated['generated'].shape[2] - image_size[1])/2, (generated['generated'].shape[3] - image_size[2])/2)
print(generated['generated'].shape, topleft)
recon = generated['generated'][:,::-1,topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
del net

print(images.shape, recon.shape)

# save results to a file
collage = patchShow.patchShow(np.concatenate((images, recon), axis=3), in_range=(-120,120))
scipy.misc.imsave('reconstructions_' + net_name + '.png', collage)

  