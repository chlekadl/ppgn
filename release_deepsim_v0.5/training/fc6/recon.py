import caffe
import numpy as np
import time
import os
import sys
from matplotlib import pyplot as plt
import mypy
import scipy.misc


snapshot_folder = 'snapshots'
if len(sys.argv) > 1:
  start_snapshot = sys.argv[1]
else:
  raise Exception('Usage: recon.py SNAPSHOT_ITER')
#initialize the nets
caffe.set_device(0)
caffe.set_mode_gpu()
encoder = caffe.AdamSolver('solver_encoder.prototxt')
generator = caffe.AdamSolver('solver_generator.prototxt')
data_reader = caffe.AdamSolver('solver_data_val.prototxt')
feat_shape = (4096,)
recogn_feat_shape = (256,6,6)
batch_size = 64
burn_in = 0

encoder.net.copy_from('../../trained_models/caffenet.caffemodel')

#load from snapshot
if start_snapshot:
  curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
  print '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
  generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
  if os.path.isfile(generator_caffemodel):
    generator.net.copy_from(generator_caffemodel)
  else:
    raise Exception('File %s does not exist' % generator_caffemodel)

for ni in range(burn_in):
  print 'Burn-in batch %d/%d' % (ni+1, burn_in)
  data = data_reader.net.forward()

curr_batch = burn_in
while True:
  curr_batch += 1
  print 'Showing batch %d' % curr_batch
  data = data_reader.net.forward()
  encoded = encoder.net.forward(data=data['data'], feat_gt=np.zeros((batch_size,) + recogn_feat_shape, dtype='float32'))
  generated = generator.net.forward(feat=encoded['feat_out'].reshape((batch_size,) + feat_shape), data=data['data'])
  images = encoder.net._blobs[encoder.net._blob_names_index['data']].data[:,::-1]
  recon = generated['generated'][:,::-1]
  #plt.figure(figsize=(4*5,3*4.8))
  collage=mypy.patchShow(np.concatenate((images[:16], recon[:16]), axis=3), in_range=(-150,150), display  = False)
  save_file = '../imgs/net68_' + start_snapshot + '_batch' + str(curr_batch) + '.png'
  print 'saving to ' + save_file 
  scipy.misc.imsave(save_file, collage)
  raw_input()
  #plt.show()
