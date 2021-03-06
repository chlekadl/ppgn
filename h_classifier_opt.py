import matplotlib.pyplot as plt
import os, sys
import shutil
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

gen_in = settings.generator_in_layer
gen_out = settings.generator_out_layer

import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
import util
from sampler import Sampler

if settings.gpu:
    caffe.set_mode_gpu() # sampling on GPU

# load the Generator and the Classifier
encoder = caffe.Net("./nets/caffenet/caffenet_128.prototxt", settings.encoder_weights, caffe.TEST)

classifier = caffe.Classifier("./nets/caffenet/caffenet_128.prototxt", settings.encoder_weights,
                       mean = np.float32([104.0, 117.0, 123.0]), # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
#generator = caffe.Net("./nets/generator/noiseless/generator_batchsize_128.prototxt", settings.generator_weights, caffe.TEST)
generator = caffe.Net("./nets/generator/noiseless/generator_batchsize_128.prototxt", settings.generator_weights, caffe.TEST)
#generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)

gen_in = settings.generator_in_layer
gen_out = settings.generator_out_layer

h_shape = generator.blobs[gen_in].data.shape

# Get the input and output sizes
image_shape = encoder.blobs['data'].data.shape
generator_output_shape = generator.blobs[gen_out].data.shape

# Calculate the difference between the input image of the condition net 
# and the output image from the generator
image_size = util.get_image_size(image_shape)
generator_output_size = util.get_image_size(generator_output_shape)

# The top left offset to crop the output image to get a 227x227 image
topleft = util.compute_topleft(image_size, generator_output_size)

image_mean = scipy.io.loadmat('misc/ilsvrc_2012_mean.mat')['image_mean'] # (256, 256, 3)
image_mean = np.expand_dims(np.transpose(image_mean, (2,0,1)), 0)
#image_mean = np.repeat(image_mean, 10, axis=0)


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision.utils import save_image

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.dropout1 = nn.AlphaDropout(p=0.1)
        self.fc1 = nn.Linear(4096, 4096)
        self.dropout2 = nn.AlphaDropout(p=0.1)
        self.fc2 = nn.Linear(4096, 3500)
        self.dropout3 = nn.AlphaDropout(p=0.1)
        self.fc3 = nn.Linear(3500,3000)
        self.dropout4 = nn.AlphaDropout(p=0.1)
        self.fc4 = nn.Linear(3000,2500)
        self.dropout5 = nn.AlphaDropout(p=0.1)
        self.fc5 = nn.Linear(2500,2000)
        self.dropout6 = nn.AlphaDropout(p=0.1)
        self.fc6 = nn.Linear(2000,1500)
        self.fc7 = nn.Linear(1500,1000)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.selu(self.fc1(x))
        x = self.dropout2(x)
        x = F.selu(self.fc2(x))
        x = self.dropout3(x)
        x = F.selu(self.fc3(x))
        x = self.dropout4(x)
        x = F.selu(self.fc4(x))
        x = self.dropout5(x)
        x = F.selu(self.fc5(x))
        x = self.dropout6(x)
        x = F.selu(self.fc6(x))
        x = self.fc7(x)
        return x 
    
#    def __init__(self):
#        super(MLP, self).__init__()
#        self.dropout1 = nn.Dropout()
#        self.fc1 = nn.Linear(4096, 5000)
#        self.dropout2 = nn.Dropout()
#        self.fc2 = nn.Linear(5000, 2500)
#        self.fc3 = nn.Linear(2500,1000)
#
#    def forward(self, x):
#        x = self.dropout1(x)
#        x = F.relu(self.fc1(x))
#        x = self.dropout2(x)
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x 
    
    
def save_checkpoint(state, is_best, filename='checkpoint_Adam.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_Adam.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_code(images, label, reconstruction = False, lr=1, mom1=0.9, mom2=0.999, eps=1e-8, num_steps=1000, a=0, b=1, c=1e-5, d=1e-17):
    '''
    Generate <batch_size> h's corresponding to images generated by generator. 
    optimize h by h_best = argmax_h( ||x - G(h)||_2^2)
    return h and its corresponding class
    input image is in BGR
    '''
    if not reconstruction:
        cropped_x = images[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
        cropped_x_copy = cropped_x.copy()
        encoder.forward(data=cropped_x_copy)
        h = encoder.blobs['fc6'].data.copy()
        return h
    
    d_image = 0
    d_class = 0
    d_prior = 0
    
    #save_image(images[:,::-1], "./samples/" + str(label.numpy()) +  "_00.jpg")
    #util.save_image(images, "./samples/original_picture_" + str(label.numpy()) + ".jpg")
    
    t = 1
    m_t = np.zeros(h_shape)
    v_t = np.zeros(h_shape) 
    
    # initialize h from uniform distribution
    h = np.random.normal(0, 1, (images.shape[0], h_shape[1]))

    # initialize h from Encoder
    #cropped_x = images[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
    #cropped_x_copy = cropped_x.copy()
    #encoder.forward(data=cropped_x_copy)
    #h = encoder.blobs['fc6'].data.copy()

    for i in range(num_steps):
        # Push h through Generator to get image
        generator.blobs[gen_in].data[:] = h
        generated = generator.forward()
        x_gen = generated[gen_out].copy()       # batch_sizex3x256x256
        
        if i % 10 == 0:
            x_gen_save = (x_gen + image_mean)/255
            x_gen_save = torch.from_numpy(x_gen_save[:,::-1].copy())
            save_image(x_gen_save, "./samples/" + str(label.numpy()) + "_" + str(i) + ".jpg", nrow=5)
            #util.save_image(x_gen, "./samples/" + str(label.numpy()) + "_" + str(i) + ".jpg")
        
        ################## 1) get image reconstruction loss ################
        # calculate the l2 loss gradient wrt to output of generator
        loss = (x_gen-images)**2
        grad_loss = 2*(x_gen - images)

        # back-propagate the gradient to h
        generator.blobs[gen_out].diff[:] = grad_loss 
        diffs = generator.backward(start=gen_out, diffs=[gen_in])
    
        # d(image_loss)/dh 
        d_image = diffs[gen_in].copy() 
        ################################################################
        
        generator.blobs[gen_out].diff.fill(0.)   # reset objective after each step
        
        #################### 2) get class likelihood loss #################
        cropped_x_gen = x_gen[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
        cropped_x_gen_copy = cropped_x_gen.copy()
    
        dst = classifier.blobs['fc8']
                
        acts = classifier.forward(data=cropped_x_gen_copy, end='fc8')
                
        # Get the h resulting from E(G(h))
        #classifier.forward(data=cropped_x_gen_copy)
        code = classifier.blobs['fc6'].data.copy()
                
        one_hot = np.zeros_like(dst.data)
        
        # Get the activations
        layer_acts = acts['fc8']
        
        # Compute the softmax probs by hand because it's handy in case we want to condition on hidden units as well
        exp_acts = np.exp(layer_acts - np.max(layer_acts))
        probs = exp_acts / (1e-10 + np.sum(exp_acts, keepdims=True))
        
        # The gradient of log of softmax, log(p(y|x)), reduces to:
        softmax_grad = 1 - probs.copy()
        
        obj_prob = probs.flat[label[0]]
        
        # Assign the gradient 
        for i in range(len(label)):
            one_hot[i][label[i]] = softmax_grad[i][label[i]]
            #one_hot.flat[label[0]] = softmax_grad[label[0]]
            dst.diff[:] = one_hot
            
        # Backpropagate the gradient to the image layer
        diffs = classifier.backward(start='fc8', diffs=['data'])
        d_class_dx = diffs['data'].copy()
    
        dst.diff.fill(0.)   # reset objective after each step
                
        # change gradient from 3x227x227 to 3x256x256
        d_condition_x256 = np.zeros(generator_output_shape)
        d_condition_x256[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = d_class_dx.copy()
                
        # back propagate class likelihood loss to h
        generator.blobs[gen_out].diff[:] = d_condition_x256
        diffs = generator.backward(start=gen_out, diffs=[gen_in])
                
        #d(class_loss)/dh
        d_class = diffs[gen_in].copy()
        
        generator.blobs[gen_out].diff.fill(0.)   # reset objective after each step
        ###############################################################
        
        
        #################### 3) get the prior loss #######################
        d_prior = code - h
        ##################################################################
        
        noise = np.random.normal(0, d, h.shape)  # Gaussian noise
        
        d_h = a*d_image- b*d_class - c*d_prior - noise
        
        ################ Adam ################
        m_t = mom1*m_t + (1-mom1)*d_h
        v_t = mom2*v_t + (1-mom2)*(d_h**2)
        m_t_hat = m_t/(1-mom1**t)
        v_t_hat = v_t/(1-mom2**t)
        step_size = lr
        t += 1
            
        #h -= step_size*m_t_hat/(np.sqrt(v_t_hat) + eps)
        h -= step_size/np.abs(d_h).mean() * d_h
        #h += step_size*d_h
        
        # Stochastic clipping
        h = np.clip(h, a_min=0, a_max=30)
        #h[h>=30] = np.random.uniform(0, 30)
        #h[h<=0] = np.random.uniform(0, 30)
        
        if i % 100 == 0:
            print("step[%d/%d] loss: %.1f obj_prob: %.1f" %(i, num_steps, np.sum(loss), obj_prob))
    
    return h


def train(train_loader, model, criterion, optimizer, epoch, print_freq, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if len(target) != batch_size:
            break
        
        #save_image(input, "./samples/" + str(target.numpy()) +  "_00.jpg", nrow=5)
        
        images = input.numpy()
        data = 255*images[:,::-1]   # convert from RGB to BGR

        # subtract the ImageNet mean
        data -= image_mean    # mean is already BGR
        
        inputs = get_code(data, target)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        
        # save the final generated image
        #images_gen = (images_gen + image_mean)/255
        #images_gen = torch.from_numpy(images_gen[:,::-1].copy())
        #save_image(images_gen, "./samples/" + str(target.numpy()) + "_final.jpg", nrow=5)
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, print_freq, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if len(target) != batch_size:
            break
        images = input.numpy()
        data = 255*images[:,::-1]   # convert from RGB to BGR
                
        # subtract the ImageNet mean
        data -= image_mean    # mean is already BGR
        
        inputs = get_code(data, target)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def torchweights_to_caffe(model):
    h_classifier_caffe = caffe.Net("/home/choidami/ml/ppgn/nets/h_classifier/h_classifier_Adam.prototxt", caffe.TEST)
    h_classifier_caffe.params["fc7"][0].data[...] = h_classifier.fc1.weight.data.cpu().numpy()
    h_classifier_caffe.params["fc7"][1].data[...] = h_classifier.fc1.bias.data.cpu().numpy()
    h_classifier_caffe.params["fc8"][0].data[...] = h_classifier.fc2.weight.data.cpu().numpy()
    h_classifier_caffe.params["fc8"][1].data[...] = h_classifier.fc2.bias.data.cpu().numpy()
    h_classifier_caffe.params["fc9"][0].data[...] = h_classifier.fc3.weight.data.cpu().numpy()
    h_classifier_caffe.params["fc9"][1].data[...] = h_classifier.fc3.bias.data.cpu().numpy()
    h_classifier_caffe.save('/home/choidami/ml/ppgn/nets/h_classifier/h_classifier_Adam.caffemodel')
    return
    

if __name__ == '__main__':
    # arguments
    resume = ""#"./checkpoints/checkpoint_Adam_1epoch_1e-4lr.pth.tar" #"/home/choidami/ml/ppgn/checkpoint_Adam.pth.tar"
    data = "/home/damichoi/imagenet/"
    workers = 4
    start_epoch = 0
    epochs = 1
    evaluate = False
    print_freq = 10
    best_prec1 = 0
    
    # parameters
    batch_size = 128
    num_iter = 1
    lr = 0.1
    mom1 = 0.9
    mom2 = 0.999
    eps = 1e-8

    # Create model
    h_classifier = MLP().cuda()
    
    # define loss function (criterion) and optiizer    
    criterion = nn.CrossEntropyLoss().cuda()
#    optimizer = optim.SGD(h_classifier.parameters(), 0.1,
#                                momentum=0.9,
#                                weight_decay=1e-4)
    optimizer = optim.Adam(h_classifier.parameters(), lr=1e-4,
                           betas=(0.9,0.999), eps=1e-08, weight_decay=0.0005)
    
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            h_classifier.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            
    cudnn.benchmark = True
    
    # Data loading code
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    # Initialize weights from caffeNet
    #fc1_W = torch.from_numpy(W_fc7).cuda()
    #fc1_b = torch.from_numpy(b_fc7).cuda()
    #fc2_W = torch.from_numpy(W_fc8).cuda()
    #fc2_b = torch.from_numpy(b_fc8).cuda()
    #h_classifier.fc1.weight.data = fc1_W
    #h_classifier.fc1.bias.data = fc1_b
    #h_classifier.fc2.weight.data = fc2_W
    #h_classifier.fc2.bias.data = fc2_b

    if evaluate:
        validate(val_loader, h_classifier, criterion, print_freq, batch_size)
    else:
        for epoch in range(start_epoch, epochs):
            #adjust_learning_rate(optimizer,epoch, lr)
            
            # train for one epoch
            train(train_loader, h_classifier, criterion, optimizer, epoch, print_freq, batch_size)
            
            # evaluate on validation set
            prec1 = validate(val_loader, h_classifier, criterion, print_freq, batch_size)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': h_classifier.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename="./checkpoints/7layer_SELU_Adam_1epoch_1e-4lr.pth.tar")
    
    

    
    
    
    
    
    
    
#    images = np.zeros_like(generator.blobs[gen_out].data, dtype='float32')
#    in_image = scipy.misc.imread('/home/choidami/ImageNet/train/n03584254/n03584254_52.JPEG') # ipod
#    #in_image = scipy.misc.imread('/home/choidami/ImageNet/train/n09835506/n09835506_162.JPEG') # Ballplayer
#    #in_image = scipy.misc.imread('/home/choidami/ImageNet/train/n03661043/n03661043_507.JPEG') # Library
#    #in_image = scipy.misc.imread('/home/choidami/ImageNet/train/n03345487/n03345487_209.JPEG') # Fire Engine
#    #in_image = scipy.misc.imread('/home/choidami/ImageNet/train/n07730033/n07730033_195.JPEG') # cardoon
#    
#    in_image = scipy.misc.imresize(in_image, (generator_output_size[0], generator_output_size[1]))
#    images[0] = np.transpose(in_image, (2, 0, 1))   # convert to (3, 227, 227) format
#
#    data = images[:,::-1]   # convert from RGB to BGR
#    
#    util.save_image(data, "./samples/original_image.jpg")
#
#    # subtract the ImageNet mean
#    image_mean = scipy.io.loadmat('misc/ilsvrc_2012_mean.mat')['image_mean'] # (256, 256, 3)
#    data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0)    # mean is already BGR
#    
#    get_code(data)
    
    #util.save_image(data, "./samples/original_image.jpg")
    
    
#    in_image = scipy.misc.imresize(in_image, (generator_output_shape[2], generator_output_shape[3]))
#    images[0] = np.transpose(in_image, (2, 0, 1))   # convert to (3, 227, 227) format
#    images= images[:,::-1]   # convert from RGB to BGR
#    util.save_image(images, "./samples/original_image.jpg")
#
#    
#    images= images[:,::-1]   # convert from RGB to BGR
#    
#    # initialize h from Encoder
#    images = images[:,::-1]
#    cropped_x = images[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
#    cropped_x_copy = cropped_x.copy()

#    encoder.forward(data=data)
#    h = encoder.blobs['fc6'].data.copy()
##    acts = encoder.forward(data=cropped_x_copy, end='fc6')
##    h = np.reshape(acts['fc6'][0], h_shape)
#    
#    # Push h through Generator to get image
#    generator.blobs[gen_in].data[:] = h
#    generated = generator.forward()
#    x_gen = generated[gen_out].copy()       # batch_sizex3x256x256
#        
#    util.save_image(x_gen, "./samples/" + str(0) + ".jpg")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
