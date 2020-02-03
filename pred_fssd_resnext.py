from data import *

import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")    

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--trained_model',
                    default='weights/model.pth', type=str,
                    help='Trained state_dict file path to open')                    
parser.add_argument('--use_pred_module', default=True, type=str2bool,
                    help='Use prediction module')
parser.add_argument('--confidence_threshold', default=0.6, type=float,
                    help='Detection confidence threshold')
parser.add_argument('input_file')

args = parser.parse_args()

    
from fssd512_resnext import build_ssd

if args.dataset == 'COCO':
    from data import COCO_CLASSES as labelmap    
    labelmap = COCO_CLASSES
    num_classes = len(labelmap) + 1                    # +1 for background
    cfg = cocod512
elif args.dataset == 'VOC':    
    from data import VOC_CLASSES as labelmap
    labelmap = VOC_CLASSES
    num_classes = len(labelmap) + 1                    # +1 for background
    cfg = vocd512
    
net = build_ssd('test', cfg, args.use_pred_module) 
net.load_state_dict(torch.load(args.trained_model))

net.eval()

image = cv2.imread(args.input_file, cv2.IMREAD_COLOR) 
image = image[:, :, (2, 1, 0)]
x = base_transform(image, 512, (104.0, 117.0, 123.0))
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
plt.imshow(image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= args.confidence_threshold:
        score = detections[0,i,j,0]
        label_name = labelmap[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1
        print(display_txt, score, coords)

# plt.show()
plt.savefig('figure.png')
