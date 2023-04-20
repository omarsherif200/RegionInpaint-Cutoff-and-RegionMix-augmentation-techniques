import numpy as np
import tensorflow as tf
import keras
import glob
import math
import argparse
import Dataloader_seg
from segmentation_utils import *
from ResUNet import ResUNet
from SegNet import SegNet
from Unet import UNet
from VGGUnet import VGGUnet

train_data_path=r"D:/TA/Masters/BR35H dataset/Br35H-Mask-RCNN/TRAIN/"
train_annot_path=r"D:/TA/Masters/BR35H dataset/Br35H-Mask-RCNN/annotation_train/"
val_data_path=r"D:/TA/Masters/BR35H dataset/Br35H-Mask-RCNN/TEST/"
val_annot_path=r"D:/TA/Masters/BR35H dataset/Br35H-Mask-RCNN/annotation_test/"

parser=argparse.ArgumentParser(description='Segmentation Script')

parser.add_argument('--train_data',default=train_data_path,type=str,help='Training images path')
parser.add_argument('--train_annot',default=train_annot_path,type=str,help='Training masks path')
parser.add_argument('--val_data',default=val_data_path,type=str,help='Validation images path')
parser.add_argument('--val_annot',default=val_annot_path,type=str,help='Validation masks path')
parser.add_argument('--image_size',default=256,type=int,help='Image Dimension')
parser.add_argument('--lr',default=1e-5,type=int,help='Learning Rate')
parser.add_argument('--epochs',default=90,type=int,help='Number of Iterations')
parser.add_argument('--batch_size',default=2,type=int,help='Batch size')
parser.add_argument('--pretrained_weights',default=None,type=str,help='pretrained weights path of the model')

args=parser.parse_args()
obj=VGGUnet(args)
#obj=UNet(args)
#obj=SegNet(args)
#obj=ResUNet(args)

seg=Dataloader_seg.Dataloader()



xt_images = glob.glob(args.train_data + "*.jpg")
xt_images.sort()

xa_images = glob.glob(args.train_annot + "*.png")
xa_images.sort()

xt_images_val = glob.glob(args.val_data+ "*.jpg")
xt_images_val.sort()

xa_images_val = glob.glob(args.val_annot + "*.png")
xa_images_val.sort()

Batch_size=args.batch_size
train_generator = seg.Generator(xt_images,xa_images,image_size=args.image_size, batch_size=Batch_size)
validation_generator = seg.Generator(xt_images_val,xa_images_val,image_size=args.image_size, batch_size=Batch_size)

train_num_of_batches=math.ceil(len(xt_images)/Batch_size)
val_num_of_batches=math.ceil(len(xt_images_val)/Batch_size)

obj.run_model(train_generator,validation_generator,train_num_of_batches,val_num_of_batches)