import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from inpainting_preprocess import create_input_pipeline_func, create_input_pipeline_test, create_input_dataset
from pconv_layer import PConv2D
from pconv_model import pconv_model

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def display(display_list):
  plt.figure(figsize=(8, 8))
  title = ['Input Masked Image', 'Input Mask Image', 'Ground Truth', 'Predicted Image']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.tight_layout()
    plt.axis('off')
  plt.show()

model_path=r"D:\TA\Masters\brain tumor small dataset\pconv_imagenet.h5"
parser=argparse.ArgumentParser(description='Image inpainting')
checkpoint_path=r'inpainting_model' ##
if os.path.exists(checkpoint_path)==False:
  os.makedirs(checkpoint_path)

train_dir = r'D:\TA\Masters\BR35H dataset\no for inpainting'
validation_dir = r'D:\TA\Masters\brain tumor small dataset\no'
finetune=True

args=None

parser.add_argument('--train_dir', default=train_dir, type=str, help='Directory for training images')
parser.add_argument('--validation_dir', default=validation_dir, type=str, help='Directory for validation images')
parser.add_argument('--training_option',default='finetune',type=str,help='train the model from scratch or finetune',choices=['train', 'finetune'])
parser.add_argument('--IMAGE_SIZE', default=256, type=int, help='Image Dimension')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epochs', default=20, type=int, help='Number of iterations')
parser.add_argument('--steps_per_epoch', default=400, type=int, help='Number of steps per epoch')
parser.add_argument('--validation_steps', default=250, type=int, help='Number of validation_steps per epoch')
parser.add_argument('--checkpoint_path', default=checkpoint_path, type=str,
                    help='The path where the trained model will be saved')
args,_=parser.parse_known_args()
if args.training_option=='train':
  parser.add_argument('--train_bn', default=True, action=argparse.BooleanOptionalAction,
                      help='Batch Normalization is freezed or not')
  parser.add_argument('--pretrained_model_path', default=None, type=str, help='Path of the pretrained model')
  parser.add_argument('--lr', default=0.0001, type=int, help='Learning rate')

elif args.training_option=='finetune':
  parser.add_argument('--train_bn', default=False, action=argparse.BooleanOptionalAction,
                      help='Batch Normalization is freezed or not')
  parser.add_argument('--pretrained_model_path', default=r"D:\TA\Masters\brain tumor small dataset\pconv_imagenet.h5", type=str, help='Path of the pretrained model')
  parser.add_argument('--lr', default=0.00001, type=int, help='Learning rate')
args = parser.parse_args()

# return the training dataset in batches
train_dataset = create_input_pipeline_func(args.train_dir, batch_size=args.batch_size)
# return the validation dataset in batches
val_dataset = create_input_pipeline_func(args.validation_dir, batch_size=args.batch_size)


for target_batch in train_dataset.take(1):
  (masked_batch, mask_batch), target_batch = create_input_dataset(target_batch, batch_size=args.batch_size)

for b in range(args.batch_size):
  display([masked_batch[b], mask_batch[b], target_batch[b]])

inpainting_model=pconv_model(args)
inpainting_model.run_model(train_dataset,val_dataset)