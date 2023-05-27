import cv2
import numpy as np
import os
import random
import keras
import copy
from keras.preprocessing.image import ImageDataGenerator
#from Dataloader import Dataloader ##
import imutils
import matplotlib.pyplot as plt

def cropImage(PATH):
  img = cv2.imread(PATH)
  # img = cv2.resize(img, (256, 256))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  gray = cv2.GaussianBlur(gray, (5, 5), 0)

  # threshold the image, then perform a series of erosions +
  # dilations to remove any small regions of noise
  thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)
  # find contours in thresholded image, then grab the largest one
  # cv2.imwrite('thresh.jpg', thresh)
  thresh_copy = thresh.copy()
  cnts = cv2.findContours(thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  # print(cnts)
  c = max(cnts, key=cv2.contourArea)
  min_left_point_idx = c[:, :, 0].argmin()
  max_right_point_idx = c[:, :, 0].argmax()
  min_bottom_point_idx = c[:, :, 1].argmin()
  max_top_point_idx = c[:, :, 1].argmax()
  # print(c[min_left_point_idx])
  # find the extreme points
  leftExtremePoint = tuple(c[min_left_point_idx][0])
  rightExtremePoint = tuple(c[max_right_point_idx][0])
  bottomExtremePoint = tuple(c[min_bottom_point_idx][0])
  topExtremePoint = tuple(c[max_top_point_idx][0])

  # crop
  min_x, max_x, min_y, max_y = leftExtremePoint[0], rightExtremePoint[0], bottomExtremePoint[1], topExtremePoint[
    1]
  # print("The extreme points coordinates ",min_x,max_x,min_y,max_y)
  cropped_img = img[min_y:max_y, min_x:max_x].copy()
  return cropped_img

def RegionInpaintAugmentation(image_path, seg_model, inpainting_model):
  """
  :param image_path: A tumor image path to be augmented by removing the tumor from the image
  :param seg_model: Segmentation model
  :param inpainting_model: Inpainting model
  :return: Augmented image
  """
  cropped_img = cropImage(image_path)
  img = cv2.resize(cropped_img, (256, 256))
  img = img / 255.0

  fontdict = {'fontsize': 18}
  plt.figure(figsize=(10, 10))
  plt.subplot(2, 2, 1)
  # plt.subplot(141)
  plt.xticks([])
  plt.yticks([])
  plt.title('Original image', fontdict=fontdict)
  plt.imshow(img)

  predicted = seg_model.predict(np.reshape(img, (1, 256, 256, 3)))
  predicted = np.reshape(predicted, (256, 256))
  predicted = predicted.astype(np.float32) * 255
  _, thresh = cv2.threshold(predicted, 127, 255, cv2.THRESH_BINARY)
  thresh = cv2.dilate(thresh, None, iterations=3)
  thresh = cv2.erode(thresh, None, iterations=3)
  thresh = cv2.dilate(thresh, None, iterations=2)

  plt.subplot(2, 2, 2)
  plt.xticks([])
  plt.yticks([])
  plt.title('Predicted mask', fontdict=fontdict)
  plt.imshow(thresh, vmin=0, vmax=255, cmap="gray")

  plt.subplot(2, 2, 3)
  plt.xticks([])
  plt.yticks([])
  plt.title('Inverted mask', fontdict=fontdict)

  plt.imshow(cv2.bitwise_not(thresh), vmin=0, vmax=255, cmap="gray")

  mask = thresh / 255
  mask = 1 - mask
  mask = np.stack((mask,) * 3, axis=-1)
  mask = np.uint8(mask)

  inpainting_result = inpainting_model.predict([np.expand_dims(img, 0), np.expand_dims(mask, 0)])

  plt.subplot(2, 2, 4)
  plt.xticks([])
  plt.yticks([])
  plt.title('Inpainted Image', fontdict=fontdict)
  plt.imshow(np.squeeze(inpainting_result))

  plt.show()

  return inpainting_result

def CutoffAugmentation(tumor_file_path,no_tumor_file_path,seg_model):
  """
   :param tumor_file_path: A tumor image path
   :param no_tumor_file_path: A non-tumor image path
   :param seg_model: Segmentation model
   :return: Augmented image
   """
  tumor_img = cropImage(tumor_file_path)
  non_tumor_img = cropImage(no_tumor_file_path)

  tumor_img = cv2.resize(tumor_img, (256, 256))
  tumor_img = tumor_img / 255.0

  non_tumor_img = cv2.resize(non_tumor_img, (256, 256))
  non_tumor_img = non_tumor_img / 255.0

  fontdict = {'fontsize': 18}
  plt.figure(figsize=(15, 10))
  plt.subplot(2, 3, 1)
  plt.xticks([])
  plt.yticks([])
  plt.title('Tumor image', fontdict=fontdict)
  plt.imshow(tumor_img)

  plt.subplot(2, 3, 2) 
  plt.xticks([])
  plt.yticks([])
  plt.title('Non Tumor image', fontdict=fontdict)
  plt.imshow(non_tumor_img)

  predicted = seg_model.predict(np.reshape(tumor_img, (1, 256, 256, 3)))
  predicted = np.reshape(predicted, (256, 256))
  predicted = predicted.astype(np.float32) * 255
  _, thresh = cv2.threshold(predicted, 127, 255, cv2.THRESH_BINARY)
  thresh = cv2.dilate(thresh, None, iterations=5)
  thresh = cv2.erode(thresh, None, iterations=5)

  plt.subplot(2, 3, 3)  
  plt.xticks([])
  plt.yticks([])
  plt.title('Predicted mask', fontdict=fontdict)
  plt.imshow(thresh, vmin=0, vmax=255, cmap="gray")

  x1, x2 = np.where(thresh == 255)
  segmented_tumor = np.zeros((256, 256, 3))   
  segmented_tumor[x1, x2, :] = tumor_img[x1, x2, :] 
  plt.subplot(2, 3, 4)  
  plt.xticks([]) 
  plt.yticks([])  
  plt.title('Segmented Tumor', fontdict=fontdict)  
  plt.imshow(segmented_tumor) 

  non_tumor_img[x1, x2, :] = tumor_img[x1, x2, :]
  cutoff_image = non_tumor_img.copy()
  blurred_img = cv2.GaussianBlur(cutoff_image, (3, 3), 0)
  plt.subplot(2, 3, 5)
  plt.xticks([])
  plt.yticks([])
  plt.title('Augmented Image', fontdict=fontdict)
  plt.imshow(blurred_img)
  plt.show()

  return cutoff_image


def MixUpGenerator(X1,X2,y1,y2,alpha=1):
  #h, w, c = X1.shape
  l = None
  while True:
    l = np.random.beta(alpha, alpha, 1)
    if l[0] > 0.3 and l[0] < 0.7:
      continue
    else:
      break
  X_l = l.reshape(1, 1, 1, 1)
  y_l = l.reshape(1, 1)

  print("probabolity = {}".format(l[0]))
  X = X1 * X_l + X2 * (1 - X_l)
  y = y1 * y_l + y2 * (1 - y_l)
  return X[0], y[0]


def RegionMixAugmentation(tumor_file_path,no_tumor_file_path, seg_model):

  tumor_img = cropImage(tumor_file_path)
  non_tumor_img = cropImage(no_tumor_file_path)

  tumor_img = cv2.resize(tumor_img , (256, 256))
  tumor_img = tumor_img / 255.0

  non_tumor_img = cv2.resize(non_tumor_img , (256, 256))
  non_tumor_img = non_tumor_img / 255.0

  fontdict = {'fontsize': 18}
  plt.figure(figsize=(15, 10))
  plt.subplot(2, 3, 1)
  plt.xticks([])
  plt.yticks([])
  plt.title('Tumor image', fontdict=fontdict)
  plt.imshow(tumor_img)

  plt.subplot(2, 3, 2)
  plt.xticks([])
  plt.yticks([])
  plt.title('Non Tumor image', fontdict=fontdict)
  plt.imshow(non_tumor_img)


  predicted = seg_model.predict(np.reshape(tumor_img, (1, 256, 256, 3)))
  predicted = np.reshape(predicted, (256, 256))
  predicted = predicted.astype(np.float32) * 255
  _, thresh = cv2.threshold(predicted, 127, 255, cv2.THRESH_BINARY)
  thresh = cv2.dilate(thresh, None, iterations=3)
  thresh = cv2.erode(thresh, None, iterations=3)

  plt.subplot(2, 3, 3)
  plt.xticks([])
  plt.yticks([])
  plt.title('Predicted mask', fontdict=fontdict)
  plt.imshow(thresh, vmin=0, vmax=1, cmap="gray")

  x1, x2 = np.where(thresh == 255)
  segmented_tumor = np.zeros((256, 256, 3))
  segmented_tumor[x1, x2, :] = tumor_img[x1, x2, :]

  plt.subplot(2, 3, 4)
  plt.xticks([])
  plt.yticks([])
  plt.title('Segmented tumor', fontdict=fontdict)
  plt.imshow(segmented_tumor)

  mixed_img, Y = MixUpGenerator(segmented_tumor, non_tumor_img, [1, 0], [0, 1])

  non_tumor_img[x1, x2, :] = mixed_img[x1, x2, :]
  mixed_img= non_tumor_img.copy()
  plt.subplot(2, 3, 5)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("Label= {}".format(Y),fontsize=12)
  plt.title('Augmented Image', fontdict=fontdict)
  plt.imshow(mixed_img)   #mixed_img
  plt.show()

  #plt.xticks([])
  #plt.yticks([])
  #plt.imshow(scipy.ndimage.rotate(mixed_img, -27, reshape=False, order=0))
  #plt.show()

def generate_basic_augmented_images_through_path(file_path,num_samples,save_path):
  if os.path.exists(save_path)==False:
    os.makedirs(save_path)
  datagen = ImageDataGenerator(
        rotation_range = 10,
        horizontal_flip = True,
        vertical_flip=True,
        brightness_range = (0.7, 1.3),
        fill_mode='nearest')
  all_images=[]
  files=[]
  fontdict = {'fontsize': 18}
  for file in os.listdir(file_path):

    img = cv2.imread(file_path + '/' + file)
    img=cv2.resize(img,(256,256))
    all_images.append(img)
    files.append(file)
  i = 0
  while(True):
    rand_img_index=random.randint(0,len(all_images)-1)
    rand_img=all_images[rand_img_index]
    rand_filename=files[rand_img_index]
    save_prefix=rand_filename.split('.')[0]+'_aug'
    plt.figure(figsize=(10, 10))
    plt.subplot(num_samples, 2, (i*2)+1)
    #plt.subplot(1, 2, 1)
    plt.title("Original image",fontdict=fontdict)
    plt.axis('off')
    plt.imshow(rand_img)
    plt.axis('off')
    #plt.show()
    rand_img=np.expand_dims(rand_img,0)
    augmented_samples_per_image=0
    for batch in datagen.flow(rand_img, batch_size=1,
                          save_to_dir= save_path,
                          save_prefix=save_prefix,
                          save_format='jpg'):
      augmented_img=batch[0].astype('uint8')
      plt.subplot(num_samples, 2, (i*2)+2)
      #plt.subplot(1, 2, 2)
      plt.title("Augmented image",fontdict=fontdict)
      plt.imshow(augmented_img)
      plt.axis('off')
      plt.show()
      augmented_samples_per_image=augmented_samples_per_image+1
      if augmented_samples_per_image==1:
        break
    i += 1
    if i > num_samples-1:
      break

def generate_basic_augmented_images(data,num_samples,save_path):
  if os.path.exists(save_path)==False:
    os.makedirs(save_path)
  datagen = ImageDataGenerator(
        rotation_range = 10,
        horizontal_flip = True,
        vertical_flip=True,
        brightness_range = (0.7, 1.3),
        fill_mode='nearest')

  i = 0
  while(True):
    rand_img_index=random.randint(0,len(data)-1)
    rand_img=data[rand_img_index]
    save_prefix=str(i+1)+'_aug'

    rand_img=np.expand_dims(rand_img,0)
    augmented_samples_per_image=0
    for batch in datagen.flow(rand_img, batch_size=1,
                          save_to_dir= save_path,
                          save_prefix=save_prefix,
                          save_format='jpg'):
      augmented_samples_per_image=augmented_samples_per_image+1
      if augmented_samples_per_image==1:
        break
    i += 1
    if i > num_samples:
      break
