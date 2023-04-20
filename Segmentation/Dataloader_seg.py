import skimage.transform as trans
import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import math
class Dataloader():

    def read_imgs(self,images,image_size):
        """
        :param images:  images path
        :param image_size: required image size
        :return: List of images
        """
        imgs_list = []
        for img in images:
            # image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(img)
            image = image / 255
            # image = trans.resize(image,(256,256))
            image = cv2.resize(image, (image_size, image_size))
            imgs_list.append(image)
        return imgs_list

    def read_masks(self,masks,image_size):
        """
        :param masks: corresponding masks path
        :param image_size: required mask size
        :return: List of masks
        """
        annot_list = []
        for msk in masks:
            mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255
            # mask = trans.resize(mask,(256,256))
            mask = cv2.resize(mask, (image_size, image_size))
            mask[mask != 0] = 1
            annot_list.append(mask)
        return annot_list



    def Generator(self,imgs_files, masks_files, image_size=256, batch_size=2):
        """
        :param imgs_files: images path
        :param masks_files: masks path
        :param image_size: required size
        :param batch_size: number of mini-batches
        :return: generator object for each batch of images and masks
        """
        while True:
            num_batches = math.ceil(len(imgs_files) / batch_size)
            imgs = None
            masks = None
            for i in range(0, num_batches):
                if i < num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:current_batch_index + batch_size]

                    batch_masks_files = masks_files[current_batch_index:current_batch_index + batch_size]
                    imgs = self.read_imgs(batch_imgs_files,image_size)
                    masks = self.read_masks(batch_masks_files,image_size)


                elif i == num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:]
                    batch_masks_files = masks_files[current_batch_index:]
                    imgs = self.read_imgs(batch_imgs_files,image_size)
                    masks = self.read_masks(batch_masks_files,image_size)
                imgs = np.array(imgs)
                masks = np.array(masks)
                yield (imgs, masks)


