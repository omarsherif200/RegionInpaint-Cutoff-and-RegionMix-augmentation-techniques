import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from keras.metrics import Recall, Precision
from keras.optimizers import Adam
from segmentation_utils import *

class ResUNet():
    def __init__(self, args):
        self.image_size = args.image_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.pretrained_weights = args.pretrained_weights

    def bn_act(self,x, act=True):
        x = keras.layers.BatchNormalization()(x)
        if act == True:
            x = keras.layers.Activation("relu")(x)
        return x

    def conv_block(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = self.bn_act(x)
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)

        output = keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)

        output = keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(self,x, xskip):
        u = keras.layers.UpSampling2D((2, 2))(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c

    def ResUNet_model(self):
        """
        this function creates the ResUnet model and loads the pretrained weights if exists
        """
        f = [16, 32, 64, 128, 256]
        inputs = keras.layers.Input(shape=(self.image_size, self.image_size, 3))

        ## Encoder
        e0 = inputs
        e1 = self.stem(e0, f[0])
        e2 = self.residual_block(e1, f[1], strides=2)
        e3 = self.residual_block(e2, f[2], strides=2)
        e4 = self.residual_block(e3, f[3], strides=2)
        e5 = self.residual_block(e4, f[4], strides=2)

        ## Bridge
        b0 = self.conv_block(e5, f[4], strides=1)
        b1 = self.conv_block(b0, f[4], strides=1)

        ## Decoder
        u1 = self.upsample_concat_block(b1, e4)
        d1 = self.residual_block(u1, f[4])

        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, f[3])

        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, f[2])

        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, f[1])

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
        model = keras.models.Model(inputs, outputs)
        if (self.pretrained_weights):
            model.load_weights(self.pretrained_weights)
        return model

    def run_model(self,train_gen,val_gen,steps_per_epoch,val_steps_per_epoch):
        """
        this function creates, compiles and fit the model,
        the loss function and optimizer can be tuned ex: optimizer can be SGD or Rmsprop etc.
        :param train_gen: training generator
        :param val_gen: validation generator
        :param steps_per_epoch: training steps for each epoch
        :param val_steps_per_epoch: validation steps for each epoch
        :return: the model after being trained
        """
        model = self.ResUNet_model()

        model.compile(optimizer=Adam(lr=self.lr),  # 1e-5
                         loss=DiceBCELoss,
                         metrics=['accuracy', dice_coef, Recall(), Precision()])

        model.summary()
        model.fit(train_gen,
                  validation_data=val_gen,
                  steps_per_epoch=steps_per_epoch,  # 2
                  epochs=self.epochs,
                  validation_steps=val_steps_per_epoch,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint('resunet_brain_mri_seg.hdf5', verbose=1, monitor='val_loss',
                                                         mode='min', save_best_only=True)],
                  shuffle=True)
        return model
