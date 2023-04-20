import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from segmentation_utils import *
class UNet():
    def __init__(self,args):
        self.image_size=args.image_size
        self.lr=args.lr
        self.epochs=args.epochs
        self.pretrained_weights=args.pretrained_weights
    def unet_model(self):
        """
        this function creates the Unet model and loads the pretrained weights if exists
        """
        inputs = Input(shape=(self.image_size, self.image_size, 3))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(drop4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
        model = Model(inputs, conv10)
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
        model = self.unet_model()

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
                      tf.keras.callbacks.ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, monitor='val_loss',
                                                         mode='min', save_best_only=True)],
                  shuffle=True)
        return model