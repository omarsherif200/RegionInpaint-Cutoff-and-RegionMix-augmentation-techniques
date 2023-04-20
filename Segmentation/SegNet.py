import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D,Conv2DTranspose, Concatenate, Input,Dense,Reshape
from tensorflow.keras.models import Model
from keras.metrics import Recall, Precision
from keras.optimizers import Adam,SGD
from segmentation_utils import *

class SegNet():
    def __init__(self,args):
        self.image_size = args.image_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.pretrained_weights = args.pretrained_weights

    def SegNet_model(self):
        """ this function creates the SegNet model and loads the pretrained weights if exists """
        # Encoding layer
        img_input = Input(shape=(self.image_size, self.image_size, 3))
        x = Conv2D(64, (3, 3), padding='same', name='conv1', strides=(1, 1))(img_input)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
        x = BatchNormalization(name='bn5')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
        x = BatchNormalization(name='bn8')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
        x = BatchNormalization(name='bn9')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
        x = BatchNormalization(name='bn10')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
        x = BatchNormalization(name='bn11')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
        x = BatchNormalization(name='bn12')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
        x = BatchNormalization(name='bn13')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        # Decoding Layer 
        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
        x = BatchNormalization(name='bn14')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
        x = BatchNormalization(name='bn15')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
        x = BatchNormalization(name='bn16')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
        x = BatchNormalization(name='bn17')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
        x = BatchNormalization(name='bn18')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
        x = BatchNormalization(name='bn19')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
        x = BatchNormalization(name='bn20')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
        x = BatchNormalization(name='bn21')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
        x = BatchNormalization(name='bn22')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
        x = BatchNormalization(name='bn23')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
        x = BatchNormalization(name='bn24')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
        x = BatchNormalization(name='bn25')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
        x = BatchNormalization(name='bn26')(x)
        x = Activation('sigmoid')(x)
        pred = Reshape((self.image_size, self.image_size))(x)
        model = Model(inputs=img_input, outputs=pred)
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
        model = self.SegNet_model()

        model.compile(optimizer=Adam(lr=self.lr), loss=DiceBCELoss
                      , metrics=[dice_coef, precision, recall, accuracy])

        model.summary()
        model.fit(train_gen,
                  validation_data=val_gen,
                  steps_per_epoch=steps_per_epoch,  # 2
                  epochs=self.epochs,
                  validation_steps=val_steps_per_epoch,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint('segnet_brain_mri_seg.hdf5', verbose=1, monitor='val_loss',
                                                         mode='min', save_best_only=True)],
                  shuffle=True)
        return model