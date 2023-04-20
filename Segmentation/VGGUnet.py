import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from keras.metrics import Recall, Precision
from keras.optimizers import Adam
from segmentation_utils import *
class VGGUnet():
    def __init__(self,args):
        self.image_size=args.image_size
        self.lr=args.lr
        self.epochs=args.epochs
        self.pretrained_weights = args.pretrained_weights


    def VGG16_encoder(self, weights="imagenet"):
        """
        this function creates the encoder of the VGGUNET model
        :param weights: the encoder pretrained weights which is imagenet by default
        :return: residual layers (encoder layers to be concatenated with their corresponding decoder layers), output layer and the encoder
        """
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

        img_input = Input(shape=(self.image_size, self.image_size, 3))
        # Determine proper input shape

        # Block 1
        x = Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv1"
        )(img_input)
        x = Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv2"
        )(x)
        b1 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv1"
        )(x)
        x = Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv2"
        )(x)
        b2 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
        # Block 3
        x = Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv1"
        )(x)
        x = Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv2"
        )(x)
        x = Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv3"
        )(x)
        b3 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv1"
        )(x)
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv2"
        )(x)
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv3"
        )(x)
        b4 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv1"
        )(x)
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv2"
        )(x)
        x = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv3"
        )(x)
        out = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        # model=Model(img_input, x, name="vgg16_encoder")
        if weights == "imagenet":
            VGG_Weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
            # Model(img_input, x).load_weights(VGG_Weights_path)
            model = Model(img_input, x)
            model.load_weights(VGG_Weights_path)
        return [b1, b2, b3, b4], out, model

    def conv_block(self,input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def VGGUNET(self):
        """
        this function creates the VGGUNET model and loads the pretrained weights if exists
        """
        # img_input = Input(shape=(img_height, img_width, 3))
        encoder_layers, out, encoder_model = self.VGG16_encoder()
        filters = [64, 128, 256, 512]
        for i in range(1, len(encoder_layers) + 1):
            if i == 1:
                x = Conv2DTranspose(filters[-i], (2, 2), strides=2, padding="same")(out)
                x = Concatenate()([x, encoder_layers[-i]])
                x = self.conv_block(x, filters[-i])
            else:
                x = Conv2DTranspose(filters[-i], (2, 2), strides=2, padding="same")(x)
                x = Concatenate()([x, encoder_layers[-i]])
                x = self.conv_block(x, filters[-i])
        output = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

        model = Model(encoder_model.inputs, output, name="VGGUNet")
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
        model = self.VGGUNET()

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
                      tf.keras.callbacks.ModelCheckpoint('vggunet_brain_mri_seg.hdf5', verbose=1, monitor='val_loss',
                                                         mode='min', save_best_only=True)],
                  shuffle=True)
        return model
