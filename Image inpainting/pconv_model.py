from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, BatchNormalization, Activation, UpSampling2D, Concatenate, ReLU, LeakyReLU, Conv2D
from inpainting_loss import get_vgg16_weights, StyleModel
from inpainting_train import fit
from pconv_layer import PConv2D
import os
import tensorflow as tf
class pconv_model():
    def __init__(self,args):
        self.train_bn=args.train_bn
        self.IMAGE_SIZE=args.IMAGE_SIZE
        self.pretrained_model_path=args.pretrained_model_path
        self.batch_size=args.batch_size
        self.epochs=args.epochs
        self.steps_per_epoch=args.steps_per_epoch
        self.validation_steps=args.validation_steps
        self.lr=args.lr
        self.checkpoint_path=args.checkpoint_path
    def build_pconv_unet(self,train_bn=True):
        """
        this function defines the architecture of the inpainting model
        """
        # INPUTS
        inputs_img = Input((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='inputs_img')
        inputs_mask = Input((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='inputs_mask')

        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN' + str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask

        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)  ###
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
        e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3)

        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2, 2))(img_in)
            up_mask = UpSampling2D(size=(2, 2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv, up_img])
            concat_mask = Concatenate(axis=3)([e_mask, up_mask])
            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask

        d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
        d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
        outputs = Conv2D(3, 1, activation='sigmoid', name='outputs_img')(d_conv16)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

        return model, inputs_mask

    def run_model(self,train_dataset,val_dataset):
        """
        this function builds and train the model
        :param train_dataset: training dataset
        :param val_dataset: validation dataset
        :return: the trained model and its history
        """
        weights_dir = r'VGG_weights'  ##
        if os.path.exists(weights_dir) == False:
            os.makedirs(weights_dir)
        get_vgg16_weights(weights_dir)
        vgg16_weights =weights_dir+'/vgg16_pytorch2keras.h5'   ##
        vgg16 = StyleModel(weights=vgg16_weights)

        model=None
        if self.pretrained_model_path==None:
            model = self.build_pconv_unet(train_bn=self.train_bn)[0]
            model.summary()
        else:
            model = self.build_pconv_unet(train_bn=self.train_bn)[0]
            model.load_weights(self.pretrained_model_path)
            model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # 0.0002

        history = fit(model=model,
                         input_data=iter(train_dataset),
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         steps_per_epoch=self.steps_per_epoch,
                         validation_data=iter(val_dataset),
                         validation_steps=self.validation_steps,
                         vgg16=vgg16,
                         optimizer=optimizer,
                         save_dir=self.checkpoint_path)
        return history,model