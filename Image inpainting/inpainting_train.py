import os
from datetime import datetime
import tensorflow as tf
import numpy as np

from inpainting_preprocess import create_input_dataset
from inpainting_loss import loss_total


from tensorflow._api.v2.image import ssim


def calculate_psnr(HR,SR):
  """
  Calculates the peak signal-to-noise ratio (PSNR) between the HR and the reconstructed image
  :param HR: HR image
  :param SR: Reconstructed image
  :return:psnr value
  """
  HR=tf.convert_to_tensor(HR,dtype=tf.float32)
  SR=tf.convert_to_tensor(SR,dtype=tf.float32)
  psnr=tf.image.psnr(HR,SR,max_val=1)
  return tf.math.reduce_mean(psnr,axis=None,keepdims=False,name=None)

def calculate_ssim(HR,SR):
  """
  Calculates the structural similarity index measure (SSIM) between the HR and the reconstructed image
  :param HR: HR image
  :param SR: Reconstructed image
  :return:ssim value
  """
  HR=tf.convert_to_tensor(HR,dtype=tf.float32)
  SR=tf.convert_to_tensor(SR,dtype=tf.float32)
  ssim=tf.image.ssim(HR,SR,max_val=1)
  return tf.math.reduce_mean(ssim,axis=None,keepdims=False,name=None)



@tf.function
def train_step(model, example, batch_size, vgg16, optimizer):
    """
    this function takes the input and calculates the total loss and accordingly update the model weights using the given optimizer
    :param model: the inpainting model
    :param example: training examples
    :param batch_size: number of mini-batch
    :param vgg16: it is used for calculating the perceptual and style loss
    :param optimizer: optimizer to adapt the network weights
    :return: the inputs, targets and the loss
    """
    inputs, targets = create_input_dataset(example, batch_size)

    with tf.GradientTape() as tape:
        #output = model(inputs, training=True)
        loss = loss_total(model, inputs, targets, vgg16, training=True)

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    #return loss
    return inputs, targets, loss  ## added 22-2-2023


@tf.function
def val_step(model, example, batch_size, vgg16):
    """
    this function takes the validation input and calculates the total loss
    :param model: the inpainting model
    :param example: training examples
    :param batch_size: number of mini-batch
    :param vgg16: it is used for calculating the perceptual and style loss
    :return: the inputs, targets and the loss
    """
    inputs, targets = create_input_dataset(example, batch_size)
    loss = loss_total(model, inputs, targets, vgg16, training=False)
    #return loss
    return inputs, targets, loss ## added 22-2-2023


def fit(model, input_data, batch_size, epochs, validation_data,
        steps_per_epoch, validation_steps, vgg16, optimizer, save_dir):
    """
    fits the model and prints the loss,PSNR and SSIM each epoch on the training and validation steps. In addition, the model weights are saved each epoch
    :param model: inpainting model
    :param input_data: training data
    :param batch_size: number of mini-batches
    :param epochs: number of epochs
    :param validation_data: validation data
    :param steps_per_epoch: number of training steps per epoch
    :param validation_steps: number of validation steps per epoch
    :param vgg16: it is used for calculating the perceptual and style loss
    :param optimizer: optimizer to adapt the network weights
    :param save_dir: the given path, where the model is saved
    :return: training and validation losses
    """

    train_loss_results = []
    val_loss_results = []
    psnr_train_results = []
    psnr_val_results = []
    ssim_train_results = []
    ssim_val_results = []

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch, ))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        psnr_avg=tf.keras.metrics.Mean()
        psnr_val_avg=tf.keras.metrics.Mean()
        ssim_avg=tf.keras.metrics.Mean()
        ssim_val_avg=tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            example = next(input_data)
            inputs, targets, loss_train = train_step(model, example, batch_size, vgg16,
                                    optimizer)

            reconstructed_imgs = model.predict([inputs[0], inputs[1]])
            psnr_train = calculate_psnr(targets, reconstructed_imgs)
            ssim_train = calculate_ssim(targets, reconstructed_imgs)

            # Update training metric.
            epoch_loss_avg.update_state(loss_train)
            psnr_avg.update_state(psnr_train)
            ssim_avg.update_state(ssim_train)
        # Log every epoch.
        train_loss_results.append(epoch_loss_avg.result().numpy())
        psnr_train_results.append(psnr_avg.result().numpy())
        ssim_train_results.append(ssim_avg.result().numpy())

        # Run a validation loop at the end of each epoch.
        for step_val in range(validation_steps):
            example_val = next(validation_data)
            inputs, targets,loss_val = val_step(model, example_val, batch_size, vgg16)
            reconstructed_val_imgs = model.predict([inputs[0], inputs[1]])
            psnr_val = calculate_psnr(targets, reconstructed_val_imgs)
            ssim_val = calculate_ssim(targets, reconstructed_val_imgs)

            epoch_val_loss_avg.update_state(loss_val)
            psnr_val_avg.update_state(psnr_val)
            ssim_val_avg.update_state(ssim_val)

        # Log every epoch.
        val_loss_results.append(epoch_val_loss_avg.result().numpy())
        psnr_val_results.append(psnr_val_avg.result().numpy())
        ssim_val_results.append(ssim_val_avg.result().numpy())

        print('''Training loss: %.4f, Validation loss: %.4f,Training PSNR: %.4f,Validation PSNR: %.4f 
                ,Training SSIM: %.4f, Validation SSIM: %.4f''' % (epoch_loss_avg.result(), epoch_val_loss_avg.result()
                                                                              , psnr_avg.result(),
                                                                psnr_val_avg.result(), ssim_avg.result(),
                                                                ssim_val_avg.result()))  ## added 22-2-2023

        # Reset metrics at the end of each epoch
        epoch_loss_avg.reset_states()
        epoch_val_loss_avg.reset_states()
        psnr_avg.reset_states()
        psnr_val_avg.reset_states()

    # Save weights file.
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_name = save_dir + '/epoch-' + str(
        epoch) + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.h5'
    model.save_weights(checkpoint_name, overwrite=True, save_format='h5')

    print(psnr_val_results)
    print(ssim_val_results)
    print(psnr_train_results)
    print(ssim_train_results)

    return {'loss': train_loss_results, 'val_loss': val_loss_results}