
import tensorflow as tf
import numpy as np
import cv2
from random import randint

AUTOTUNE = tf.data.experimental.AUTOTUNE

def process_path(file_path):
    """
    takes the image path and return a jpg image
    :param file_path: image path
    :return: jpg image
    """
    # load the raw data from the file as a string
    raw_data = tf.io.read_file(file_path)
    # decode
    return tf.image.decode_jpeg(raw_data, channels=3)


def normalize(image):
    """
    normalize the input image from 0 to 1
    :param image: input image
    :return: normalized image
    """
    image = tf.cast(image, tf.float32)
    return image / 255


def random_jitter(image,IMAGE_SIZE=256):
    """
    resize the image to the required size and flip the image with a probability of 50% to increase the variety of input samples
    :param image: input image
    :param IMAGE_SIZE: required image size
    :return: the preprocessed image
    """
    image = tf.image.resize(image,
                            size=[IMAGE_SIZE, IMAGE_SIZE],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # random flipping with an probability 50%
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_up_down(image)

    return image


@tf.function
def preprocess_image_train(file_path):
    """
    this function takes the input image and applies all the preprocessing functions that are defined above.
    :param file_path: input file path
    :return: preprocessed image
    """
    image = process_path(file_path)
    image = random_jitter(image)
    image = normalize(image)
    return image

def random_mask(IMAGE_SIZE=256, channels=3):
      """
      takes the required input size and number of channels and returns a random mask of 0's and 1's
      :param IMAGE_SIZE: required input size
      :param channels: required number of channels
      :return: a random binary mask
      """
      random_number=randint(0,1);
      BlackBoxImage=None
      if random_number==0:
        img=np.ones((IMAGE_SIZE,IMAGE_SIZE))
        thres=IMAGE_SIZE/8  # this threshold corresponds that how much you want to leave from the beginning of the image and how much to leave from the end of the image
        while True:
          x1=randint(thres,IMAGE_SIZE-(thres*2))
          y1=randint(thres,IMAGE_SIZE-(thres*2))
          r=randint(15,40)   # 15,40   #50,60
          if x1+r<(IMAGE_SIZE-thres) and y1+r<(IMAGE_SIZE-thres):
            cv2.circle(img,(x1,y1),r,(0,0,0),-1)
            break
        BlackBoxImage=np.stack((img,)*3,axis=-1)
      else:
        img = np.ones((IMAGE_SIZE, IMAGE_SIZE, channels), dtype='float32')

        # Set size scale
        size = int((IMAGE_SIZE + IMAGE_SIZE) * 0.03)
        if IMAGE_SIZE < 64 or IMAGE_SIZE < 64:
            raise Exception("Image shape of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, 10)):
            x1, x2 = randint(1, IMAGE_SIZE), randint(1, IMAGE_SIZE)
            y1, y2 = randint(1, IMAGE_SIZE), randint(1, IMAGE_SIZE)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

        # Draw random circles
        for _ in range(randint(1, 10)):
            x1, y1 = randint(1, IMAGE_SIZE), randint(1, IMAGE_SIZE)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (0, 0, 0), -1)

        # Draw random ellipses
        for _ in range(randint(1, 10)):
            x1, y1 = randint(1, IMAGE_SIZE), randint(1, IMAGE_SIZE)
            s1, s2 = randint(1, IMAGE_SIZE), randint(1, IMAGE_SIZE)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (0, 0, 0), thickness)
        BlackBoxImage=img

      return BlackBoxImage

def create_input_dataset(image_batch, batch_size,IMG_SIZE=256):
    """
    :param image_batch: batch of images according to the batch_size
    :param batch_size: number of mini-batch
    :param IMG_SIZE: required size
    :return: masked images, masks, original ground truth images
    """
    mask_batch = np.stack(
        [random_mask(IMAGE_SIZE=IMG_SIZE) for _ in range(image_batch.shape[0])],   # 
        axis=0)
    print(batch_size)  ###
    # Condition tensor
    bool_mask_batch = tf.convert_to_tensor(mask_batch.copy().astype('bool'),
                                           dtype=tf.bool)

    mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
    masked_batch = tf.where(bool_mask_batch, image_batch, 1.0)

    return (masked_batch, mask_batch), image_batch

def create_input_pipeline_func(dir, batch_size, buffer_size=1000):
    """
    returns batch of training images according to the batch size
    """
    list_ds = tf.data.Dataset.list_files(dir + '/*.jpg')    #/*.jpg   #/**/*.jpg
    dataset = list_ds.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    return dataset.prefetch(buffer_size=AUTOTUNE)

def create_input_pipeline_test(dir, batch_size):
    """
    returns batch of testing images according to the batch size
    """
    list_ds = tf.data.Dataset.list_files(dir + '/*.jpg')
    dataset = list_ds.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size)