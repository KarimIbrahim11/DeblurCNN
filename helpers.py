import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def display_images(array1, array2):
    # Select two images from the arrays
    image1 = array1[0]
    image2 = array2[0]

    # Plot the first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")

    # Plot the second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("Image 2")
    plt.axis("off")

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


def load_images_as_arrays(folder_path, input_shape):
  image_list = os.listdir(folder_path)
  num_images = len(image_list)

  images = np.empty((num_images, *input_shape), dtype=np.float32)
  images = images.transpose(0,2,1,3)
  # print(images[0].shape)

  for i, image_name in enumerate(image_list):
    image_path = os.path.join(folder_path, image_name)
    image = Image.open(image_path)
    # print("Loaded image size:", image.size)
    image = image.resize(input_shape[:2])  # Resize image to the desired input shape
    # print("Resized image size:", image.size)
    image_array = np.array(image)
    # Display the image using matplotlib
    normalized_image = (image_array / 127.5) - 1.0
    images[i] = normalized_image

  return images


def load_data_from_dir(dir, input_shape):
  blur = load_images_as_arrays(dir+'/blurred', input_shape)
  sharp = load_images_as_arrays(dir+'/sharpened', input_shape)
  return blur, sharp


def display(array1, array2, n):
    """
    Displays n random images from each one of the supplied arrays.
    """


    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(100,360,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image1.reshape(100,360,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()