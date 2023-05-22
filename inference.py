import tensorflow as tf
from pipeline import autoencoder_model
from pipeline import MAE_SSIM, ssim
from PIL import Image
import numpy as np
import configparser
from helpers import *


def load_img(pth):
    return Image.open(pth)


def preprocess_img(img):
    img = img.resize((360, 100, 3)[:2])  # Resize image to the desired input shape
    image_array = np.array(img)
    # [-1,-1] Normalizing
    img_normalized = (image_array / 127.5) - 1.0
    # Handling .png 4th channel
    if len(img_normalized.shape) > 2 and img_normalized.shape[2] == 4:
        # slice off the alpha channel
        img_normalized = img_normalized[:, :, :3]
    return img_normalized[np.newaxis, :, :]


if __name__ == '__main__':
    # Specifying Directories
    # Read config.ini file
    config_obj = configparser.ConfigParser()
    config_obj.read("D:\Coding Projects\Pycharm Projects\DeblurCNN\configfile.ini")
    dbparam = config_obj["ds"]
    ckpt = config_obj["ckpt"]

    data_dir = dbparam['dataset_path']
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    ckpt_dir = ckpt["weights_path"]
    model_dir = ckpt["model_path"]

    # Load model
    model = autoencoder_model((100, 360, 3))

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss=MAE_SSIM, metrics=[ssim, 'accuracy'])

    # Load Weights
    model.load_weights(ckpt_dir)

    # Load and Preprocess images
    img = load_img('demo/1.png')
    img = preprocess_img(img)

    # Inference
    predictions = model.predict(img)

    # Display Image
    print(img.shape)
    display(img, predictions, n=1)


