import configparser
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
# from tensorflow.keras.losses import MS_SSIM
from helpers import *


def autoencoder_model(in_shape):
    inputs = layers.Input(shape=in_shape)
    # normalized_inputs = layers.Lambda(lambda x: (x / 127.5) - 1.0)(input)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # normalized_decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    # decoded = layers.Lambda(lambda x: (x * 0.5) + 0.5)(normalized_decoded)

    # Autoencoder
    return Model(inputs, x)


def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


if __name__ == '__main__':
    # # Limit VRAM Usage
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # USE GPU
    with tf.device('/GPU:0'):

        # Specifying Directories
        # Read config.ini file
        config_obj = configparser.ConfigParser()
        config_obj.read("D:\Coding Projects\Pycharm Projects\DeblurCNN\configfile.ini")
        dbparam = config_obj["ds"]

        data_dir = dbparam['dataset_path']
        train_dir = data_dir + '/dummytrain'
        test_dir = data_dir + '/dummytest'

        # Load data
        input_shape = (360, 100, 3)

        trainblur, trainsharp = load_data_from_dir(train_dir, input_shape)
        testblur, testsharp = load_data_from_dir(test_dir, input_shape)

        # display_images(trainblur, trainsharp)
        # display_images(testblur, testsharp)

        print("TRAIN data-array:", trainblur.shape, trainsharp.shape)
        print("TEST data-array:", testblur.shape, testsharp.shape)

        # AUTO ENCODER
        autoencoder = autoencoder_model((100, 360, 3))

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(0.001)
        autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=[ssim_loss, 'accuracy'])
        autoencoder.summary()

        # Train
        autoencoder.fit(
            x=trainblur,
            y=trainsharp,
            epochs=1,
            batch_size=16,
            shuffle=True,
            validation_data=(testblur, testsharp),
        )

        # Test set prediction
        predictions = autoencoder.predict(testblur)
        display(testblur, predictions, n=5)  # Display n results from test set
