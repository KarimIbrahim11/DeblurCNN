from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
import tensorflow as tf


# Loss functtion
def ssim_loss(y_true, y_pred):
    #tf.train.AdamOptimizer(learning_rate).minimize(-1 * loss_rec)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

# def CNN_model(img_rows, img_cols, channel=3, num_class=None):
#
#     input = Input(shape=(channel, img_rows, img_cols))
#     conv1_7x7= Convolution2D(16,7,7,name='conv1/7x7',activation='relu')(input)#,W_regularizer=l2(0.0002)
#     pool2_2x2= MaxPooling2D(pool_size=(2,2),strides=(1,1),border_mode='valid',name='pool2')(conv1_7x7)
#     poll_flat = Flatten()(pool2_2x2)
#     #MLP
#     fc_1 = Dense(200,name='fc_1',activation='relu')(poll_flat)
#     drop_fc = Dropout(0.5)(fc_1)
#     out = Dense(1,name='fc_2',activation='sigmoid')(drop_fc)
#     # Create model
#     model = Model(input=input, output=out)
#     # Load cnn pre-trained data
#     #model.load_weights('models/weights.h5')#NOTE
#     #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     model.compile(optimizer=adam, loss='mean_absolute_error')
#     return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))







    #
    #
    #
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(32, 32, 1)))
    # model.add(Conv2D(1, kernel_size=(3, 3),
    #                  activation='relu'))
    #
    # model.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss, 'accuracy'])
    #
    # # Train
    # model.fit(np.random.randn(10, 32, 32, 1), np.random.randn(10, 28, 28, 1))

