import cv2
import imghdr
import os
import configparser

#Read config.ini file
config_obj = configparser.ConfigParser()
config_obj.read("D:\Coding Projects\Pycharm Projects\DeblurCNN\configfile.ini")
dbparam = config_obj["ds"]

# Data dir
dataset_path = dbparam['dataset_path']


# Removing images less than 9 KB, with weird extensions and not-readable
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image in os.listdir(os.path.join(dataset_path,'blurred')):
    blurred_image_path = os.path.join(dataset_path, 'blurred', image)
    sharpened_image_path = os.path.join(dataset_path, 'sharpened', image)
    # Removing images less than 9 KB
    file_stats = os.stat(blurred_image_path)
    if file_stats.st_size <= 9000:
        os.remove(blurred_image_path)
        os.remove(sharpened_image_path)
    # Making sure its readable and has one of the supported extensions
    try:
        img = cv2.imread(blurred_image_path)
        tip = imghdr.what(blurred_image_path)
        if tip not in image_exts:
            print('Image not in ext list {}'.format(blurred_image_path))
            os.remove(blurred_image_path)
            os.remove(sharpened_image_path)
    except Exception as e:
        print('Issue with image {}'.format(blurred_image_path))