import os
import collections
from shutil import copy2
import configparser

#Read config.ini file
config_obj = configparser.ConfigParser()
config_obj.read("D:\Coding Projects\Pycharm Projects\DeblurCNN\configfile.ini")
dbparam = config_obj["ds"]

# Data dir
dataset_path = dbparam['dataset_path']

# Helper method to split dataset into train and test folders
def copy_images_from_list(src_folder, dest_folder, txt_file):
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Read the .txt file
    with open(txt_file, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Remove leading/trailing whitespaces and newline characters
            image_name = line.strip()

            # Construct source and destination paths
            src_path = os.path.join(src_folder, image_name+'.jpg')
            dest_path = os.path.join(dest_folder, image_name+'.jpg')

            # Copy the image from source to destination
            copy2(src_path, dest_path)


# Prepare train dataset by copying images from blurred to train/blurred using the file train.txt

print("Creating train data...")
copy_images_from_list(dataset_path+'/blurred', dataset_path+'/train/blurred', dataset_path+'/meta/train.txt')
copy_images_from_list(dataset_path+'/sharpened', dataset_path+'/train/sharpened', dataset_path+'/meta/train.txt')
print("Success")

# Prepare train dataset by copying images from blurred to test/blurred using the file test.txt

print("Creating train data...")
copy_images_from_list(dataset_path+'/blurred', dataset_path+'/test/blurred', dataset_path+'/meta/test.txt')
copy_images_from_list(dataset_path+'/sharpened', dataset_path+'/test/sharpened', dataset_path+'/meta/test.txt')
print("Success")
