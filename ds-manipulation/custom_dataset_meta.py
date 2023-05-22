from os import listdir
from os import path
from os import mkdir
from os.path import isfile, join
import configparser

#Read config.ini file
config_obj = configparser.ConfigParser()
config_obj.read("D:\Coding Projects\Pycharm Projects\DeblurCNN\configfile.ini")
dbparam = config_obj["ds"]

dataset_path = dbparam['dataset_path']

# Create meta folder
metapath = dataset_path + "/meta"
if not path.isdir(metapath):
    mkdir(metapath)


# Create train.txt [0.75 of the class] and test.xt [0.25 of the class]
train = open(metapath + '/train.txt', 'w')
test = open(metapath + '/test.txt', 'w')

class_images_count = len(listdir(join(dataset_path , "blurred")))
count = 1
for imagename in listdir(dataset_path + "/blurred"):
    if count <= 0.8*class_images_count:
        train.write(str(imagename.replace(".jpg", "")) + "\n")
    else:
        test.write(str(imagename.replace(".jpg", "")) + "\n")
    count = count + 1

train.close()
test.close()