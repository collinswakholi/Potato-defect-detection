import tensorflow as tf
import os
import pathlib
import shutil

dataset_dir = "./Dataset"
data_dir = pathlib.Path(dataset_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

img_height = 1900
img_width = 1900

Other_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="training",
    labels='inferred',
    label_mode='int',
    seed=123,
    shuffle=True,        #or False
    color_mode = 'rgb',
    interpolation='bilinear',
    image_size=(img_height, img_width))

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    labels='inferred',
    label_mode='int',
    seed=123,
    shuffle=True,        #or False
    color_mode = 'rgb',
    interpolation='bilinear',
    image_size=(img_height, img_width))


    # Save data
sav_dir_Tr = 'Training_Dataset'
sav_dir_T = 'Test_Dataset'
hh_folder = 'hollow_heart'
other_folder = 'other_potatoes'

try:
    os.makedirs(os.path.join(sav_dir_Tr,hh_folder))
    os.makedirs(os.path.join(sav_dir_Tr,other_folder))
except:
    print('folder already exist')

try:
    os.makedirs(os.path.join(sav_dir_T,hh_folder))
    os.makedirs(os.path.join(sav_dir_T,other_folder))
except:
    print('folder already exist')

# Training DS
tr_len = len(Other_dataset.file_paths)
print(tr_len)

for fp in Other_dataset.file_paths:
    # split fn
    base, name = os.path.split(fp)

    #check if base contains hollow_heart or else
    if hh_folder in base:
        new_path = os.path.join(sav_dir_Tr,hh_folder,name)
    else:
        new_path = os.path.join(sav_dir_Tr,other_folder,name)
    
    print(new_path)
    shutil.copy(fp, new_path)
    
# Test DS
t_len = len(test_dataset.file_paths)
print(t_len)

for fh in test_dataset.file_paths:
    # split fn
    base, name = os.path.split(fh)

    #check if base contains hollow_heart or else
    if hh_folder in base:
        new_path = os.path.join(sav_dir_T,hh_folder,name)
    else:
        new_path = os.path.join(sav_dir_T,other_folder,name)
    
    print(new_path)
    shutil.copy(fh, new_path)