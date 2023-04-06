import sys
import os
import subprocess

# install pip packages from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])# install pip packages if not already installed

""" 
--------------------------------------------------------------------------------------------------------------------------------------------------
Imports
--------------------------------------------------------------------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
from functions import *
from preprocess_ import *
import shutil


# !nvidia-smi -L 

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

# multi GPU strategy
strategy = tf.distribute.MirroredStrategy() # you can define which gpus to use tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16') # On TPUs, use 'mixed_bfloat16'

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



"""
--------------------------------------------------------------------------------------------------------------------------------------------------
Set up parameters
--------------------------------------------------------------------------------------------------------------------------------------------------
"""


data_dir = 'Dataset/'
sav_dir = os.getcwd()+'/'
# print(sav_dir)
num_runs = 20

batch_size_per_replica = 16
img_height = 1024
img_width = 1024

buffer_size = 8000
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

model_choice = 0 #for custom model[0], Resnet50[1], ConvNext[2], EfficientNetv2[3]
filters = 16
epochs = 400

# delete Model folder
try:
    shutil.rmtree(sav_dir+'Models')
except:
    pass


# FOR IMAGE PREPROCESSING
base_folder = 'Data/'
csv_file = base_folder+'A08241_internals.csv'
saving_dir = base_folder+'Dataset/'
defect = 'hollow_heart' # 'tuber','sprouting','hollow_heart','IBS','bruise','cracks','stemend_browning','anthocyanin','vascular_discoloration','greening','unknow_defect'
others = 'others'
image_folder = base_folder+'tubers'
target_num = 1000 # rough target number of images
split = 0.8 # train/test split



"""
--------------------------------------------------------------------------------------------------------------------------------------------------
Training
--------------------------------------------------------------------------------------------------------------------------------------------------
"""

# loop over all runs
for run in range(num_runs):
    
    seed = np.random.randint(0, 1000)# set seed
    print('Training Seed: ', seed)
    
    # preprocess the data
    preprocess_data(csv_file=csv_file,
                  saving_dir=saving_dir,
                  defect=defect,
                  others=others,
                  image_folder=image_folder,
                  target_num=target_num,
                  split=split)
                  
    
    Train_dir = data_dir+'Dataset/train'
    Test_dir = data_dir+'Dataset/test'
    
    Train_dir = pathlib.Path(Train_dir)
    Test_dir = pathlib.Path(Test_dir)
    
    # print number of images in each folder
    image_count = len(list(Train_dir.glob('*/*.jpg')))
    print('Number of images in Train folder: ', image_count)
    image_count = len(list(Test_dir.glob('*/*.jpg')))
    print('Number of images in Test folder: ', image_count)
    
    # load data
    train_ds, val_ds, test_ds, class_names = get_data_sets(Train_dir=Train_dir,
                                                           Test_dir=Test_dir,
                                                           Seed=seed,
                                                           Batch_size=batch_size,
                                                           Buffer=buffer_size,
                                                           IMG_HEIGHT=img_height,
                                                           IMG_WIDTH=img_width)
    
    print(class_names)

    # visualize some data
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(3, 2, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
    plt.show()
            

    num_classes = len(class_names)
    
    # load the model
    model, model_folder = my_model(choice=model_choice,
                                    img_height=img_height,
                                    img_width=img_width,
                                    strategy=strategy,
                                    num_classes=num_classes,
                                    filters=filters,
                                    sav_dir=sav_dir,
                                    run=run)
    
    # train the model
    model = train_net(model=model,
                      model_folder=model_folder,
                      train_ds=train_ds,
                      val_ds=val_ds,
                      test_ds=test_ds,
                      epochs=epochs,
                      strategy=strategy)

print('Training Complete')