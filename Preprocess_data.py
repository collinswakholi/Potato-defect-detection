from genericpath import exists
from math import floor
from msilib.schema import Directory
import os
import numpy as np
import cv2
import pandas as pd
import random
import albumentations as A

save_others = True # save all other images in one folder

# create directories
csv_file = 'Data/A08241_internals.csv'
saving_dir = 'Dataset'
defect = 'hollow_heart' # 'tuber','sprouting','hollow_heart','IBS','bruise','cracks','stemend_browning','anthocyanin','vascular_discoloration','greening','unknow_defect'
others = 'other_potatoes'
image_folder = 'Data/tubers'
target_num = 1000 # rough target number of images

try:
    os.makedirs(os.path.join(saving_dir,defect))
    if save_others:
        os.makedirs(os.path.join(saving_dir,others))
except:
    print('folder(s) already exists...')
    # os.rmdir(os.path.join(saving_dir,defect))
    # os.rmdir(os.path.join(saving_dir,others))

# Functions

# define augmentaiton transforms 
transform = A.Compose([
    A.Flip(),
    A.Affine(),
    A.RandomBrightnessContrast(p=.2),
    A.GaussNoise(p=0.2),
    A.GridDistortion(),
    A.ShiftScaleRotate(p=0.5),
    A.Downscale(p=.01),
])
random.seed(42)

def load_image (folder, name):
    try:
        img = cv2.imread(os.path.join(folder,name))
    except:
        print('Could NOT find image!!!')
    return img

def prewiew_random(folder, name, nn):
    """
    Preview nn random images
    """
    leng = name.size
    ids = random.sample(range(leng),nn)
    print(ids)

    for i in range(nn):
        Name = name[0,ids[i]]
        img = load_image(folder,Name)
        cv2.namedWindow(Name, cv2.WINDOW_NORMAL)
        cv2.imshow(Name,img)
        cv2.waitKey(0)

def write_images(img, file, sav_str, transform, num):
    # write orignal image
    cv2.imwrite((sav_str+'/Aug_0_'+file), img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(num):
        # random.seed(42)
        img2 = transform(image = img)["image"]
        img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite((sav_str+'/Aug_' + str(i+1) + '_' + file), img3)

# read csv file (find which ones have defects, specify which defect)
df = pd.read_csv(csv_file, sep=',')
colss = np.array(df.columns)
values = df.values

# find indices of defect potatoes
col_idx = np.where(colss == defect)
def_values = np.array(values[:,col_idx],dtype=int)
dd = def_values.reshape(-1)
idx_def = np.array(np.where(dd==1))

if save_others:
    idx_not = np.array(np.where(dd!=1))

len_ = len(df)
def_count = idx_def.size
num = floor(target_num/def_count)-1

print(def_count, defect,'potatoes(',round(100.0*def_count/len_,2),'% of total dataset) found')

all_image_names = values[:,1]

defect_image_names = all_image_names[idx_def]

if save_others:
    other_image_names = all_image_names[idx_not]

prewiew_random(image_folder,defect_image_names,5) # preview random hollow heart samples

# read defect potato images, augment, and save in respective Directory
for file in defect_image_names[0]:
    img = load_image(image_folder,file) #load image
    if img is None:
        print('"'+ file + '" was not found')
    else:
        sav_str = os.path.join(saving_dir,defect)
        write_images(img, file, sav_str, transform, num)

if save_others:
    for file in other_image_names[0]:
        img = load_image(image_folder,file) #load image
        if img is None:
            print('"'+ file + '" was not found')
        else:
            sav_str = os.path.join(saving_dir,others,file)
            cv2.imwrite(sav_str, img)

print('Done...')