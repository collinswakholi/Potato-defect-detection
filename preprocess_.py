from genericpath import exists
from math import floor
import glob
import shutil
import os
import numpy as np
import cv2
import pandas as pd
import random
import albumentations as A


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
    # print(ids)

    for i in range(nn):
        Name = name[0,ids[i]]
        img = load_image(folder,Name)
        cv2.namedWindow(Name, cv2.WINDOW_NORMAL)
        cv2.imshow(Name,img)
        cv2.waitKey(0)

def write_images(img, file, sav_str, transform, num):
    # write orignal image
    if num>0:
        cv2.imwrite((sav_str+'/Aug_0_'+file), img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(num):
            # random.seed(42)
            img2 = transform(image = img)["image"]
            img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cv2.imwrite((sav_str+'/Aug_' + str(i+1) + '_' + file), img3)
    else:
        # choose whether to augment or not
        if random.random() > 0.55:
            cv2.imwrite((sav_str+'/'+file), img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = transform(image = img)["image"]
            img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cv2.imwrite((sav_str+'/Aug_0_' + file), img3)

def preprocess_data(csv_file, saving_dir, defect, others, image_folder, target_num=1000, split=0.8):

    # create directories
    # csv_file = 'Data/A08241_internals.csv'
    # saving_dir = 'Dataset'
    # defect = 'hollow_heart' # 'tuber','sprouting','hollow_heart','IBS','bruise','cracks','stemend_browning','anthocyanin','vascular_discoloration','greening','unknow_defect'
    # others = 'others'
    # image_folder = 'Data/tubers'
    # target_num = 1000 # rough target number of images
    # split = 0.8 # train/test split

    shutil.rmtree(saving_dir)
    if os.path.exists(saving_dir):
        # Force remove the directory tree
        shutil.rmtree(saving_dir, ignore_errors=False, onerror=None)
        
    # create saving_dir with all subfolders using os.makedirs
    try:
        os.makedirs(os.path.join(saving_dir,"train",defect))
        os.makedirs(os.path.join(saving_dir,"train",others))
        os.makedirs(os.path.join(saving_dir,"test",defect))
        os.makedirs(os.path.join(saving_dir,"test",others))
    except:
        print('folder(s) already exists...')


    # get a random number for a seed
    rand_num = random.randint(0,1000)
    random.seed(rand_num)
    print('Random seed:',rand_num)



    # read csv file (find which ones have defects, specify which defect)
    df = pd.read_csv(csv_file, sep=',')
    colss = np.array(df.columns)
    values = df.values

    # find indices of defect potatoes
    col_idx = np.where(colss == defect)
    def_values = np.array(values[:,col_idx],dtype=int)
    dd = def_values.reshape(-1)
    idx_def = np.array(np.where(dd==1))

    # if save_others:
    idx_not = np.array(np.where(dd!=1))

    len_ = len(df)
    def_count = idx_def.size
    num = floor(target_num/def_count)-1

    print(def_count, defect,'potatoes(',round(100.0*def_count/len_,2),'% of total dataset) found')

    all_image_names = values[:,1]

    defect_image_names = all_image_names[idx_def]

    # if save_others:
    other_image_names = all_image_names[idx_not]

    # prewiew_random(image_folder,defect_image_names,5) # preview random hollow heart samples

    #  check the images in defect_image_names and other_image_names, remove non-existing images
    defect_image_names = np.array([x for x in defect_image_names[0] if os.path.exists(os.path.join(image_folder,x))])
    other_image_names = np.array([x for x in other_image_names[0] if os.path.exists(os.path.join(image_folder,x))])


    # shuffle the datasets
    random.seed(rand_num)
    random.shuffle(defect_image_names)
    random.shuffle(other_image_names)

    # split the dataset(defect and other) into train and test
    n_images_defect = len(defect_image_names)
    n_images_other = len(other_image_names)

    n_train_defect = floor(split*n_images_defect)
    n_train_other = floor(split*n_images_other)

    train_defect = defect_image_names[:n_train_defect]
    train_other = other_image_names[:n_train_other]

    test_defect = defect_image_names[n_train_defect+1:]
    test_other = other_image_names[n_train_other+1:]

    print('Number of train images: {0} ({1} hollow hearts, {2} others)'.format(len(train_defect)+len(train_other),len(train_defect),len(train_other)))
    print('Number of test images: {0} ({1} hollow hearts, {2} others)'.format(len(test_defect)+len(test_other),len(test_defect),len(test_other)))

    # load images, augment, and save in respective Directory
    for file in train_defect:
        img = load_image(image_folder,file) #load image
        if img is None:
            print('"'+ file + '" was not found')
        else:
            sav_str = os.path.join(saving_dir,'train',defect)
            write_images(img, file, sav_str, transform, num)

    for file in train_other:
        img = load_image(image_folder,file) #load image
        if img is None:
            print('"'+ file + '" was not found')
        else:
            sav_str = os.path.join(saving_dir,'train',others)
            write_images(img, file, sav_str, transform, 0)
            
    for file in test_defect:
        img = load_image(image_folder,file) #load image
        if img is None:
            print('"'+ file + '" was not found')
        else:
            sav_str = os.path.join(saving_dir,'test',defect)
            write_images(img, file, sav_str, transform, num)

    for file in test_other:
        img = load_image(image_folder,file) #load image
        if img is None:
            print('"'+ file + '" was not found')
        else:
            sav_str = os.path.join(saving_dir,'test',others)
            write_images(img, file, sav_str, transform, 0)

    # print the number of images in each folder
    print('Number of train images: {0} ({1} hollow hearts, {2} others)'.format(
        len(os.listdir(os.path.join(saving_dir,'train',defect)))+len(os.listdir(os.path.join(saving_dir,'train',others))),
        len(os.listdir(os.path.join(saving_dir,'train',defect))),
        len(os.listdir(os.path.join(saving_dir,'train',others)))))

    print('Number of test images: {0} ({1} hollow hearts, {2} others)'.format(
        len(os.listdir(os.path.join(saving_dir,'test',defect)))+len(os.listdir(os.path.join(saving_dir,'test',others))),
        len(os.listdir(os.path.join(saving_dir,'test',defect))),
        len(os.listdir(os.path.join(saving_dir,'test',others)))))

    print('Done...')
    
    return True