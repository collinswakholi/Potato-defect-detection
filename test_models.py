""" 
------------------------------------------------------------------------------------------------------------------------
Imports
------------------------------------------------------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import time

import pandas as pd

from sklearn.metrics import confusion_matrix
import itertools
import random

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Define the parameters
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Folder = 'Custom_512' # Path to the folder containing the subfolders with the n models
t_data_dir = 'Dataset/test' # Path to test dataset
size = 512


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#load test dataset
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
img_height = size
img_width = size
test_dataset = tf.keras.utils.image_dataset_from_directory(
    t_data_dir,
    validation_split=None,
    labels='inferred',
    label_mode='int',
    seed=123,
    color_mode = 'rgb',
    interpolation='bilinear',
    image_size=(img_height, img_width))



n_test = len(test_dataset.file_paths)
print("Test dataset = ",n_test,"images")

columns=['Run','Model','Model overall accuracy','Model loss','Number of images','Correct predictions','Incorrect predictions','Correct hollow',
         'Correct other','Incorrect hollow','Incorrect other','Correct (%)','Incorrect (%)','Average inference time per image (ms)']
Results = pd.DataFrame(columns=columns)


# check if the folder exists
if os.path.exists(Folder):
    # read all the subfolders in the folder
    subfolders = [f.path for f in os.scandir(Folder) if f.is_dir()]
    
    #remove the .ipynb_checkpoints folder
    subfolders = [x for x in subfolders if '.ipynb_checkpoints' not in x]
    subfolders.sort()
    for subfolder in subfolders:
        if not os.path.exists(subfolder + '/my_model.h5'):
            continue
        # do the rest of the code
        split = subfolder.split('run_')
        run_number = int(split[1])
        # print(run_number)
        model=tf.keras.models.load_model(subfolder+'/my_model.h5')
        checkpoint_path = subfolder+"/cp.ckpt"
        model.load_weights(checkpoint_path)

        model.summary() # Preview model architecture

        """
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Evaluate model performance on the test set
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        """
        
        loss, acc = model.evaluate(test_dataset, verbose=2)
        print("Model, accuracy: {:5.2f}%".format(100 * acc))

        class_names = test_dataset.class_names
        plt.figure(figsize=(10, 10))
        rand_idx = random.sample(range(n_test),6) #preview 6 random images
        for i in range(6):
            file_path = test_dataset.file_paths[rand_idx[i]]
            img = tf.keras.utils.load_img(
                file_path, target_size=(img_height, img_width)
            )

            imgg = np.array(img, dtype = "uint8")

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])


            ax = plt.subplot(3, 2, i + 1)
            plt.imshow(imgg)
            plt.title(class_names[np.argmax(score)]+' : ' + str(round(100 * np.max(score),2))+ '% confidence')
            plt.axis("off")
        plt.savefig(subfolder+'/predictions.png')
        plt.show()
        
        """
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get the predictions for each image in the test dataset
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        """
        # time overall inference time
        tic = time.perf_counter()
        pppp = model.predict(test_dataset)
        toc = time.perf_counter()
        inf_time_per_img = (toc-tic)/n_test
        print("Total inference time = ",toc-tic,"seconds")
        print("Inference time per image = ",inf_time_per_img,"seconds")

        y_pred = []
        y_real = []
        scores =[]
        classes = []
        img_names = []
        Inf_time = []
        
        for i in range(n_test):
            img_name = test_dataset.file_paths[i].split('/')[-1]
            img = tf.keras.utils.load_img(
                test_dataset.file_paths[i], target_size=(img_height, img_width)
            )
            # check if path contains 'hollow_heart'
            if 'hollow_heart' in test_dataset.file_paths[i]:
                test_label = 0
                class_ = 'hollow_hearts'
            else:
                test_label = 1
                class_ = 'others'
            
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch
            tic = time.perf_counter()
            predictions = model.predict(img_array)
            toc = time.perf_counter()
            el_time = toc-tic
            Inf_time.append(1000*el_time)
            
            pred = np.argmax(predictions[0])
            score = tf.nn.softmax(predictions[0])
            
            scores.append(round(100 * np.max(score),2))
            y_pred.append(pred)
            y_real.append(test_label)
            classes.append(class_)
            img_names.append(img_name)
            
        # create a dataframe with the predictions and the real labels
        df = pd.DataFrame({'Img_name':img_names,'Class':classes,'Real':y_real,'Pred value':y_pred,'Score (%)':scores,'Inference time (ms)':Inf_time})
        df.to_csv(subfolder+'/Predictions.csv',index=False)

        # compute the confusion matrix
        cm = confusion_matrix(y_real, y_pred)
        # print('Confusion matrix',cm)


        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names, normalize=True,
                                title='Normalized confusion matrix')
        plt.savefig(subfolder+'/confusion_matrix.png')
        plt.show()

        y_pred = np.array(y_pred)
        y_real = np.array(y_real)

        # get number of correct predictions
        correct = np.sum(y_pred == y_real)
        correct_hollow = np.sum((y_pred == y_real) & (y_real == 0))
        correct_other = np.sum((y_pred == y_real) & (y_real == 1))
        incorrect = np.sum(y_pred != y_real)
        incorrect_hollow = np.sum((y_pred != y_real) & (y_real == 0))
        incorrect_other = np.sum((y_pred != y_real) & (y_real == 1))

        perc_correct = round(100*correct/n_test,2)
        perc_incocorrect = round(100*incorrect/n_test,2)
        # print(perc_correct, '% correct')
        # print(perc_incocorrect, '% incorrect')

        df_perf = pd.DataFrame({'Run':run_number,
                                'Model':subfolder,
                                'Model overall accuracy':acc,
                                'Model loss':loss,
                                'Number of images':n_test,
                                'Correct predictions':correct,
                                'Incorrect predictions':incorrect,
                                'Correct hollow':correct_hollow,
                                'Correct other':correct_other,
                                'Incorrect hollow':incorrect_hollow,
                                'Incorrect other':incorrect_other,
                                'Correct (%)':perc_correct,
                                'Incorrect (%)':perc_incocorrect,
                                'Average inference time per image (ms)':1000*inf_time_per_img},index=[0])
        save_path = subfolder+'/Performance.csv'
        df_perf.to_csv(save_path,index=False)
        
        # vertical concatenate all the performance dataframes
        Results = pd.concat([Results,df_perf],axis=0)
    Results.to_csv(Folder+'\Results.csv',index=False)