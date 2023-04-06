import matplotlib.pyplot as plt
import tensorflow as tf
import os

import numpy as np
import pandas as pd

import random


# Functions

def get_data_sets(Train_dir, Test_dir, Batch_size, IMG_HEIGHT, IMG_WIDTH, Seed, Buffer=1000):
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      Train_dir,
      validation_split=0.3,
      subset="training",
      labels='inferred',
      label_mode='int',
      shuffle=True,
      seed=Seed,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=Batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      Train_dir,
      validation_split=0.3,
      subset="validation",
      labels='inferred',
      label_mode='int',
      shuffle=True,
      seed=Seed,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=Batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      Test_dir,
      seed=Seed,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=Batch_size)

    class_names = train_ds.class_names

    # Configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(Buffer).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def my_model(num_classes, img_height, img_width, strategy, run, choice = 0, filters=16):
    end_Str = str(img_height)+'_run_'+str(run)
    if choice == 0: # Custom model
        model_folder = 'Models/Custom_model_'+end_Str+'/'
    elif choice == 1: # Resnet50V2
        model_folder = 'Models/Resnet50V2_'+end_Str+'/'
    elif choice == 2: # ConvNext
        model_folder = 'Models/ConvNext_'+end_Str+'/'
    elif choice == 3: # EfficientNetv2
        model_folder = 'Models/EfficientNetv2_'+end_Str+'/'

    nf = filters
    shape = (img_height, img_width, 3)

    with strategy.scope():
        
        in_layer = tf.keras.layers.Input(shape = shape)
        top_layer = tf.keras.layers.Rescaling(1./255)(in_layer)
        
        if choice == 0:
            # Custom model
            b_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(nf, 3, padding='same', activation='relu',input_shape=shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(2*nf, 3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(4*nf, 3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(8*nf, 3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(16*nf, 3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32*nf, 3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization()
            ])
        
        # Other prebuilt architectures from KERAS (performance =  https://keras.io/api/applications/)
        elif choice == 1:
            # 1 Resnet50 V2 (https://arxiv.org/abs/1603.05027)
            b_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False,
                                                                weights=None,
                                                                input_tensor=None,
                                                                input_shape=shape)
                
        elif choice == 2:
            # 2 ConvNext-small (https://arxiv.org/abs/2201.03545)
            b_model = tf.keras.applications.convnext.ConvNeXtSmall(weights=None,
                                                                include_top=False,
                                                                input_tensor=None,
                                                                input_shape=shape)
                
        elif choice == 3:
            # 4 EfficientNetv2 (https://arxiv.org/abs/2104.00298)
            b_model = tf.keras.applications.EfficientNetV2B0(weights=None,
                                                            include_top=False,
                                                            input_tensor=None,
                                                            input_shape=shape)
        
        X = b_model(top_layer)
        
        X = tf.keras.layers.GlobalAveragePooling2D()(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        
        X = tf.keras.layers.Dense(1024, activation = 'relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        # X = layers.Dropout(0.2)(X)
        
        X = tf.keras.layers.Dense(128, activation = 'relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        # X = layers.Dropout(0.2)(X)
        
        X = tf.keras.layers.Dense(16, activation = 'relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        # X = layers.Dropout(0.2)(X)
        
        X = tf.keras.layers.Dense(num_classes)(X)
        outputs = tf.keras.layers.Activation('relu', dtype='float32', name='outputs')(X)
        
        
        model = tf.keras.Model(in_layer, outputs)
        
        optimizer = tf.keras.optimizers.Adam(
                        learning_rate = 0.001,
                        beta_1 = 0.90,
                        beta_2 = 0.999,
                        epsilon = 1e-07,
                        amsgrad = True,
                        name='Adam')
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # optimizer.minimize(loss_fn, var_list=var)
        
        model.compile(optimizer=optimizer, 
                    loss=loss_fn,
                    metrics=['accuracy'])

    model.summary()
    #create a folder for the model if it doesn't exist
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    tf.keras.utils.plot_model(model, to_file=model_folder+'model_plot.png', show_shapes=True, show_layer_names=True)
    
    return model, model_folder


def train_net(model, model_folder, train_ds, val_ds, test_ds, epochs, strategy):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # In detail:-
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed

    checkpoint_path = model_folder + "cp.ckpt"

    # Create a callback that saves the model's weights

    with strategy.scope():
        cp_callback = [
            tf.keras.callbacks.TensorBoard(log_dir = model_folder+'logs',
                                        update_freq='epoch'),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_best_only=True,
                                            verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=100,
                                            verbose=1,
                                            restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.95,
                                                patience=10,
                                                verbose=1,
                                                mode='auto',
                                                cooldown=5,
                                                min_lr=0)
        ]

        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = epochs,
            callbacks=[cp_callback],
            verbose = 1
        )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # epochs run by model
    epochs_range = range(len(acc))

    # plt.figure(figsize=(8, 8))
    plt.figure(21)
    # plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.figure(22)
    # plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(model_folder+'training_history.csv')

    # Evaluate the model on the test data
    loss, acc = model.evaluate(test_ds, verbose=2)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
    
    # save loss and accuracy in a csv file using pandas
    test_results = pd.DataFrame({'loss': [loss], 'accuracy': [acc]})
    test_results.to_csv(model_folder+'test_results.csv', index=False)

    # Alternative save whole model
    model.save(model_folder+'my_model.h5')
    
    return model