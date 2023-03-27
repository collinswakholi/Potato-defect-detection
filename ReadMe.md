# Potato defect detection using DL #
This is a project to detect potato defects (specifically hollow-heart defects) using deep learning. The project uses a dataset of about 1500 images of potatoes(both healthy and with defects) to train a deep learning classification model. The model is then used to detect defects in new images of potatoes.

## Dataset ##
The dataset used for training the model is in the folder `"Data\tubers.zip"`. The dataset contains 1500 images of potatoes, each of which is labelled as either healthy or with a defect (see `"Data\A08241_internals.csv"`). The images are all of the same size (1900 x 1900 pixels) and are in the RGB format. Note: Only a few images have been uploaded in the `tubers.zip`

## Training & Testing the model ##
To train the model, follow the steps below:
1. Intall the required packages by running `"pip install -r requirements.txt"` in the command line.
2. Unzip the dataset in the folder `"Data\tubers.zip"`.
3. Preprocess the dataset by running the script `"preprocess_data.py"`. This will create a new folder (defined by the variable `saving_dir` in the script) containing the preprocessed images.
4. Split the dataset into training and test sets by running the script `"split_training_test.py"`. This will create two new folders containing the training and validation images respectively.
5. Train the model by running the script `"training.ipynb"`. In the script, you can choose the model to train by changing the variable `choice` to train a custom model or one of the pre-trained models (`ResNet50`, `ConvNext`, or `EfficientNet`). The script will save the trained model in a defined folder (defined by the variable `model_folder` in the script).
6. Test the model by running the script `"test_model.ipynb"`. The script will load the trained model and test it on the test set.s
