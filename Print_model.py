
import tensorflow as tf
from tensorflow.keras.utils import plot_model

model_folder = 'Resnet50_1024/'

model = tf.keras.models.load_model(model_folder+'my_model.h5')
checkpoint_path = model_folder+"cp.ckpt"
model.load_weights(checkpoint_path)

print(model.summary()) # Preview model architecture

dst_folder = model_folder+'model_plot.png'
plot_model(model, to_file=dst_folder, show_shapes=True, show_layer_names=True, dpi = 600, expand_nested=True)