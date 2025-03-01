from keras.utils import img_to_array, load_img
import numpy as np
import tensorflow as tf
import os
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose  #sequential api
from keras.models import Sequential
import cv2

#tf.config.run_functions_eagerly(True)

store_image = []
train_path = r"D:\SAHITHI BALLA\projects\sahithi\video_abnormal-detection\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train\Train001"
fps = 5
train_videos = os.listdir(train_path)
train_images_path = train_path + '/frames'
os.makedirs(train_images_path, exist_ok=True)

def store_inarray(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2] #lc
    store_image.append(gray)

for video in train_videos:
    os.system('ffmpeg -i {}/{} -r 1/{} {}/frames/%03d.jpg'.format(train_path, video, fps, train_path))
    images = os.listdir(train_images_path)
    for image in images:
        image_path = train_images_path + '/' + image
        store_inarray(image_path)

# Check if store_image array is empty
if len(store_image) == 0:
    print("store_image is empty, no images found.")
else:
    # Reshape store_image array
    store_image = np.array(store_image)
    num_samples = len(store_image) // 10  # Number of samples
    image_height, image_width = 227, 227  # Image dimensions
    num_channels = 1  # Grayscale image
    store_image = store_image.reshape(num_samples, image_height, image_width, 10, num_channels)

    # Normalize the store_image array
    store_image = (store_image - store_image.mean()) / store_image.std()
    store_image = np.clip(store_image, 0, 1)

    # Save the preprocessed data
    np.save('training.npy', store_image)
# Load the preprocessed data
training_data = np.load('training.npy')
# Split the training data into input and target data
input_data = training_data[:, :, :, :-1]
target_data = training_data[:, :, :, 1:]

# Update the input shape of the model
input_shape = training_data.shape[1:]
# Enable eager execution for tf.data functions
#tf.data.experimental.enable_debug_mode()
# Enable eager execution globally
#tf.config.run_functions_eagerly(True)
# Define and compile the model
stae_model = Sequential()
stae_model.add(
    Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid',
           input_shape=input_shape, activation='tanh'))
stae_model.add(
    Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4,
                           recurrent_dropout=0.3, return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3,
                           return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5))
stae_model.add(
    Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(
    Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tanh'))

stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'], run_eagerly=True)

# Print the shapes of training_data and target_data
print("Training data shape:", training_data.shape)
print("Target data shape:", target_data.shape)

epochs = 5
batch_size = 1

for epoch in range(epochs):
    print("Epoch:", epoch+1)
    history = stae_model.fit(training_data, target_data, batch_size=batch_size)
    print("Epoch loss:", history.history['loss'])
    print("Epoch accuracy:", history.history['accuracy'])

stae_model.save("model.h5")