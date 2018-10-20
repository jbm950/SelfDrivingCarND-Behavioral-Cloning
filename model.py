import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Cropping2D
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split


def preprocess_images():
    """This function will import all of the example images
    sets in the data folder.
    
    Returns
    =======
    
        images: list
            This is a list containing all of the images
            from the data sets
        measurements: list
            This is a list containing all of the steering
            measurements
    """
    
    # Set up the lists to collect the images and measurements
    images = []
    measurements = []
    
    # Set up the path to the data files 
    data_sets_path = 'data'
    data_sets = [os.path.join(data_sets_path, i) for i
                 in os.listdir(data_sets_path)]
    
    # Step through the data folders and collect the images
    # and the steering angles
    for data_set in data_sets:
        lines = []
        
        # Open up the csv file of image paths and steering angles
        with open(os.path.join(data_set,
                               'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        for line in lines:
            source_path = line[0]
            filename = source_path.split('\\')[-1]
            current_path = os.path.join(data_set, 'IMG',
                                        filename)
            
            # Import each image and change it to RGB
            BGR_image = cv2.imread(current_path)
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)
            rows, cols, depth = image.shape
            flipped_image = cv2.flip(image, 1)
            
            # Create a scaled version of the image
            scale = [0.9, 1.1]
            zoomfactor = random.choice(scale)
            scale_matrix = cv2.getRotationMatrix2D((cols/2, rows/2),
                                                   0, zoomfactor)
            scaled_image = cv2.warpAffine(image, scale_matrix,
                                          (cols, rows))

            # Append the images to the image list
            images.append(image)
            images.append(scaled_image)
            images.append(flipped_image)
            
            # Append the steering angle to the measurements list
            measurement = float(line[3])
            measurements.append(measurement)
            measurements.append(measurement)
            measurements.append(-1*measurement)
            
    return images, measurements


# Load the images if the preprocessing has already been done
# in the past otherwise preprocess the images
preprocess_images_directory = 'preprocess-images'
if not os.path.exists(preprocess_images_directory):
    os.makedirs(preprocess_images_directory)
    images, measurements = preprocess_images()
    example_images = np.array(images)
    example_labels = np.array(measurements)
    X_train, X_val, y_train, y_val = train_test_split(example_images,
                                                      example_labels,
                                                      test_size=0.33,
                                                      random_state=0)
    with open(os.path.join(preprocess_images_directory,
                           'train.npz'), 'wb') as train_file:
        np.savez(train_file, X_train, y_train, allow_pickle=False)
    with open(os.path.join(preprocess_images_directory,
                           'valid.npz'), 'wb') as valid_file:
        np.savez(valid_file, X_val, y_val, allow_pickle=False)
else:
    with open(os.path.join(preprocess_images_directory,
                           'train.npz'), 'rb') as train_file:
        data = np.load(train_file, allow_pickle=False)
        X_train = data['arr_0']
        y_train = data['arr_1']
    with open(os.path.join(preprocess_images_directory, 
                           'valid.npz'), 'rb') as valid_file:
        data = np.load(valid_file, allow_pickle=False)
        X_val = data['arr_0']
        y_val = data['arr_1']
        
# Display the data set characteristics
print("\nData Set Characteristics")
print("Number of Training Examples: %d" % len(X_train))
print("Number of Validation Examples: %d" % len(X_val))
print('\n\n')

def generator(features, labels, batch_size):
    """This function will produce a batch of features
    and labels for each epoch step to reduce the memory usage.
    
    Parameters
    ==========
    
        features: np.array
            This is the array of images for the model to train on
        labels: np.array
            This is the array of labels for the model to train on
    
    Returns
    =======
    
        batch_features: np.array
            This is a subset of the features that will be used 
            for each batch
        batch_labels: np.array
            This is the corresponding labels to the batch of features
    """
    
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 160, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(range(len(features)))
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

# Set up the model
model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(6, (5, 5), padding='valid',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, (5, 5), padding='valid',
                 activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (5, 5), padding='valid', 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(150, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
    
# Set up training parameters
batch_size = 10
epochs = 10

model.fit_generator(generator(X_train, y_train, batch_size), 
                    validation_data=generator(X_val, y_val, batch_size),
                    steps_per_epoch=int(len(X_train)/batch_size), 
                    nb_val_samples=int(len(X_val)/batch_size),
                    nb_epoch=epochs)

# Save the trained model
#model.save('model.h5')