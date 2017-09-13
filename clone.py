import csv
import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.utils.visualize_util import plot
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, BatchNormalization, Dropout, MaxPooling2D

# References of the path and filenames used for training and validation
drive_log = './data/carnd_data/driving_log.csv'
url_path = './data/carnd_data/IMG/'
# drive_log = './data/full_data/driving_log.csv'
# url_path = './data/full_data/IMG/'

# Load each record into reader for iteration
samples = []
with open(drive_log) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)

# Split samples for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# Parameters used during training and validation
batch = 32 # basic size, that will be multiplied during execution in the pipeline according to pre-processing
correction = 0.28  # 0.28 is good value

# references used to print and store example images for the write-up
list_images = []
pp_images = []

# generator function, recieving a sample list. verbose only used for printing and debugging
def generator(samples, batch_size=batch, verbose=False):
    pp_sampling = True
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        cycle = 0 # helper variable for debugging
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steer_angles = []

            for batch_sample in batch_samples:

                # "realtime" preprocessing pipeline equal for test/validation
                # cam_idx iterates over center/left/right camera image
                for cam_idx in range(0, 2): # batch x3

                    #build the URL and read the image, transform BGR2RGB
                    URL = url_path + batch_sample[cam_idx].split('/')[-1]  # index[0] => center image
                    image = cv2.cvtColor(cv2.imread(URL), cv2.COLOR_BGR2RGB)  # convert: cv2.imread is BGR but drive.py sends RGB
                    #determine which camera and correct steering angle accordingly
                    if cam_idx == 2:
                        angel = float(batch_sample[3]) - correction # right
                    elif cam_idx == 1:
                        angel = float(batch_sample[3]) + correction # left
                    else:
                        angle = float(batch_sample[3])

                    # add center/left/right image and corrected angle to pipe
                    images.append(image)
                    steer_angles.append(angle)
                    if pp_sampling: pp_images.append(image)

                    # flip and add to pipe
                    flipped = cv2.flip(np.copy(image), 1) # x2
                    images.append(flipped)
                    steer_angles.append(angle * -1.0)
                    if pp_sampling: pp_images.append(flipped)


                    # Further image augmentation (blur, median, bilateral, clahe)
                    # blur and add to pipe
                    blurred = cv2.blur(np.copy(image), (3, 3)) # x2
                    images.append(blurred)
                    steer_angles.append(angle)
                    if pp_sampling: pp_images.append(blurred)

                    # median blur and add to pipe
                    median = cv2.medianBlur(np.copy(image), 3) # 3 is good value
                    images.append(median)
                    steer_angles.append(angle)
                    if pp_sampling: pp_images.append(median)

                    # bilateral filter and add to pipe
                    bilateral = cv2.bilateralFilter(np.copy(image), 5, 75, 55) # x2
                    images.append(bilateral)
                    steer_angles.append(angle)
                    if pp_sampling: pp_images.append(bilateral)

                    # apply contrast using channel split and CLAHE filter
                    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
                    lab = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
                    l, a, b = cv2.split(lab)  # split on 3 different channels
                    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
                    lab = cv2.merge((l2, a, b))  # merge channels
                    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
                    images.append(contrast)
                    steer_angles.append(angle)
                    if pp_sampling: pp_images.append(contrast)

                    pp_sampling = False # switch-off to collect augmented image examples only once not each batch/epoch

            cycle += 1 # counter for debugging (ignore)

            # create numpy arrays as required by Keras
            X_train = np.array(images)
            y_train = np.array(steer_angles)

            if verbose: (cycle, ': ', X_train.shape) # print shape for debugging

            for i in range(0, len(images)):
                list_images.append(images[i])

            yield shuffle(X_train, y_train)


# feed the generator to operate on batches
train_generator = generator(train_samples, batch_size=batch)
validation_generator = generator(validation_samples, batch_size=batch)


# Building the model, pipeline can be run with existing model from file or as build here: Used for this project N001
def get_initialized_model(name, path='', reload=False, verbose=True):
    if reload:
        model = load_model(path+name)
        if verbose: print('Model and weights loaded from file: ', path+name)
    else:
        # Initialize pre-processing parameters used with Keras functions
        row, column, channel = 160, 320, 3
        top_crop, bottom_crop, left_crop, right_crop = 65, 30, 0, 0

        model = Sequential()
        model.add(Cropping2D(cropping=((top_crop, bottom_crop), (left_crop, right_crop)), input_shape=(row, column, channel)))
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row, column, channel)))
        # model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, mode=2, gamma_regularizer=None,input_shape=(row, column, channel)))

        if name=='N001': # best model used for this project
            model.add(Convolution2D(24, 3, 3, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(48, 3, 3, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(72, 3, 3, activation='relu'))
            model.add(Convolution2D(96, 3, 3, activation='relu'))
            model.add(Convolution2D(96, 3, 3, activation='relu'))
            model.add(Convolution2D(128, 1, 1, activation='relu'))
            model.add(Convolution2D(128, 1, 1, activation='relu'))
            model.add(Dropout(0.6))

            model.add(Flatten())
            # model.add(Dense(1164))
            model.add(Dense(200))
            model.add(Dense(25))
            model.add(Dense(9))
            model.add(Dense(1))
            if verbose: print('N001: New model and weights from scratch...')

        elif name=='N002': # optional model
            model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(Convolution2D(64, 1, 1, activation='relu'))
            model.add(Convolution2D(64, 1, 1, activation='relu'))
            model.add(Convolution2D(64, 1, 1, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Flatten())
            # model.add(Dense(1164))
            model.add(Dense(200))
            model.add(Dense(22))
            model.add(Dense(10))
            model.add(Dense(1))
            if verbose: print('N002: New model and weights from scratch...')

        elif name == 'N003': # optional model
            model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            # model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(1164))
            model.add(Dense(100))
            model.add(Dense(50))
            model.add(Dense(10))
            model.add(Dense(1))
            if verbose: print('Nvidia: New model and weights from scratch...')

        else:
            # Simple direct prediction
            model.add(Flatten())
            model.add(Dense(1))
            if verbose: print('D')

    # print model summary and save visualization
    if verbose: model.summary(); plot(model, to_file=name+'_model.png', show_shapes=True)
    return model


# Compile and train the model
# model = get_initialized_model('model002.h5','./myArchive/', True, True)
model = get_initialized_model('N001') # set the model to used
print('sample_per_epoch: ', len(train_samples))
model.compile(loss='mse', optimizer='adam', )
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), \
                    nb_epoch=15, verbose=2)  # try no more than 3 epochs


def plot_history_object(history_object, verbose=True):
    if verbose:
        print('History_object: ', history_object)

        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


plot_history_object(history_object)

# save the model with weights
if False: # switch, TODO: check if exist and handle eg. with dialog (y/n)
    filename = 'model.h5'
    model.save(filename)
    print('Saved model: ', filename)


# utility to show 10 random examples, or in sequence
def plot_images(img, haxis=1, vaxis=1, random=True, verbose=True):
    fig, axes = plt.subplots(haxis, vaxis, gridspec_kw={'wspace': 0, 'hspace': 0})
    for i in range(haxis):
        for j in range(vaxis):
            if random:
                idx = randint(0, len(img))
                axes[i, j].imshow(img[idx])
                if verbose: axes[i, j].set_title(idx)
            else:
                axes[i, j].imshow(img[j])
                if verbose: axes[i, j].set_title(j)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    if verbose: print('len plot: ', len(img))
    if verbose: plt.show()


plot_images(pp_images, 2, 6, False) # show in sequence
plot_images(list_images, 10, 10) # show random selection

exit()
