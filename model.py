import numpy as np
import matplotlib.pyplot as plt

import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.src.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Concatenate
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model
IMGWIDTH = 256

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMGWIDTH = 256

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}


class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)




class Meso10(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)

        x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(x4)
        x5 = BatchNormalization()(x5)
        x5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x5)

        x6 = Conv2D(32, (5, 5), padding='same', activation='relu')(x5)
        x6 = BatchNormalization()(x6)
        x6 = MaxPooling2D(pool_size=(2, 2), padding='same')(x6)

        x7 = Conv2D(64, (5, 5), padding='same', activation='relu')(x6)
        x7 = BatchNormalization()(x7)
        x7 = MaxPooling2D(pool_size=(2, 2), padding='same')(x7)

        x8 = Conv2D(64, (5, 5), padding='same', activation='relu')(x7)
        x8 = BatchNormalization()(x8)
        x8 = MaxPooling2D(pool_size=(2, 2), padding='same')(x8)

        y = Flatten()(x8)
        y = Dropout(0.5)(y)
        y = Dense(128)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


# Instantiate a MesoNet model with pretrained weights
meso = Meso10()
# meso.load('./weights/Meso4_DF')
meso.load('C:/Users/Rajnish/OneDrive/Desktop/deepfake_detection/weights/Meso4_DF1')

# Prepare image data

# Rescaling pixel values (between 1 and 255) to a range between 0 and 1
dataGenerator = ImageDataGenerator(rescale=1. / 255)

# Instantiating generator to feed images through the network
generator = dataGenerator.flow_from_directory(
    './data/',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Checking class assignment
# generator.class_indices

# Recreating generator after removing '.ipynb_checkpoints'



# Re-checking class assignment after removing it
# generator.class_indices

X, y = generator.__next__()

# Evaluating prediction
print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0]) == y[0]}")

# Showing image
plt.imshow(np.squeeze(X));


correct_real = []
correct_real_pred = []

correct_deepfake = []
correct_deepfake_pred = []

misclassified_real = []
misclassified_real_pred = []

misclassified_deepfake = []
misclassified_deepfake_pred = []

# Generating predictions on validation set, storing in separate lists
for i in range(len(generator.labels)):

    # Loading next picture, generating prediction
    X, y = generator.next()
    pred = meso.predict(X)[0][0]

    # Sorting into proper category
    if round(pred) == y[0] and y[0] == 1:
        correct_real.append(X)
        correct_real_pred.append(pred)
    elif round(pred) == y[0] and y[0] == 0:
        correct_deepfake.append(X)
        correct_deepfake_pred.append(pred)
    elif y[0] == 1:
        misclassified_real.append(X)
        misclassified_real_pred.append(pred)
    else:
        misclassified_deepfake.append(X)
        misclassified_deepfake_pred.append(pred)

    # Printing status update
    if i % 1000 == 0:
        print(i, ' predictions completed.')

    if i == len(generator.labels) - 1:
        print("All", len(generator.labels), "predictions completed")


def plotter(images, preds):
    fig = plt.figure(figsize=(16, 9))
    subset = np.random.randint(0, len(images) - 1, 12)
    for i, j in enumerate(subset):
        fig.add_subplot(3, 4, i + 1)
        plt.imshow(np.squeeze(images[j]))
        plt.xlabel(f"Model confidence: \n{preds[j]:.4f}")
        plt.tight_layout()
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    plt.show()
    return


plotter(correct_real, correct_real_pred)

plotter(misclassified_real, misclassified_real_pred)

plotter(correct_deepfake, correct_deepfake_pred)

plotter(misclassified_deepfake, misclassified_deepfake_pred)
