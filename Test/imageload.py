from keras.src.utils import image_utils
import matplotlib.pyplot as plt
# from keras.src.legacy.preprocessing.image import load_img
img = image_utils.load_img('img.png')
plt.imshow(img)
plt.show()


from numpy import expand_dims
# from keras.src.legacy.preprocessing.image import load_img
from keras.src.legacy.preprocessing.image import img_to_array

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = image_utils.load_img('img.png')
# convert to numpy array
data = img_to_array(img)

samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=0.2,fill_mode="wrap")
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# for i in range(9):
# 	# define subplot
# 	plt.subplot(330 + 1 + i)
# 	# generate batch of images
#
# 	# my_iterator = iter(it)
# 	# batch = my_iterator.__next__()
#
# 	batch =it.__next__()
# 	# convert to unsigned integers for viewing
# 	image = batch[0].astype('uint8')
# 	# plot raw pixel data
# 	plt.imshow(image)
# show the figure
plt.show()
try:
    batch = next(it)
except StopIteration:
    print("Iterator exhausted.")
