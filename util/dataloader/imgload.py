"""
This module implements basic operations to load images for model training
>>> # Generates a random image
>>> import numpy
>>> from PIL import Image
>>> imarray = numpy.random.rand(100,100,3) * 255  # Generates random image
>>> im = Image.fromarray(imarray.astype('uint8')).convert('L')
>>> im.save('test_image.png')  # Save Image
>>> # Read the image
>>> im_l = img_load('test_image.png')
>>> np.asarray(np.asarray(imarray/255)).all()  # Compare with the original array
True
"""


import imageio
import numpy as np


def img_load(*params, return_channels=True):
    """
    Loads an image and

    :param params: params
        The parameters are passed on through the imageio.imread function.
    :param return_channels: bool
        If True, will return a numpy array with shape (height, width, channels),
        (height, width) otherwise.

    :return:
    """
    i = imageio.imread(*params)
    i = np.asarray(i/255)
    if len(i.shape) > 2 and i.shape[2] > 1:
        i = i[:, :, 0]
    i = i.reshape(i.shape[0], i.shape[1], 1)
    if not return_channels:
        return i[:, :, 0]
    else:
        return i
