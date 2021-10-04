import numpy as np
import math
import cv2
from utils import jpeg_compression_in_buffer, rotate, largest_rotated_rect, crop_around_center


def augment_predict(x, width, jpeg_probability, jpeg_range, rotation_probability, rotation_range,
                    flip_probability, resize_probability, resize_range):
    """
    Random augmentations
    :param x: input image
    :return: augmented image
    """

    do_jpeg = np.random.randint(0, 101)
    do_rotate = np.random.randint(0, 101)
    do_flip = np.random.randint(0, 101)
    do_resize = np.random.randint(0, 101)

    # Random JPEG compression
    if do_jpeg <= jpeg_probability:
        x = random_jpeg_compression(x, jpeg_range=jpeg_range)

    # Random rotation
    if do_rotate <= rotation_probability:
        x = random_rotation(x, rotation_range=rotation_range, width=width)

    # Random flip
    elif do_flip <= flip_probability:
        x = random_flip(x)

    # Random resize
    if do_resize <= resize_probability:
        x = random_resize(x, resize_range=resize_range, width=width)

    return x


def random_jpeg_compression(x, jpeg_range):
    """
    JPEG compression in file stream with random quality
    :param x: input image
    :param jpeg_range: array of quality factors
    :return: JPEG compressed image
    """
    qf = np.random.choice(jpeg_range)
    return np.array(jpeg_compression_in_buffer(x, int(qf)))


def random_resize(x, resize_range, width):
    """
    Resize with random scale factor f. If f<1 the image is resized to the training size. If f>1 the image is cropped
    from the center to the training size
    :param x: input image
    :param resize_range: array of scale factors
    :param width: original image size (H and W)
    :return: resized image
    """

    scale_f = np.random.choice(resize_range)

    x = cv2.resize(x, dsize=None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC)

    if scale_f < 1:
        x = cv2.resize(x, dsize=(width, width), interpolation=cv2.INTER_CUBIC)

    elif scale_f > 1:
        center = (x.shape[0] // 2, x.shape[1] // 2)
        x = x[center[0] - math.ceil(width / 2.):center[0] + math.floor(float(width) / 2.),
            center[1] - math.ceil(width / 2.):center[1] + math.floor(width / 2.)]
    return x


def random_flip(x):
    """
    Randomly flip image (left-right or up-down)
    :param x: input image
    :return: flipped image
    """
    # Randomly choose to flip rows (np.fliplr) or column (np.flipud)
    return np.flip(x, np.random.randint(2))


def random_rotation(x, rotation_range, width):
    """
    Image rotation with random angle. Black borders are cropped and cropped image is resize to training size
    :param x: input image
    :param rotation_range: array of angles for rotation
    :param width: original image size (H and W)
    :return: rotated image
    """
    angle = np.random.choice(rotation_range)
    x = rotate(x, angle)
    x = crop_around_center(x, *largest_rotated_rect(width, width, math.radians(angle)))
    x = cv2.resize(x, dsize=(width, width), interpolation=cv2.INTER_CUBIC)
    return x
