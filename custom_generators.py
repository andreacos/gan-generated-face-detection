import tensorflow as tf
import numpy as np
import cv2
import math
import configuration as cfg
from utils import rotate, crop_around_center, largest_rotated_rect, jpeg_compression_in_buffer


class SimpleBatchGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, width, scale, augmentation, jpeg_prob, jpeg_factors,
                 resize_prob, resize_factors, flip_prob, rotation_prob, rotation_range, noise_probability=0):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.width = width
        self.scale_factor = scale
        self.enable_augmentation = augmentation
        self.jpeg_probability = jpeg_prob
        self.jpeg_range = jpeg_factors
        self.resize_probability = resize_prob
        self.resize_range = resize_factors
        self.flip_probability = flip_prob
        self.rotation_probability = rotation_prob
        self.rotation_range = rotation_range
        self.noise_probability = noise_probability

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        arr = []
        for file_name in batch_x:

            im = cv2.imread(str(file_name))

            # Augment data if augmentation is enable
            if self.enable_augmentation:
                im = self.augment(im, fname=file_name)

            arr.append(im)

        return np.array(arr) * self.scale_factor, np.array(batch_y)

    def augment(self, x, fname=''):
        """
        Random augmentations
        :param x: input image
        :return: augmented image
        """

        do_jpeg = np.random.randint(0, 101)
        do_rotate = np.random.randint(0, 101)
        do_flip = np.random.randint(0, 101)
        do_resize = np.random.randint(0, 101)
        do_gaussian_noise = np.random.randint(0, 101)

        # Random JPEG compression
        if do_jpeg <= self.jpeg_probability:
            x = self.random_jpeg_compression(x)

        # Random rotation
        if do_rotate <= self.rotation_probability:
            x = self.random_rotation(x)

        # Random flip
        elif do_flip <= self.flip_probability:
            x = self.random_flip(x)

        elif do_gaussian_noise <= self.noise_probability:
            if 'print_scan' not in fname:
                x = self.random_noise(x)

        # Random resize
        if do_resize <= self.resize_probability:
            x = self.random_resize(x)

        return x

    def random_noise(self, x):
        gauss = np.random.normal(0, 1, x.size)
        gauss = gauss.reshape((x.shape[0], x.shape[1], x.shape[2])).astype('uint8')
        return cv2.add(x, gauss)

    def random_jpeg_compression(self, x):
        """
        JPEG compression in file stream with random quality
        :param x: input image
        :return: JPEG compressed image
        """
        qf = np.random.choice(self.jpeg_range)
        return np.array(jpeg_compression_in_buffer(x, int(qf)))

    def random_resize(self, x):
        """
        Resize with random scale factor f. If f<1 the image is resized to the training size. If f>1 the image is cropped
        from the center to the training size
        :param x: input image
        :return: resized image
        """

        scale_f = np.random.choice(self.resize_range)

        x = cv2.resize(x, dsize=None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC)

        if scale_f < 1:
            x = cv2.resize(x, dsize=(self.width, self.width), interpolation=cv2.INTER_CUBIC)

        elif scale_f > 1:
            center = (x.shape[0]//2, x.shape[1]//2)
            x = x[center[0]-math.ceil(self.width/2.):center[0]+math.floor(float(self.width)/2.),
                  center[1]-math.ceil(self.width/2.):center[1]+math.floor(self.width/2.)]
        return x

    def random_flip(self, x):
        """
        Randomly flip image (left-right or up-down)
        :param x: input image
        :return: flipped image
        """
        # Randomly choose to flip rows (np.fliplr) or column (np.flipud)
        return np.flip(x, np.random.randint(2))

    def random_rotation(self, x):
        """
        Image rotation with random angle. Black borders are cropped and cropped image is resize to training size
        :param x: input image
        :return: rotated image
        """
        angle = np.random.choice(self.rotation_range)
        x = rotate(x, angle)
        x = crop_around_center(x, *largest_rotated_rect(self.width, self.width, math.radians(angle)))
        x = cv2.resize(x, dsize=(self.width, self.width), interpolation=cv2.INTER_CUBIC)
        return x
