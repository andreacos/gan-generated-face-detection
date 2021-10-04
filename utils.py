
"""
    2021 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from io import BytesIO
import imageio


def cooccurence_fast(image1, image2, distance_x, distance_y, levels):

    fast_cooccurence_matrix = np.zeros((levels, levels), dtype='int64')

    sx = np.size(image1, 0)
    sy = np.size(image1, 1)
    image1_ready = image1[0:sx-distance_x, 0:sy-distance_y]
    image2_ready = image2[distance_x:, distance_y:]

    for i in range(levels):
        image2_ready_temp = image2_ready[image1_ready == i]
        for j in range(levels):
            fast_cooccurence_matrix[i, j] = np.sum(image2_ready_temp == j)

    return fast_cooccurence_matrix


def compute_cmat(image, levels=256):

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Horizontal
    distance_x = 1
    distance_y = 1
    RR_D = cooccurence_fast(R, R, distance_x, distance_y, levels)
    GG_D = cooccurence_fast(G, G, distance_x, distance_y, levels)
    BB_D = cooccurence_fast(B, B, distance_x, distance_y, levels)

    distance_xx = 0
    distance_yy = 0
    RG_D = cooccurence_fast(R, G, distance_xx, distance_yy, levels)
    RB_D = cooccurence_fast(R, B, distance_xx, distance_yy, levels)
    GB_D = cooccurence_fast(G, B, distance_xx, distance_yy, levels)

    # Normalization VIPP
    max_val = np.amax([np.amax(RR_D), np.amax(GG_D), np.amax(BB_D), np.amax(RG_D), np.amax(RB_D), np.amax(GB_D)])

    RR_D1 = RR_D / max_val
    GG_D1 = GG_D / max_val
    BB_D1 = BB_D / max_val
    RG_D1 = RG_D / max_val
    RB_D1 = RB_D / max_val
    GB_D1 = GB_D / max_val

    return np.stack((RR_D1, GG_D1, BB_D1, RG_D1, RB_D1, GB_D1))


def plot_average_accuracy(arr, savefile='accuracy_vs_coeffs.png'):

    """ Plots exact accuracy (averaged over all dataset) for each estimated coefficient of the
        quantisation matrix
    Keyword arguments:
    arr : array with test accuracy
    savefile: if equal to '', displays plots otherwise path to the saved plot image
    """

    fig, ax1 = plt.subplots(1, 1)
    cax1 = ax1.matshow(arr, cmap='Blues')
    ax1.set_title('Average accuracy for each coefficient')
    cbar = fig.colorbar(cax1, ax=ax1, cmap='Blues', fraction=0.046, pad=0.04)

    # # Write values over matrix data
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            c = arr[j][i]
            ax1.text(i, j, '{:3.2f}'.format(c), va='center', ha='center', fontsize=18)

    if savefile != '':
        plt.savefig(savefile, dpi=fig.dpi)
        plt.close(fig)
    else:
        plt.show()

    return


def plot_average_epoch_metric(metric, loss_arr, n_epochs, exp_identifier, show=True):

    loss_arr = np.array(loss_arr)
    iter_epochs = loss_arr.shape[0] // n_epochs
    me = []
    for i in range(n_epochs):
        ls = loss_arr[i * iter_epochs:(i + 1) * iter_epochs]
        me.append(np.mean(ls))

    if show:
        plt.figure(figsize=(12, 9))
        plt.plot(np.arange(n_epochs), np.array(me), 'o-', markersize=8)
        plt.xticks(np.arange(0, n_epochs, step=1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.xlabel('Epoch ({:d} iterations)'.format(iter_epochs))
        plt.ylabel('Average {}'.format(metric))
        plt.title(exp_identifier)
        plt.grid()
        plt.show()

    return me


def jpeg_compression_in_buffer(x_in, jpg_quality):

    x = Image.fromarray(np.uint8(x_in))

    buf = BytesIO()
    x.save(buf, format='jpeg', quality=jpg_quality)

    with BytesIO(buf.getvalue()) as stream:
        x_jpg = imageio.imread(stream)

    return x_jpg


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def largest_rotated_rect(w, h, angle):

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def plot_training_history(history, file_name):

    # history = pickle.load(open('/trainHistoryDict'), "rb")

    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"{file_name}training-accuracy.png")
    plt.close()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"{file_name}training-loss.png")

    return


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = abs(min(width - x1, size_bb))
    size_bb = abs(min(height - y1, size_bb))

    return x1, y1, size_bb


if __name__ == '__main__':
    import pickle
    history = pickle.load(open(
        'results/EfficientNet-augmentation-ep-15-mixed-dataset-from-no-aug-mixed-15-epochs/EfficientNet-augmentation-ep-15-mixed-dataset-from-no-aug-mixed-15-epochstraining-history.pkl', "rb"))
    plot_training_history(history, 'EfficientNet-augmentation-mixed-ep-15-from-mixed-no-aug')
