"""
    2020 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:

"""

# ---------------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------------

XCP_TRAIN_FOLDER = './Datasets/CNN/299/Train'
XCP_TEST_FOLDER = './Datasets/CNN/299/Test'


# ---------------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------------

# XCEPTION NET

XCP_INPUT_SIZE = 299              # Patch size
XCP_INPUT_CHANNELS = 3            # Color mode

XCP_AUGMENTATION = True

XCP_NUM_CLASSES = 2
XCP_PRISTINE_LABEL = 0            # Tag for pristine image class
XCP_GAN_LABEL = 1                 # Tag for enhanced image class

XCP_NUM_EPOCHS = 30               # Number of training epochs
XCP_TRAIN_BATCH = 16               # Training batch size
XCP_TEST_BATCH = 16               # Test batch size
