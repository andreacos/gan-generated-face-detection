import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import configuration as config
from glob import glob
from network import xception_net, efficient_net
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from custom_generators import SimpleBatchGenerator
from utils import plot_training_history
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == '__main__':

    def create_training_validation_sets():

        filenames_train_pristine = glob(os.path.join(config.XCP_TRAIN_FOLDER, 'Pristine', '*.*'))
        filenames_train_gan = glob(os.path.join(config.XCP_TRAIN_FOLDER, 'GAN', '*.*'))

        filenames_train = filenames_train_pristine + filenames_train_gan
        labels_train = [config.XCP_PRISTINE_LABEL if "Pristine" in i else config.XCP_GAN_LABEL for i in filenames_train]

        x_train, x_validation, y_train, y_validation = train_test_split(filenames_train, labels_train,
                                                                        test_size=0.025, random_state=42)
        y_train = tf.keras.utils.to_categorical(y_train, config.XCP_NUM_CLASSES)
        y_validation = tf.keras.utils.to_categorical(y_validation, config.XCP_NUM_CLASSES)

        return x_train, x_validation, y_train, y_validation


    def create_batch_generators(x_train, x_validation, y_train, y_validation, augmentation):

        train_generator = SimpleBatchGenerator(x_train, y_train, config.XCP_TRAIN_BATCH,
                                               scale=1. / 255,
                                               width=config.XCP_INPUT_SIZE,
                                               augmentation=augmentation,
                                               jpeg_prob=100,
                                               jpeg_factors=np.arange(70, 99, 1),
                                               resize_prob=80,
                                               resize_factors=np.arange(0.8, 1.3, 0.1),
                                               flip_prob=20,
                                               rotation_prob=30,
                                               rotation_range=np.arange(-15, 15, 1),
                                               noise_probability=35)

        val_generator = SimpleBatchGenerator(x_validation, y_validation, config.XCP_TRAIN_BATCH,
                                             scale=1. / 255,
                                             width=config.XCP_INPUT_SIZE,
                                             augmentation=augmentation,
                                             jpeg_prob=100,
                                             jpeg_factors=np.arange(70, 99, 5),
                                             resize_prob=80,
                                             resize_factors=np.arange(0.8, 1.3, 0.1),
                                             flip_prob=20,
                                             rotation_prob=30,
                                             rotation_range=np.arange(-15, 15, 1),
                                             noise_probability=35)

        return train_generator, val_generator


    def initialize_network(fine_tune_model_file='', network_type='xception'):

        if fine_tune_model_file == '':

            if network_type == 'xception':
                model = xception_net(in_shape=(config.XCP_INPUT_SIZE, config.XCP_INPUT_SIZE, config.XCP_INPUT_CHANNELS))
            else:
                model = efficient_net(in_shape=(config.XCP_INPUT_SIZE, config.XCP_INPUT_SIZE, config.XCP_INPUT_CHANNELS))

        else:
            print(f"Fine-tuning from model {fine_tune_model_file}")
            model = load_model(fine_tune_model_file)

        checkpoint_saver = ModelCheckpoint(filepath=os.path.join('models', 'checkpoints', exp_id,
                                                                 'ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                           monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False)

        if network_type == 'xception':

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
            callbacks = [lr_reducer, checkpoint_saver]

            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                          metrics=['accuracy'])

        else:

            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, 80000, 1e-5, power=0.8)
            callbacks = [checkpoint_saver]

            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
                          metrics=['accuracy'])

        # Display a summary of configuration and network
        print(model.summary())
        plot_model(model, to_file=f'{exp_id}model.png', show_shapes=True)

        return model, callbacks

    exp_id = f"Xception-augmentation-ep-{config.XCP_NUM_EPOCHS}-face-dataset-with-nvidia-and-print-scan"
    print(f"Experiment id: {exp_id}")

    print(f"Creating training and validation data sets ... ")
    os.makedirs(os.path.join('models', f'checkpoints/{exp_id}'), exist_ok=True)
    x_train, x_validation, y_train, y_validation = create_training_validation_sets()

    print(f"Creating training and validation custom generators with augmentation... ")
    print(f"Augmentation is {str(config.XCP_AUGMENTATION).upper()}")
    training_generator, validation_generator = create_batch_generators(x_train=x_train, x_validation=x_validation,
                                                                       y_train=y_train, y_validation=y_validation,
                                                                       augmentation=config.XCP_AUGMENTATION)

    print(f"Initializing network model ... ")

    model_file = ''
    xception_model, callbacks = initialize_network(fine_tune_model_file=model_file, network_type='xception')

    print(f"Training model ... ")
    history = xception_model.fit(training_generator, steps_per_epoch=int(len(x_train) // config.XCP_TRAIN_BATCH),
                                 validation_data=validation_generator,
                                 validation_steps=int(len(x_validation) // config.XCP_TRAIN_BATCH),
                                 callbacks=callbacks,
                                 epochs=config.XCP_NUM_EPOCHS)

    # Save history
    # print(f"Saving training history ... ")
    # with open(f'{exp_id}training-history.pkl', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    # Plot training history (accuracy and validation)
    # print("Plotting training history ... ")
    # history = pickle.load(open(f'{exp_id}training-history.pkl'))
    # plot_training_history(history, f"{exp_id}")
