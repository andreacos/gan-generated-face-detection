from augment_on_predict import augment_predict
import tensorflow as tf
import numpy as np
import os
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import cv2
import configuration as cfg


if __name__ == '__main__':

    # Experiment id matching the training exp id
    exp_id = f"Xception-augmentation-ep-{cfg.XCP_NUM_EPOCHS}-face-dataset-with-nvidia-and-print-scan-on-nvidia-new"

    # Load model
    model_file = 'models/checkpoints/Xception-augmentation-ep-30-face-dataset-with-nvidia-and-print-scan/ckpt.epoch30-loss0.05.h5'
    model = tf.keras.models.load_model(model_file)

    # Paths to test set
    gan_images = glob(os.path.join(cfg.XCP_TEST_FOLDER, 'GAN', '*.*'))
    pristine_images = glob(os.path.join(cfg.XCP_TEST_FOLDER, 'Pristine', '*.*'))

    # Image and labels
    probes = pristine_images + gan_images
    true_labels = np.array([0 if "Pristine" in i else 1 for i in probes])

    input_shape = model.input_shape[1:]

    # Write to csv
    out_file = f'{exp_id}test-results.txt'
    with open(out_file, "w") as fp:

        print(f'Found {len(probes)} images to test')

        fp.write("Image;Score;Label;Class\n")

        idx = 0
        predictions = []
        pred_labels = []
        for im_file in tqdm(probes):

            # Read image
            img = cv2.imread(im_file)
            if img.shape != input_shape:
                img = cv2.resize(img, dsize=(input_shape[0], input_shape[1]))

            if cfg.XCP_AUGMENTATION:
                img = augment_predict(x=img,
                                      width=cfg.XCP_INPUT_SIZE,
                                      jpeg_probability=50,
                                      jpeg_range=np.arange(70, 99, 1),
                                      resize_probability=100,
                                      resize_range=np.arange(0.8, 1.3, 0.1),
                                      flip_probability=20,
                                      rotation_probability=30,
                                      rotation_range=np.arange(-15, 15, 1))

            # Classify
            score = model.predict(np.expand_dims(img / 255., 0))
            pred_label = np.argmax(score, 1)
            label_txt = 'GAN' if pred_label == 1 else 'NOT GAN'

            # store / write data
            predictions.append(score)
            pred_labels.append(pred_label)
            fp.write("{};{};{};{}\n".format(im_file, score, pred_label, label_txt))
            fp.flush()

            idx += 1

        # Save results
        predictions = np.array(predictions)
        pred_labels = np.array(pred_labels)

        # np.save(f"{exp_id}predictions.npy", predictions)
        # np.save(f"{exp_id}predicted-labels.npy", pred_labels)

        acc = np.sum(pred_labels.flatten() == true_labels.flatten()) / len(pred_labels)

        # Print results
        print(f"Accuracy: {acc}")
        fp.write(f"Accuracy: {acc}\n")

        print('Confusion Matrix: ')
        print(confusion_matrix(true_labels, pred_labels))

        fp.write('Confusion Matrix:\n')
        fp.write(str(confusion_matrix(true_labels, pred_labels)))

        print('Classification Report: ')
        print(classification_report(true_labels, pred_labels, target_names=['Pristine', 'GAN']))

        fp.write('Classification Report:\n')
        fp.write(classification_report(true_labels, pred_labels, target_names=['Pristine', 'GAN']))
