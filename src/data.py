import os
from os.path import dirname, abspath, join
from PIL import Image
import numpy as np


def load_chexpert_data():
    train_data = {}
    chexpert_path = join(dirname(dirname(abspath(__file__))), "data", "CheXpert-v1.0-small")
    chexpert_train_path = join(chexpert_path, "train")
    chexpert_test_path = join(chexpert_path, "valid")
    for patient in os.listdir(chexpert_train_path):
        patient_path = join(chexpert_train_path, patient)
        for study in os.listdir(patient_path):
            patient_study_path = join(patient_path, study)
            for image in os.listdir(patient_study_path):
                image_path = join(patient_study_path, image)
                try:
                    train_data[image_path] = np.array(Image.open(image_path))
                except:
                    print(image_path)
    print(train_data)

load_chexpert_data()

