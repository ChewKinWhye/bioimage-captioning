import os
from os.path import dirname, abspath, join
from PIL import Image
import numpy as np
import csv


def load_chexpert_data():
    train_y_data = {}
    base_path = join(dirname(dirname(abspath(__file__))), "data")
    chexpert_path = join(base_path, "CheXpert-v1.0-small")
    train_csv_path = join(chexpert_path, "train.csv")
    with open(train_csv_path) as csv_file:
        line_count = 0
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                line_count += 1
                train_y_data[row[0]] = row[5:]
    train_x_data = {}
    chexpert_train_path = join(chexpert_path, "train")
    chexpert_test_path = join(chexpert_path, "valid")
    line_count = 0
    for patient in os.listdir(chexpert_train_path):
        line_count += 1
        if line_count == 1000:
            break
        patient_path = join(chexpert_train_path, patient)
        for study in os.listdir(patient_path):
            patient_study_path = join(patient_path, study)
            for image in os.listdir(patient_study_path):
                image_path = join(patient_study_path, image)
                try:
                    key_value = image_path.replace(base_path + "/", "")
                    train_x_data[key_value] = np.array(Image.open(image_path).resize((390, 320)))
                except:
                    print(base_path)
                    print(image_path)
    train_x_numpy = []
    train_y_numpy = []
    for key, value in train_x_data.items():
        if key in train_y_data:
            train_x_numpy.append(value)
            train_y_numpy.append(train_y_data[key])
    train_x_numpy = np.array(train_x_numpy)
    train_y_numpy = np.array(train_y_numpy)
    print(train_x_numpy.shape)
    print(train_y_numpy.shape)
    return train_x_numpy, train_y_numpy
load_chexpert_data()

