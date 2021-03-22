import os
from os.path import dirname, abspath, join
from PIL import Image
import numpy as np
import csv


def load_chexpert_data():
    # Extract data from train.csv file
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

    # Extract data from test.csv file
    test_y_data = {}
    base_path = join(dirname(dirname(abspath(__file__))), "data")
    chexpert_path = join(base_path, "CheXpert-v1.0-small")
    test_csv_path = join(chexpert_path, "valid.csv")
    with open(test_csv_path) as csv_file:
        line_count = 0
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                line_count += 1
                test_y_data[row[0]] = row[5:]

    # Extract data from train directory
    train_x_data = {}
    chexpert_train_path = join(chexpert_path, "train")
    line_count = 0
    for patient in os.listdir(chexpert_train_path):
        line_count += 1
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

    # Extract data from test directory
    test_x_data = {}
    chexpert_test_path = join(chexpert_path, "valid")
    line_count = 0
    for patient in os.listdir(chexpert_test_path):
        line_count += 1
        patient_path = join(chexpert_test_path, patient)
        for study in os.listdir(patient_path):
            patient_study_path = join(patient_path, study)
            for image in os.listdir(patient_study_path):
                image_path = join(patient_study_path, image)
                try:
                    key_value = image_path.replace(base_path + "/", "")
                    test_x_data[key_value] = np.array(Image.open(image_path).resize((390, 320)))
                except:
                    print(base_path)
                    print(image_path)

    # Combine train data
    train_x_numpy = []
    train_y_numpy = []
    for key, value in train_x_data.items():
        if key in train_y_data:
            train_x_numpy.append(value)
            train_y_numpy.append(train_y_data[key])
        else:
            print("?")
    train_x_numpy = np.array(train_x_numpy)
    train_y_numpy = np.array(train_y_numpy)

    # Combine test data
    test_x_numpy = []
    test_y_numpy = []
    for key, value in test_x_data.items():
        if key in test_y_data:
            test_x_numpy.append(value)
            test_y_numpy.append(test_y_data[key])
        else:
            print("?")
    test_x_numpy = np.array(test_x_numpy)
    test_y_numpy = np.array(test_y_numpy)
    print(f"Train X shape: {train_x_numpy.shape}")
    print(f"Train Y shape: {train_y_numpy.shape}")
    print(f"Test X shape: {test_x_numpy.shape}")
    print(f"Test Y shape: {test_y_numpy.shape}")

    return train_x_numpy, train_y_numpy, test_x_numpy, test_y_numpy
load_chexpert_data()

