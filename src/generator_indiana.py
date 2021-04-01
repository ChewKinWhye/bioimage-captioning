import numpy as np
from tensorflow import keras
from os.path import dirname, abspath, join
from PIL import Image
import csv
import os
from imgaug import augmenters as iaa
import unicodedata
import re
import tensorflow as tf
from data import load_indiana_data
from tensorflow.keras import backend as K


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_dimension, model, class_names, batch_size=32, shuffle=True):
        'Initialization'
        self.class_names = class_names
        self.model = model
        self.image_dimension = (image_dimension[1], image_dimension[2])
        self.augmenter = iaa.Sequential([iaa.Fliplr(0.5),], random_order=True,)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = []
        self.base_path = join(dirname(dirname(abspath(__file__))), "data", "indiana-university")
        self.image_path = join(self.base_path, "images", "images_normalized")
        self.csv_path = join(self.base_path, "indiana_reports.csv")
        for file in os.listdir(self.image_path):
            self.list_IDs.append(file)
        self.labels = self.obtain_labels()
        self.on_epoch_end()
        self.tag_tokenizer, self.report_tokenizer = self.obtain_tokenizers()
        self.on_epoch_end()

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    def tokenize(self, lang, lang_tokenizer):
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor

    def preprocess_report(self, x, y, tag_tokenizer=None, report_tokenizer=None):
        x = [self.preprocess_sentence(i) for i in x]
        x = self.tokenize(x, tag_tokenizer)
        y = [self.preprocess_sentence(i) for i in y]
        y = self.tokenize(y, report_tokenizer)
        return x, y

    def obtain_tags(self, x):
        x = self.model(x)
        x_temp = []
        for row in x:
            tag_sentence = ""
            for tag_idx, score in enumerate(row):
                if score > 0.5:
                    tag_sentence += self.class_names[tag_idx] + " "
            x_temp.append(tag_sentence)
        return np.array(x_temp)

    def obtain_tokenizers(self):
        tag_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        report_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        for batch_idx in range(self.__len__()):
            tag, _, report = self.__getitem__(batch_idx, preprocess=False)
            tag = [self.preprocess_sentence(i) for i in tag]
            report = [self.preprocess_sentence(i) for i in report]
            tag_tokenizer.fit_on_texts(tag)
            report_tokenizer.fit_on_texts(report)
        return tag_tokenizer, report_tokenizer

    def obtain_labels(self):
        y = {}
        with open(join(self.csv_path)) as csv_file:
            line_count = 0
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    line_count += 1
                    y[row[0]] = row[6]
        return y
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index, preprocess=True):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, image_features, y = self.__data_generation(list_IDs_temp, preprocess)
        print(type(X), type(image_features), type(y))
        return X, image_features, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, preprocess):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = Image.open(join(self.image_path, ID)).resize(self.image_dimension)
            image_array = np.asarray(image.convert("RGB"))
            image_array = image_array / 255.
            X.append(image_array)
            y.append(self.labels[ID.split("_")[0]])
        X = np.array(X)
        X = self.augmenter.augment_images(X)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        X = (X - imagenet_mean) / imagenet_std
        vision_feature_model = K.function([self.model.layers[0].input],
                                          [self.model.layers[len(self.model.layers) - 2].output])
        image_features = np.squeeze(vision_feature_model(X))
        tags = self.obtain_tags(X)
        if preprocess:
            tags, y= self.preprocess_report(tags, y, self.tag_tokenizer, self.report_tokenizer)
        #print(f"Tags shape: {tags.shape}")
        #print(f"Train Y shape: {y.shape}")
        return tags, image_features, y
