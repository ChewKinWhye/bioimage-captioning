import json
import shutil
import os
import pickle
import pandas as pd
import numpy as np

from callback import MultipleClassAUROC, MultiGPUModelCheckpoint

from configparser import ConfigParser

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
#from tensorflow.keras.utils import multi_gpu_model

import importlib

from imgaug import augmenters as iaa

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import DenseNet121, DenseNet169, ResNet152V2, VGG19
from tensorflow.keras.optimizers import Adam ## ok

from generator import AugmentedImageSequence

class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None, input_shape=None):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None
        '''
        base_model_class = getattr(
            importlib.import_module(
                "keras.applications." + self.models_[model_name]['module_name']
            ),
            model_name)
        '''
        input_shape = (256, 256, 3)
        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = tf.keras.layers.Input(shape=input_shape)

        '''
        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        '''
        base_model = DenseNet169(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print("load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model

def get_sample_counts(output_dir, dataset, class_names):
    df = pd.read_csv(os.path.join(output_dir, dataset + ".csv"))
    df = df.fillna(0)
    total_count = df.shape[0]
    labels = df[class_names].values
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))

    return total_count, class_positive_counts

def get_class_weights(total_counts, class_positive_counts, multiply):
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }
    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = {}
    for i, class_name in enumerate(class_names):
        class_weights[i] = get_single_class_weight(label_counts[i], total_counts)[1]
    return class_weights

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
    ],
    random_order=True,
)

def main():
    # parser config
    print("### Input configuration file ### \n")
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    print("### Read default configurations ### \n")
    output_dir = cp["DEFAULT"].get("output_dir")
    image_train_source_dir = cp["DEFAULT"].get("image_train_source_dir")
    image_valid_source_dir = cp["DEFAULT"].get("image_valid_source_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # train config
    print("### Reading training configurations ### \n")
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    min_lr = cp["TRAIN"].getfloat("min_lr")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")

    # if previously trained weights is used, never re-split
    if use_trained_model_weights:
        # resuming mode
        print("** use trained model weights **")
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            # TODO: add loading previous learning rate?
            training_stats = json.load(open(training_stats_file))
        else:
            training_stats = {}
    else:
        # start over
        training_stats = {}

    print("### Show model summary ### \n")
    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    # end parser config

    print("### Check output directory ### \n")
    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()
    try:
        print("### Backup config file to {} \n".format(output_dir))
        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        datasets = ["train", "valid"]
        for dataset in datasets:
            shutil.copy(os.path.join(dataset_csv_dir, dataset + '.csv'), output_dir)

        # get train/dev sample counts
        print("### Get class frequencies ### \n")
        train_counts, train_pos_counts = get_sample_counts(output_dir, "train", class_names)
        dev_counts, _ = get_sample_counts(output_dir, "valid", class_names)

        # compute steps
        print("### Compute step size ### \n")
        train_steps = int(train_counts / batch_size)
        validation_steps = int(dev_counts / batch_size)

        # compute class weights
        print("### Class weights ### \n")
        class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )
        print("### Class_weights ### \n")
        print(class_weights)
        print("\n")


        print("### Loading model ### \n")
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(output_dir, "best_" + output_weights_name)
            else:
                model_weights_file = os.path.join(output_dir, output_weights_name)
        else:
            model_weights_file = None


        model_factory = ModelFactory()
        print("### Get model ### \n")
        model = model_factory.get_model(
            class_names,
            model_name=base_model_name,
            use_base_weights=use_base_model_weights,
            weights_path=model_weights_file,
            input_shape=(image_dimension, image_dimension, 3))

        print("Show model summary? {}".format(show_model_summary))
        if show_model_summary:
            print(model.summary())

        print("\n ### Create image generators ### \n")
        train_sequence = AugmentedImageSequence(
            dataset_csv_file=os.path.join(output_dir, "train.csv"),
            class_names=class_names,
            source_image_dir=image_train_source_dir,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            steps=train_steps,
        )
        validation_sequence = AugmentedImageSequence(
            dataset_csv_file=os.path.join(output_dir, "valid.csv"),
            class_names=class_names,
            source_image_dir=image_valid_source_dir,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            steps=validation_steps,
            shuffle_on_epoch_end=False,
        )

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print("### Set output weights path to {} ### \n".format(output_weights_path))

        print("### Check multiple gpu availability ### \n")
        #gpus = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
        if False: ## Turn off multiple gpu model
            print("### Multi_gpu_model is used! gpus={} ###".format(gpus))
            model_train = multi_gpu_model(model, gpus)
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            checkpoint = MultiGPUModelCheckpoint(
                filepath=output_weights_path,
                base_model=model,
            )
        else:
            model_train = model
            checkpoint = ModelCheckpoint(
                 output_weights_path,
                 save_weights_only=True,
                 save_best_only=True,
                 verbose=1,
            )

        print("### Compile model with class weights ### \n")
        optimizer = Adam(lr=initial_learning_rate)
        model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        auroc = MultipleClassAUROC(
            sequence=validation_sequence,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers,
        )
        callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(output_dir, "logs")),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=patience_reduce_lr,
                verbose=1,
                mode="min",
                min_lr=min_lr),
            auroc,
        ]

        print("### Start training ### \n")

        history = model_train.fit(
            train_sequence,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_sequence,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=generator_workers,
            shuffle=False,
        )

        # dump history
        print("### Dump history ### \n")
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
                "auroc": auroc.aurocs,
            }, f)
        print("** done! **")


    finally:
        os.remove(running_flag_file)


if __name__ == "__main__":
    main()
