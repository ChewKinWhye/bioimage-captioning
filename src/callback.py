import json
import tensorflow.keras.backend as kb
import numpy as np
import os
import pickle
import shutil
import warnings
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score


class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, sequence, class_names, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_{}".format(os.path.split(weights_path)[1]),
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.best_preds_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_preds.pkl",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.learning_rate))
        print("Current learning rate: {}".format(self.stats['lr']))

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        ## Use model.predict - predict_generator is deprecated
        y_hat = self.model.predict(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()

        print("***Epoch {} ***".format(epoch+1))
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                preds = [round(x) for x in y_hat[:,i]]
                score = accuracy_score(y[:, i], preds)
            except ValueError:
                print("\n Ground truth: \n")
                print(y[:, i])
                print("\n Predicted: \n")
                print(y_hat[: , i])
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print("{}. {}: {}".format(i+1, self.class_names[i], score))
        print("*********************************")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print("Mean auroc: {}".format(mean_auroc))
        if mean_auroc > self.stats["best_mean_auroc"]:
            print("Update best auroc from {} to {}".format(self.stats['best_mean_auroc'], mean_auroc))

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print("Update log file: {}".format(self.best_auroc_log_path))
            with open(self.best_auroc_log_path, "a") as f:
                f.write("(Epoch: {}) Auroc: {}, LR: {}\n".format(epoch + 1, mean_auroc, self.stats['lr']))

            # 3. update predications
            print("Updating predictions: {}".format(self.best_preds_log_path))
            with open(self.best_preds_log_path, "wb") as f:
                pickle.dump({
                    "y": y,
                    "yhat": y_hat,
                }, f)

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print("Update model file: {} -> {}".format(self.weights_path, self.best_weights_path))
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return


class MultiGPUModelCheckpoint(Callback):
    """
    Checkpointing callback for multi_gpu_model
    copy from https://github.com/keras-team/keras/issues/8463
    """
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)
