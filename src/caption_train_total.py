import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from generator_indiana import DataGenerator
import time
from configparser import ConfigParser
from vision_model import get_model
from caption_models import Encoder, Decoder
from os.path import dirname, abspath, join
import os


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, features, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        # Pass input through model and tokenize
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, features)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(val_x, val_x_features):
    attention_plot = np.zeros((vocab_report_size, vocab_tag_size))
    input_sentence = ""
    for idx in val_x:
        if idx == 0:
            continue
        input_sentence += generator.tag_tokenizer.index_word[idx] + ' '
    
    result = ''
    val_x_features = np.expand_dims(val_x_features, axis=0)
    val_x = np.expand_dims(val_x, axis=0)
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(val_x, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']], 0)

    for t in range(vocab_report_size):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out,
                                                             val_x_features)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += generator.report_tokenizer.index_word[predicted_id] + ' '

        if generator.report_tokenizer.index_word[predicted_id] == '<end>':
            return result, input_sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, input_sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

cp = ConfigParser()
config_file = "./config.ini"
cp.read(config_file)
image_dimension = cp["TRAIN"].getint("image_dimension")
class_names = cp["DEFAULT"].get("class_names").split(",")
base_model_name = cp["DEFAULT"].get("base_model_name")
use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
output_dir = cp["DEFAULT"].get("output_dir")
output_weights_name = cp["TRAIN"].get("output_weights_name")


vision_model_path = join(dirname(dirname(abspath(__file__))), "outs", "output4", "best_weights.h5")
model = get_model(class_names, vision_model_path)
generator = DataGenerator(image_dimension, model, class_names)

BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_tag_size = len(generator.tag_tokenizer)+1
vocab_report_size = len(generator.report_tokenizer.word_index)+1

encoder = Encoder(vocab_tag_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_report_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
EPOCHS = 10

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for batch_idx in range(generator.__len__()):
        tag_features, image_features, y = generator.__getitem__()

        batch_loss = train_step(tag_features, image_features, y, enc_hidden)
        total_loss += batch_loss

        if batch_idx % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch_idx, batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    generator.on_epoch_end()
    print('Epoch {} Total Loss {:.4f}'.format(epoch + 1, total_loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


tag_features, image_features, y = generator.__getitem__()
for i in range(len(tag_features)):
    result, input_sentence, attention_plot = evaluate(tag_features[i], image_features[i])
    print(f"Input: {input_sentence}")
    print(f"Output: {result}")
    expected_output = ""
    for idx in y[i]:
        if idx == 0:
            continue
        else:
            expected_output += generator.report_tokenizer.index_word[idx] + " "
    print(f"Expected Outputs: {expected_output}")
