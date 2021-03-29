import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from data import load_indiana_data
import unicodedata
import re
import numpy as np
import os
import io
import time
from configparser import ConfigParser
from imgaug import augmenters as iaa
from vision_model import get_model
from os.path import dirname, abspath, join


class_names = ['No_Finding',
             'Enlarged_Cardiomediastinum',
             'Cardiomegaly',
             'Lung_Opacity',
             'Lung_Lesion',
             'Edema',
             'Consolidation',
             'Pneumonia',
             'Atelectasis',
             'Pneumothorax',
             'Pleural_Effusion',
             'Pleural_Other',
             'Fracture',
             'Support_Devices']


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

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


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def preprocess_report(y, class_names):
    y = [preprocess_sentence(i) for i in y]
    y, report_idx_to_word = tokenize(y)
    class_names.insert(0, "<start>")
    class_names.append("<end>")
    return y, report_idx_to_word, class_names


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        # Pass input through model and tokenize
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([report_idx_to_word.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(image):
    attention_plot = np.zeros((max_length_y, max_length_x))
    x = model(image)
    x_temp = []
    for row in x:
        idx = np.where(row == 1).tolist()
        idx.insert(0, 0)
        idx.append(len(tag_idx_to_word) - 1)
        idx += [len(tag_idx_to_word)] * (len(tag_idx_to_word) - len(idx) + 2)
        x_temp.append(idx)

    input_sentence = ""
    for idx in x_temp[0]:
        if idx >= len(tag_idx_to_word):
            input_sentence += "<buffer>"
        else:
            input_sentence += tag_idx_to_word[idx] + " "
    inputs = np.array(x_temp)
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([report_idx_to_word.word_index['<start>']], 0)

    for t in range(max_length_y):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += report_idx_to_word.index_word[predicted_id] + ' '

        if report_idx_to_word.index_word[predicted_id] == '<end>':
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


vision_model_path = join(dirname(abspath(__file__)), "outs", "output4", "best_weights.h5")
model = get_model(class_names, vision_model_path)

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
    ],
    random_order=True,
)
x, y = load_indiana_data((image_dimension, image_dimension), augmenter)
y, report_idx_to_word, tag_idx_to_word = preprocess_report(y, class_names)
x = model(x)
x_temp = []
for row in x:
    idx = np.where(row==1).tolist()
    idx.insert(0, 0)
    idx.append(len(tag_idx_to_word)-1)
    idx += [len(tag_idx_to_word)] * (len(tag_idx_to_word) - len(idx) + 2)
    x_temp.append(idx)
x = np.array(x_temp)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
max_length_x = x.shape[1]
max_length_y = y.shape[1]


BUFFER_SIZE = len(train_x)
BATCH_SIZE = 64
steps_per_epoch = len(train_x)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_tag_size = len(tag_idx_to_word)+1
vocab_report_size = len(report_idx_to_word.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

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

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden, model)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

result, input_sentence, attention_plot = evaluate(val_x[0])
print(f"Input: {input_sentence}")
print(f"Output: {result}")
print(f"Expected Outputs: {val_y[0]}")
attention_plot = attention_plot[:len(result.split(' ')), :len(input_sentence.split(' '))]
plot_attention(attention_plot, input_sentence.split(' '), result.split(' '))
