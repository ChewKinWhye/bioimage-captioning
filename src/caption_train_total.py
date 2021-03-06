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
import sys
from nltk.translate.bleu_score import sentence_bleu


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, features, targ, enc_hidden):
    def _body(t, loss, dec_input, dec_hidden):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, features)

        loss += loss_function(targ[:, t], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)
        return t+1, loss, dec_input, dec_hidden

    loss = 0
    with tf.GradientTape() as tape:
        # Pass input through model and tokenize
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        # dec_hidden = enc_hidden

        dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        
        _t, loss, _di, _dh = tf.while_loop(
            lambda t, _l, _di, _dh: t < targ.shape[1],
            _body,
            (1, tf.constant(0, dtype=tf.float32), dec_input, enc_hidden)
        )

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

@tf.function
def tf_predict(dec_in, dec_hidden, enc_out, val_x_features, end_tensor, start_tensor):
    same_word = ""
    same_count = 0
    word_limit = 200
    
    def cond(t, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, result):
        if tf.equal(predicted_id, end_tensor):
            return False
        if t >= word_limit:
            return False
        return True

    ta = tf.TensorArray(dtype=tf.int32, size=word_limit, infer_shape=True)
    # ta.write(0, tf.constant("<start>", dtype=tf.string))
    def body(t, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, ta):
        predictions, dec_hidden, attention_weights = decoder(dec_in,
                                                             dec_hidden,
                                                             enc_out,
                                                             val_x_features)
        # storing the attention weights to plot later on
        # attention_weights = tf.reshape(attention_weights, (-1, ))
        # attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0], output_type=tf.int32) #.numpy()
        # predicted_id = tf.make_ndarray(tf.argmax(predictions[0])) # tf friendly way
        # pi = tf.get_static_value(predicted_id)
        ta.write(t, predicted_id)
        
        # the predicted ID is fed back into the model
        dec_in = tf.expand_dims([predicted_id], 0)
        return t+1, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, ta

    _t, _di, _dh, _eo, _vx, _pw, result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, dtype=tf.int32), 
            dec_in, 
            dec_hidden, 
            enc_out, 
            val_x_features, 
            start_tensor,
            ta)
    )
    return result.stack()

def predict(val_x, val_x_features):
    start = time.time()
    attention_plot = np.zeros((vocab_report_size, vocab_tag_size))
    input_sentence = []
    for idx in val_x:
        if idx == 0:
            continue
        input_sentence.append(generator.tag_tokenizer.index_word[idx])
    
    val_x_features = np.expand_dims(val_x_features, axis=0)
    val_x = np.expand_dims(val_x, axis=0)
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(val_x, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']], 0)
    
    se = generator.report_tokenizer.word_index['<start>']
    start_tensor =  tf.constant(se, dtype=tf.int32)

    ee = generator.report_tokenizer.word_index['<end>']
    end_tensor = tf.constant(ee, dtype=tf.int32)

    result = tf_predict(dec_input, dec_hidden, enc_out, val_x_features, end_tensor, start_tensor)
    indexes = tf.get_static_value(result)
    word_result = []
    for pid in indexes:
        if pid == 0:
            t = "."
        else:
            t = generator.report_tokenizer.index_word[pid]
        word_result.append(t)
        
    # for t in range(vocab_report_size):
    #     predictions, dec_hidden, attention_weights = decoder(dec_input,
    #                                                          dec_hidden,
    #                                                          enc_out,
    #                                                          val_x_features)

    #     # storing the attention weights to plot later on
    #     #attention_weights = tf.reshape(attention_weights, (-1, ))
    #     #attention_plot[t] = attention_weights.numpy()

    #     predicted_id = tf.argmax(predictions[0]).numpy()

    #     predicted_word = generator.report_tokenizer.index_word[predicted_id]
    #     result.append(predicted_word)

    #     if predicted_word == '<end>':
    #         return result, input_sentence, attention_plot

    #     # the predicted ID is fed back into the model
    #     dec_input = tf.expand_dims([predicted_id], 0)

    print("Predict took", time.time() - start)
    return word_result, input_sentence, None

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


def evaluate():
    bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score = 0, 0, 0, 0
    for batch_idx in range(generator.__testlen__()):
        tag_features, image_features, y = generator.__gettestitem__(batch_idx)
        for i in range(len(tag_features)):
            start = time.time()
            prediction, _, _ = predict(tag_features[i], image_features[i])
            expected_output = []
            for idx in y[i]:
                if idx == 0:
                    continue
                else:
                    expected_output.append(generator.report_tokenizer.index_word[idx])
            bleu_1_score += sentence_bleu([expected_output], prediction, weights=(1, 0, 0, 0))
            bleu_2_score += sentence_bleu([expected_output], prediction, weights=(0.5, 0.5, 0, 0))
            bleu_3_score += sentence_bleu([expected_output], prediction, weights=(0.333, 0.333, 0.333, 0))
            bleu_4_score += sentence_bleu([expected_output], prediction, weights=(0.25, 0.25, 0.25, 0.25))
        print(f"Predicted: {prediction}")
        print(f"Expected: {expected_output}")
    return bleu_1_score/len(tag_features), bleu_2_score/len(tag_features), \
           bleu_3_score/len(tag_features), bleu_4_score/len(tag_features)

# CODE STARTS HERE
# Define Parameters
start = time.time()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

# # To test GPU usage
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

BATCH_SIZE = 10 # > 16 will cause training to crash on David's local machine
embedding_dim = 256
units = 1024
FREEZE_VISION_MODEL = True # Freeze the model
EPOCHS = 20
LEARN_RATE = 0.001

args = sys.argv[1:]
for idx, arg in enumerate(args):
    if arg == "-bs":
        BATCH_SIZE = int(args[idx+1])
        print("Batch size set to", BATCH_SIZE)

    if arg == "-tv":
        FREEZE_VISION_MODEL = False
    
    if arg == "-e":
        EPOCHS = min(int(args[idx+1]),200)

    if arg == "-lr":
        LEARN_RATE = min(float(args[idx+1]),0.01)
    
    if arg == "-u":
        units = min(int(args[idx+1]), 1024)
    
    if arg == "-em":
        embedding_dim = int(args[idx+1])
    
print("Starting...")
cp = ConfigParser()
print("Batch size: {} Epochs: {} Learn rate: {} Freeze Vision Model: {} Encoder/Decoder Units {}".format(BATCH_SIZE, EPOCHS, LEARN_RATE, FREEZE_VISION_MODEL, units))
print('Time taken to inialize CP {} sec\n'.format(time.time() - start))

config_file = "./config.ini"
cp.read(config_file)
image_dimension = cp["TRAIN"].getint("image_dimension")
class_names = cp["DEFAULT"].get("class_names").split(",")
base_model_name = cp["DEFAULT"].get("base_model_name")
use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
output_dir = cp["DEFAULT"].get("output_dir")
output_weights_name = cp["TRAIN"].get("output_weights_name")


# vision_model_path = join(dirname(dirname(abspath(__file__))), "outs", "output4", "best_weights.h5")

print("Num_classes", len(class_names))

vision_model_path = join(dirname(dirname(abspath(__file__))), "outs", "outputindiana", "best_weights.h5")
model = get_model(class_names, vision_model_path)

model.trainable = not FREEZE_VISION_MODEL 
# model.summary() # Debug
print('Time taken to inialize Vision Model {} sec\n'.format(time.time() - start))

generator = DataGenerator(model.layers[0].input_shape[0], model, class_names, batch_size=BATCH_SIZE)
print('Time taken to inialize Generator {} sec\n'.format(time.time() - start))

vocab_tag_size = len(generator.tag_tokenizer.word_index)+1
vocab_report_size = len(generator.report_tokenizer.word_index)+1
print(f"Number of tags: {vocab_tag_size}\n Number of words: {vocab_report_size}")
# vocab_report_size = generator.report_max_length + 1
print('Time taken to inialize Classes {} sec\n'.format(time.time() - start))
encoder = Encoder(vocab_tag_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_report_size, embedding_dim, units, BATCH_SIZE)

print('Time taken to inialize Encoder/decoder {} sec\n'.format(time.time() - start))

optimizer = tf.keras.optimizers.Adam(learning_rate = LEARN_RATE)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

print("Training has started! {}".format(time.time() - start))
batches_per_epoch = generator.__len__()
print("Batches per epoch {}".format(batches_per_epoch))

PRINT_PER_EPOCH = 2
for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    count = 0
    for batch_idx in range(generator.__len__()):
        tag_features, image_features, y = generator.__getitem__(batch_idx)
        batch_loss = train_step(tag_features, image_features, y, enc_hidden)
        total_loss += batch_loss

        count += BATCH_SIZE
        if batch_idx % (batches_per_epoch//PRINT_PER_EPOCH) == 0 or True:
            print('Epoch {} Batch {} Batch Loss {:.4f}'.format(epoch + 1, batch_idx, batch_loss.numpy()))
            result, _, _ = predict(tag_features[0], image_features[0])
            print("Result {}".format(result))

    # Now training is fast enough, we print once per several epochs
    
    
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    
    if epoch > 10 and (epoch + 1) % 10 or True:
        print(evaluate())


    generator.on_epoch_end()
    print("Counted: %s" % count)
    print('Epoch {} Total Loss {:.4f}'.format(epoch + 1, total_loss))
    print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))
