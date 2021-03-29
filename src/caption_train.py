import numpy as np
import tensorflow as tf
from PIL import Image # Image library

from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Concatenate, Reshape, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


# from keras.layers.merge import add # add([x,y]) seems it just takes a scalar addition of


# Images can be found on the google docs folder under Sample Data
# Picked 2 random samples
# image = Image.open('s1001.png')
# s1_front = np.asarray(image)
# print(s1_front.shape) # 2496 x 2048 Pixels (Front) wider

# image = Image.open('s2001.png')
# s1_side = np.asarray(image)
# print(s1_side.shape) # 2048 x 2496 Pixels (Side) taller

vocabulary_size = 200

s1_tags = "Calcified Granuloma/lung/upper lobe/right"
s1_label = "The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality."

start_token_vector = np.zeros(vocabulary_size) # TODO: Make this dynamic by calling Embedding["<START>"] or something

# Defining the custom Word RNN
# Taken from TF tutorial on Text Generation
class WordRNN(tf.keras.Model):
  def __init__(self, rnn_units, max_sent_len=20, batch_size=16, vocab_size=vocabulary_size):
    super().__init__(self)
    self.msl = max_sent_len
    self.vocab_size = vocab_size
    self.start_token_vec = start_token_vector
    # This should really be stateful but I can't get it to recognize input shapes
    self.lstm = LSTM(rnn_units,
                    stateful = False,
                    batch_input_shape= [batch_size,1,vocab_size],
                    return_sequences=False, 
                    return_state=True)
    self.dense = Dense(vocab_size, activation='softmax')

  def compute_output_shape(self, in_shape):
    print("COMPUTE SHAPE", in_shape.as_list())
    bs = in_shape.as_list()[0] # Batch size
    return (in_shape.as_list()[0],21,in_shape.as_list()[-1])

  def call(self, inputs, training=False, verbose=False):
    if verbose: print("Input shape", tf.shape(inputs))
    bs = tf.shape(inputs)[0] # Batch size
    # in_vec_size = tf.shape(inputs)[-1]
    in_vec_size = self.vocab_size
    if verbose: print("Batch size", bs, "vec size", in_vec_size)

    # Shape is (batch_size(32;64), sequence_len=21, vector_size=vocab_size)
    # Shape padding
    # seq = tf.zeros((bs,self.msl-1,in_vec_size))
    batch_one_vocab = (-1,1,in_vec_size)
    
    # Shape input
    r_inputs = tf.reshape(inputs, batch_one_vocab)
    # r_inputs = tf.transpose(r_inputs, [1,0,2])
    if verbose: print("reshaped_input", tf.shape(r_inputs))
    
    # Shape start_token
    start_token = tf.convert_to_tensor(self.start_token_vec, dtype=tf.float32)
    rst = tf.reshape(start_token, [1,1,in_vec_size])
    tst = tf.tile(rst, [bs,1,1])
    # tst = tf.transpose(tst, [1,0,2])
    if verbose: print("tiled start_token", tf.shape(tst),"\n")

    # Create x_seq and tensor array
    # x_seq = tf.concat((r_inputs, tst, seq), 1)
    ta = tf.TensorArray(dtype=tf.float32, size=self.msl+1,
                     dynamic_size=False, infer_shape=True)
    
    # LSTM(topic_vector) --> NOT USED
    _, topic_state, _ = self.lstm(r_inputs, training=training)
    # LSTM(s0) --> s1
    first_x, state,_ = self.lstm(tst, training=training)
    first_x = self.dense(first_x, training=training)
    first_x = tf.expand_dims(first_x,1)
    ta.write(0,first_x)

    def _recur(i, x_in, buffer):
      if verbose: print("i", i)
      if verbose: print("xin", tf.shape(x_in))
      x, s, _carry = self.lstm(x_in, training=training)
      if verbose: print("x", tf.shape(x))
      next_word = self.dense(x, training=training)
      if verbose: print("word shape", tf.shape(next_word))
      next_word = tf.expand_dims(next_word,1)
      if verbose: print("word shape exdim", tf.shape(next_word))
      buffer = buffer.write(i,next_word)      
      return i+1, next_word, buffer

    start_index = 1
    _, _, out = tf.while_loop(
                  cond=lambda i, _1, _2: i < self.msl,
                  body=_recur, 
                  loop_vars=(
                      tf.constant(start_index,dtype=tf.int32),
                      first_x,
                      ta
                    )
                  )
    out_t = out.stack()
    if verbose: print("out shape", tf.shape(out_t))
    out_t = tf.reshape(out_t,(-1,tf.shape(out_t)[1],tf.shape(out_t)[-1]))

    return out_t

vocabulary_size = 200
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
sample_tag = "Calcified Granuloma/lung/upper lobe/right"
encoded_sample = one_hot(sample_tag, vocabulary_size)
print(encoded_sample)

word_vec_size = 128
max_tag_length = 32
max_sents = 6 # As defined in the research paper
para_dims = 128 # chosen arbitrarily

######## Model Building ########
dense_activ = "relu"
lstm_units = 512
word_rnn = WordRNN(lstm_units)

# Input of the pic_in input should be a hidden layer of the Image processing model
cnn_output = 2048 
pic_in = Input(shape=(cnn_output,))
cm = Dense(256, activation=dense_activ)(pic_in)

# Process Tags
tag_in = Input(shape=(max_tag_length,)) # For testing
tm = Embedding(vocabulary_size, output_dim=word_vec_size, mask_zero=True)(tag_in)
tm = LSTM(units=128)(tm)

# Combine tag hidden layer and convolutional hidden layer
combined = Concatenate()([cm, tm]) 
combined_hidden = Dense(max_sents*para_dims, activation = dense_activ)(combined)
combined_hidden = Reshape((max_sents, para_dims))(combined_hidden)
# We want 6 sentence vectors
sentence_out = LSTM(lstm_units, return_sequences=True)(combined_hidden) 

stop_para = TimeDistributed(Dense(2, activation='softmax'))(combined_hidden) # Softmax sums to 1

# Time distributed applies the layer at every timeslice. Convenient
topic = TimeDistributed(Dense(512, activation='relu'))(sentence_out) # Hidden topic layer
# The topic vector must be the same size as a word vector
topic = TimeDistributed(Dense(vocabulary_size, activation='relu'))(topic) 
print(topic.shape)
words = TimeDistributed(word_rnn)(topic)

# summarize model
model = Model(inputs=[pic_in, tag_in],outputs=[stop_para,words])
model.compile(loss='categorical_crossentropy', optimizer='adam')
# summarize model
print(model.summary())

# plot_model(model, to_file='./model.png', show_shapes=True)


# Testing the pipeline
npt = pad_sequences([encoded_sample], maxlen=max_tag_length)
npt.reshape(32)
random_pic_in = np.array([np.random.rand(cnn_output)])
print(random_pic_in.shape, npt.shape)
print("model in", model.input)
p = model.predict([random_pic_in, npt], verbose=1)
print(p)

