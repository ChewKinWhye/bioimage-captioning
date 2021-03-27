import numpy as np
import tensorflow as tf
from PIL import Image # Image library

from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Concatenate
from keras.models import Model,Input
from keras.utils import plot_model
from keras.preprocessing.text import one_hot

# from keras.layers.merge import add # add([x,y]) seems it just takes a scalar addition of


# Images can be found on the google docs folder under Sample Data
# Picked 2 random samples
# image = Image.open('s1001.png')
# s1_front = np.asarray(image)
# print(s1_front.shape) # 2496 x 2048 Pixels (Front) wider

# image = Image.open('s2001.png')
# s1_side = np.asarray(image)
# print(s1_side.shape) # 2048 x 2496 Pixels (Side) taller

s1_tags = "Calcified Granuloma/lung/upper lobe/right"
s1_label = "The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality."


vocabulary_size = 200
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
sample_tag = "Calcified Granuloma/lung/upper lobe/right"
encoded_sample = one_hot(sample_tag, vocabulary_size)
print(encoded_sample)

word_vec_size = 128
max_tag_length = 32

######## Model Building ########
dense_activ = "relu"
# Input of the pic_in input should be a hidden layer of the Image processing model
cnn_output = 2048 
pic_in = Input(shape=(cnn_output,))
cm = Dense(256, activation=dense_activ)(pic_in)

# a = Input(shape=(max_tag_length))
tag_in = Input(shape=(max_tag_length,)) # For testing
tm = Embedding(vocabulary_size, output_dim=word_vec_size, mask_zero=True)(tag_in)
tm = LSTM(units=128)(tm)

# Visual attention uses Feed-forward aka Dense layers?
# b = Conv2D(filters=8, kernel_size=2,strides=1,activation='relu')(a)
combined = Concatenate()([cm, tm])  
final_out = Dense(256, activation = dense_activ)(combined)
# As of now I'm putting it as a one-hot encoding across vocabulary to predict word. 
# Not sure how to build the topic/sentence output. Still reading
final_out = Dense(vocabulary_size, activation='softmax')(final_out) # Softmax sums to 1
model = Model(inputs=[pic_in, tag_in],outputs=final_out)

model.compile(loss='categorical_crossentropy', optimizer='adam')
# summarize model
print(model.summary())
plot_model(model, to_file='./model.png', show_shapes=True)


# Testing the pipeline
npt = pad_sequences([encoded_sample], maxlen=max_tag_length)
npt.reshape(32)
random_pic_in = np.array([np.random.rand(cnn_output)])
print(random_pic_in.shape, npt.shape)
print("model in", model.input)
p = model.predict([random_pic_in, npt], verbose=1)
print(p)

