import tensorflow as tf

import numpy as np
from generator_indiana import DataGenerator
import time
from configparser import ConfigParser
from vision_model import get_model
from caption_models import Encoder, Decoder
from os.path import dirname, abspath, join
import os
from nltk.translate.bleu_score import sentence_bleu


class_names = ['normal',
             'cardiomegaly',
             'pulmonary_artery',
             'pulmonary_disease',
             'bullous_emphysema',
             'pulmonary_fibrosis',
             'cicatrix',
             'opacity',
             'osteophyte',
             'thickening',
             'lung',
             'pulmonary_atelectasis',
             'spondylosis',
             'arthritis',
             'calcified_granuloma',
             'density',
             'cardiac_shadow',
             'markings',
             'granulomatous_disease',
             'pleural_effusion',
             'aorta',
             'thoracic_vertebrae',
             'breast_implants',
             'deformity',
             'sutures',
             'pulmonary_emphysema',
             'airspace_disease',
             'diaphragm',
             'consolidation',
             'pulmonary_congestion',
             'tube',
             'costophrenic_angle',
             'surgical_instruments',
             'emphysema',
             'no_indexing',
             'volume_loss',
             'lung_diseases',
             'stents',
             'nodule',
             'bone_and_bones',
             'spine',
             'scoliosis',
             'hernia',
             'mass',
             'calcinosis',
             'technical_quality_of_image_unsatisfactory_',
             'pneumothorax',
             'atherosclerosis',
             'kyphosis',
             'osteoporosis',
             'implanted_medical_device',
             'bone_diseases',
             'heart_failure',
             'shift',
             'foreign_bodies',
             'dislocations',
             'shoulder',
             'mediastinum',
             'catheters',
             'spinal_fusion',
             'infiltrate',
             'expansile_bone_lesions',
             'fractures',
             'lumbar_vertebrae',
             'diaphragmatic_eventration',
             'pulmonary_edema',
             'subcutaneous_emphysema',
             'pneumonia',
             'medical_device',
             'lucency',
             'sclerosis',
             'cysts',
             'granuloma',
             'funnel_chest',
             'epicardial_fat',
             'blister',
             'heart',
             'bronchiectasis',
             'mastectomy',
             'pneumoperitoneum',
             'aortic_aneurysm',
             'cervical_vertebrae',
             'heart_atria',
             'adipose_tissue',
             'trachea',
             'sulcus',
             'hypertension',
             'cystic_fibrosis',
             'humerus',
             'nipple_shadow',
             'hydropneumothorax',
             'pectus_carinatum',
             'fibrosis',
             'tuberculosis',
             'sarcoidosis',
             'colonic_interposition',
             'cholelithiasis',
             'ribs',
             'pleura',
             'hyperostosis',
             'heart_ventricles',
             'pneumonectomy',
             'pericardial_effusion',
             'bronchitis',
             'thorax',
             'contrast_media',
             'hypovolemia',
             'abdomen',
             'lymph_nodes',
             'cavitation',
             'hemopneumothorax',
             'subcutaneous__emphysema',
             'bronchiolitis',
             'blood_vessels',
             'hemothorax']


location_words = ["left",
                  "right",
                  "up",
                  "down",
                  "upper",
                  "lower",
                  "lateral",
                  "mid",
                  "anterior",
                  "proximal",
                  "superior",
                  "medial",
                  "distal",
                  "interior",
                  "ventral",
                  "cephalic",
                  "caudal",
                  "dorsal",
                  "posterior"]

severity_words = ["normal",
                  "increased",
                  "stable",
                  "suspicious",
                  "slight",
                  "worsening",
                  "chronic",
                  "small",
                  "moderate",
                  "minimal",
                  "unchanged",
                  "mild",
                  "significant",
                  "low",
                  "concern",
                  "advanced"]

def severity_fmeasure(label, predict):
    label_severity = [0] * len(severity_words)
    predict_severity = [0] * len(severity_words)
    for word in label:
        if word in severity_words:
            label_severity[severity_words.index(word)] = 1
    for word in predict:
        if word in severity_words:
            predict_severity[severity_words.index(word)] = 1
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(severity_words)):
        if label_severity[i] == 1 and predict_severity[i] == 1:
            tp += 1
        elif label_severity[i] == 0 and predict_severity[i] == 0:
            tn += 1
        elif label_severity[i] == 1 and predict_severity[i] == 0:
            fn += 1
        elif label_severity[i] == 0 and predict_severity[i] == 1:
            fp += 1
        else:
            print("SM ERROR")
    if tp + 0.5 * (fp + fn) == 0:
        # Division by 0 but tp is 0 anyways lol
        return 0
    return tp / (tp + 0.5 * (fp + fn))

def location_fmeasure(label, predict):
    label_locations = []
    predict_locations = []
    for word_index in range(len(label)):
        if label[word_index] in location_words:
            label_locations.append(label[word_index] + label[word_index+1])

    for word_index in range(len(predict)-1):
        if predict[word_index] in location_words:
            predict_locations.append(predict[word_index] + predict[word_index+1])

    tp, fp = 0, 0
    for location in predict_locations:
        if location in label_locations:
            label_locations.remove(location)
            tp += 1
        else:
            fp += 1
    fn = len(label_locations)

    if tp + 0.5 * (fp + fn) == 0:
        # Division by 0 but tp is 0 anyways lol
        return 0
    return tp / (tp + 0.5 * (fp + fn))


def keyword_fmeasure(label, predict):
    label_keywords = [0] * len(class_names)
    predict_keywords = [0] * len(class_names)
    for word in label:
        if word in class_names:
            label_keywords[class_names.index(word)] = 1
    for word in predict:
        if word in class_names:
            predict_keywords[class_names.index(word)] = 1
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(class_names)):
        if label_keywords[i] == 1 and predict_keywords[i] == 1:
            tp += 1
        elif label_keywords[i] == 0 and predict_keywords[i] == 0:
            tn += 1
        elif label_keywords[i] == 1 and predict_keywords[i] == 0:
            fn += 1
        elif label_keywords[i] == 0 and predict_keywords[i] == 1:
            fp += 1
        else:
            print("KW ERROR")
    if tp+0.5*(fp+fn) == 0:
        # print("KW: Division by 0!")
        # Division by 0 but tp is 0 anyways lol
        return 0
    return tp/(tp+0.5*(fp+fn))


# @tf.function
# def tf_predict(dec_in, dec_hidden, enc_out, val_x_features, end_tensor, start_tensor):
#     same_word = ""
#     same_count = 0
#     word_limit = 200

#     def cond(t, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, result):
#         if tf.equal(predicted_id, end_tensor):
#             return False
#         if t >= word_limit:
#             return False
#         return True

#     ta = tf.TensorArray(dtype=tf.int32, size=word_limit, infer_shape=True)

#     # ta.write(0, tf.constant("<start>", dtype=tf.string))
#     def body(t, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, ta):
#         predictions, dec_hidden, attention_weights = decoder(dec_in,
#                                                              dec_hidden,
#                                                              enc_out,
#                                                              val_x_features)
#         # storing the attention weights to plot later on
#         # attention_weights = tf.reshape(attention_weights, (-1, ))
#         # attention_plot[t] = attention_weights.numpy()
#         predicted_id = tf.argmax(predictions[0], output_type=tf.int32)  # .numpy()
#         # predicted_id = tf.make_ndarray(tf.argmax(predictions[0])) # tf friendly way
#         # pi = tf.get_static_value(predicted_id)
#         ta.write(t, predicted_id)

#         # the predicted ID is fed back into the model
#         dec_in = tf.expand_dims([predicted_id], 0)
#         return t + 1, dec_in, dec_hidden, enc_out, val_x_features, predicted_id, ta

#     _t, _di, _dh, _eo, _vx, _pw, result = tf.while_loop(
#         cond,
#         body,
#         (
#             tf.constant(0, dtype=tf.int32),
#             dec_in,
#             dec_hidden,
#             enc_out,
#             val_x_features,
#             start_tensor,
#             ta)
#     )
#     return result.stack()


# # def predict(val_x, val_x_features):
#     start = time.time()
#     attention_plot = np.zeros((vocab_report_size, vocab_tag_size))
#     input_sentence = []
#     for idx in val_x:
#         if idx == 0:
#             continue
#         input_sentence.append(generator.tag_tokenizer.index_word[idx])

#     val_x_features = np.expand_dims(val_x_features, axis=0)
#     val_x = np.expand_dims(val_x, axis=0)
#     hidden = [tf.zeros((1, units))]
#     enc_out, enc_hidden = encoder(val_x, hidden)
#     print(enc_hidden.shape)

#     dec_hidden = enc_hidden
#     dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']], 0)

#     se = generator.report_tokenizer.word_index['<start>']
#     start_tensor = tf.constant(se, dtype=tf.int32)

#     ee = generator.report_tokenizer.word_index['<end>']
#     end_tensor = tf.constant(ee, dtype=tf.int32)

#     result = tf_predict(dec_input, dec_hidden, enc_out, val_x_features, end_tensor, start_tensor)
#     indexes = tf.get_static_value(result)
#     word_result = []
#     for pid in indexes:
#         if pid == 0:
#             t = "."
#         else:
#             t = generator.report_tokenizer.index_word[pid]
#         word_result.append(t)

#     print("Predict took", time.time() - start)
#     return word_result, input_sentence, None

def old_predict(val_x, val_x_features):
    attention_plot = np.zeros((vocab_report_size, vocab_tag_size))
    input_sentence = ""
    for idx in val_x:
        if idx == 0:
            continue
        input_sentence += generator.tag_tokenizer.index_word[idx] + ' '
    
    result = []
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
        #attention_weights = tf.reshape(attention_weights, (-1, ))
        #attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result.append(generator.report_tokenizer.index_word[predicted_id])

        if generator.report_tokenizer.index_word[predicted_id] == '<end>':
            return result, input_sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, input_sentence, None

if __name__ == "__main__":
    start = time.time()
    print("Starting...")

    # Obtain Parameters
    cp = ConfigParser()
    config_file = "./config.ini"
    cp.read(config_file)
    image_dimension = cp["TRAIN"].getint("image_dimension")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    output_dir = cp["DEFAULT"].get("output_dir")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    BATCH_SIZE = 10 # > 16 will cause training to crash on David's local machine
    embedding_dim = 128
    units = 512
    LEARN_RATE = 0.0001
    FREEZE_VISION_MODEL = True # Freeze the model

    # Load CNN Model
    vision_model_path = join(dirname(dirname(abspath(__file__))), "outs", "outputindiana", "best_weights.h5")
    model =  get_model(class_names, vision_model_path)
    print('Time taken to inialize Vision Model {} sec\n'.format(time.time() - start))

    # Load Data Generator
    generator = DataGenerator(model.layers[0].input_shape[0], model, class_names, batch_size=BATCH_SIZE)
    print('Time taken to inialize Generator {} sec\n'.format(time.time() - start))
    vocab_tag_size = len(generator.tag_tokenizer.word_index)+1
    vocab_report_size = len(generator.report_tokenizer.word_index)+1
    print(f"Number of tags: {vocab_tag_size}\n Number of words: {vocab_report_size}")
    print('Time taken to inialize Classes {} sec\n'.format(time.time() - start))

    # Load Caption Model
    encoder = Encoder(vocab_tag_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_report_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARN_RATE)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    checkpoint_dir = 'testing'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    print("ENCODER", encoder.enc_units)
    print("DECODER", decoder.dec_units)
    # print(decoder.summary())
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.assert_existing_objects_matched()
    print('Time taken to Load Trained Encoder/decoder {} sec\n'.format(time.time() - start))
    location_f_measure, keyword_f_measure, severity_f_measure = 0, 0, 0
    bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score = 0, 0, 0, 0

    for batch_idx in range(generator.__testlen__()):
        tag_features, image_features, y = generator.__gettestitem__(batch_idx)
        print("input shapes", tag_features.shape, image_features.shape)
        for i in range(len(tag_features)):
            start = time.time()
            prediction, _, _ = old_predict(tag_features[i], image_features[i])
            expected_output = []
            for idx in y[i]:
                if idx == 0:
                    continue
                else:
                    expected_output.append(generator.report_tokenizer.index_word[idx])
            # print("Prediction:", prediction)
            # print("Expected:", expected_output)
            # Calculate Metrics
            location_f_measure += location_fmeasure(expected_output, prediction)
            keyword_f_measure += keyword_fmeasure(expected_output, prediction)
            severity_f_measure += severity_fmeasure(expected_output, prediction)
            bleu_1_score += sentence_bleu([expected_output], prediction, weights=(1, 0, 0, 0))
            bleu_2_score += sentence_bleu([expected_output], prediction, weights=(0.5, 0.5, 0, 0))
            bleu_3_score += sentence_bleu([expected_output], prediction, weights=(0.333, 0.333, 0.333, 0))
            bleu_4_score += sentence_bleu([expected_output], prediction, weights=(0.25, 0.25, 0.25, 0.25))
        
        if batch_idx > 10:
            break

    num_examples = generator.__testlen__() * generator.batch_size

    print(location_f_measure/num_examples)
    print(keyword_f_measure / num_examples)
    print(severity_f_measure / num_examples)
    print(bleu_1_score / num_examples)
    print(bleu_2_score / num_examples)
    print(bleu_3_score / num_examples)
    print(bleu_4_score / num_examples)