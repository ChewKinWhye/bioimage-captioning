import tensorflow as tf

import numpy as np
from generator_indiana import DataGenerator
import time
from configparser import ConfigParser
from vision_model import get_model
from caption_models import Encoder, Decoder
from e2e_models import ImageDecoder
from os.path import dirname, abspath, join
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk import tokenize
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')


use_gensim = False
if use_gensim:
    from gensim.models import Doc2Vec


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


def sentence_embedding_cosine_similarity(label, predict):
    total_cosine_similarity = 0
    total_counts = 0

    # Split by sentences
    label_sentences = tokenize.sent_tokenize(label)
    predict_sentences = tokenize.sent_tokenize(predict)

    # Use pretrained model to obtain encoding
    label_sentences_embeddings = sbert_model.encode(label_sentences)
    predict_sentences_embeddings = sbert_model.encode(predict_sentences)

    # For each sentence embedding in label
    for label_embedding in label_sentences_embeddings:
        # Find the closest embedding in the prediction
        max_cosine_score = 0
        for predict_embedding in predict_sentences_embeddings:
            max_cosine_score = max(max_cosine_score, cosine_similarity([label_embedding], [predict_embedding])[0][0])
        # Add to the average_score
        total_cosine_similarity += max_cosine_score
        total_counts += 1

    # Repeat for the predictions
    for predict_embedding in predict_sentences_embeddings:
        # Find the closest embedding in the prediction
        max_cosine_score = 0
        for label_embedding in label_sentences_embeddings:
            max_cosine_score = max(max_cosine_score, cosine_similarity([label_embedding], [predict_embedding])[0][0])
        # Add to the average_score
        total_cosine_similarity += max_cosine_score
        total_counts += 1
    return total_cosine_similarity/total_counts

def paragraph_embedding_cosine_similarity(label, predict, doc_2_vec_model):
    # Load Model
    label_embedding = doc_2_vec_model.infer_vector(label)
    predict_embedding = doc_2_vec_model.infer_vector(predict)
    return cosine_similarity(label_embedding, predict_embedding)

def scores(tp, tn, fp, fn):
    if tp + 0.5 * (fp + fn) == 0:
        # All true negatives align
        # # print("SM division by 0")
        # print("Label", label)
        # print("Predict", predict[:30])
        acc = 1
    else:
        acc = tp/(tp+0.5*(fp+fn))
    
    if tn + 0.5 * (fp + fn) == 0:
        # All true negatives align
        # # print("SM division by 0")
        # print("Label", label)
        # print("Predict", predict[:30])
        recall = 1
    else:
        recall = tn/(tn+0.5*(fp+fn))
    return acc, recall
    

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
    
    return scores(tp, tn, fp, fn)

def location_fmeasure(label, predict):
    label_locations = []
    predict_locations = []
    for word_index in range(len(label)):
        if label[word_index] in location_words:
            label_locations.append(label[word_index] + " " + label[word_index+1])

    for word_index in range(len(predict)-1):
        if predict[word_index] in location_words:
            predict_locations.append(predict[word_index] + predict[word_index+1])
    print(label)
    print(predict)
    print(label_locations)
    print(predict_locations)
    tp, fp = 0, 0
    for location in predict_locations:
        if location in label_locations:
            label_locations.remove(location)
            tp += 1
        else:
            fp += 1
    fn = len(label_locations)

    if tp + 0.5 * (fp + fn) == 0:
        return 1
    print(tp/(tp + 0.5 * (fp + fn)))
    return tp/(tp + 0.5 * (fp + fn))


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
    return scores(tp, tn, fp, fn)


def e2e_predict(val_x_features):
    result = []
    img_x_in = np.expand_dims(val_x_features, axis=0)
   
    dec_input = tf.expand_dims([generator.report_tokenizer.word_index['<start>']], 0)

    for t in range(vocab_report_size):
        predictions, dec_hidden = decoder(dec_input, img_x_in)

        # storing the attention weights to plot later on
        #attention_weights = tf.reshape(attention_weights, (-1, ))
        #attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result.append(generator.report_tokenizer.index_word[predicted_id])

        if generator.report_tokenizer.index_word[predicted_id] == '<end>':
            return result

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

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

def list2string(l):
    s = ""
    for word in l:
        s = s + " " + word
    return s[1:] # Take out first space

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

    END_TO_END = False
    if not END_TO_END:
        # Load Caption Model
        embedding_dim = 128
        units = 512
        encoder = Encoder(vocab_tag_size, embedding_dim, units, BATCH_SIZE)
        decoder = Decoder(vocab_report_size, embedding_dim, units, BATCH_SIZE)
        optimizer = tf.keras.optimizers.Adam(learning_rate = LEARN_RATE)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_dir = 'testing'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        encoder=encoder,
                                        decoder=decoder)

        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()    
        print("Using Combined model")

    else:
        # Load End to End Model
        embedding_dim = 256
        units = 1024
        decoder = ImageDecoder(vocab_report_size, embedding_dim, units, BATCH_SIZE)
        optimizer = tf.keras.optimizers.Adam(learning_rate = LEARN_RATE)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_dir = 'e2e_testing'
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        decoder=decoder)

        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()
        print("Using End 2 End (image features only) model")


    print('Time taken to Load Trained Encoder/decoder {} sec\n'.format(time.time() - start))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    loc_a, loc_r, kw_a, kw_r, s_a, s_r = 0, 0, 0, 0, 0, 0
    bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score = 0, 0, 0, 0
    rouge_precision = 0
    rouge_recall = 0
    rouge_fmeasure = 0
    meteor_score = 0
    sentence_embedding_score = 0
    paragraph_embedding_cosine_score = 0
    if use_gensim: doc_2_vec_model = Doc2Vec.load(join("enwiki_dbow", "doc2vec.bin"))

    for batch_idx in range(generator.__testlen__()):
        tag_features, image_features, y = generator.__gettestitem__(batch_idx)
        # print("input shapes", tag_features.shape, image_features.shape)
        for i in range(len(tag_features)):
            start = time.time()
            if END_TO_END:
                prediction = e2e_predict(image_features[i])
            else:
                prediction, _, _ = old_predict(tag_features[i], image_features[i])
            expected_output = []
            for idx in y[i]:
                if idx == 0:
                    continue
                else:
                    expected_output.append(generator.report_tokenizer.index_word[idx])
            
            # Ignore <start>. But don't ignore <end>
            expected_output = expected_output[1:]

            ps = list2string(prediction)
            es = list2string(expected_output)
            # print(ps)
            # print(es)
            # print("Pred str", ps[:150])
            # print("Ex str", es)
            
            # Calculate Metrics
            loc_a += location_fmeasure(expected_output, prediction)
            kwfm = keyword_fmeasure(expected_output, prediction)
            kw_a += kwfm[0]
            kw_r += kwfm[1]
            sfm = severity_fmeasure(expected_output, prediction)
            s_a += sfm[0]
            s_r += sfm[1]

            sentence_embedding_score += sentence_embedding_cosine_similarity(es, ps)
            bleu_1_score += sentence_bleu([expected_output], prediction, weights=(1, 0, 0, 0))
            bleu_2_score += sentence_bleu([expected_output], prediction, weights=(0.5, 0.5, 0, 0))
            bleu_3_score += sentence_bleu([expected_output], prediction, weights=(0.333, 0.333, 0.333, 0))
            bleu_4_score += sentence_bleu([expected_output], prediction, weights=(0.25, 0.25, 0.25, 0.25))
            rogue_out = scorer.score(es, ps)['rouge1']
            rouge_precision += rogue_out.precision
            rouge_recall += rogue_out.recall
            rouge_fmeasure += rogue_out.fmeasure
            meteor_score += nltk_meteor_score(es, ps)
            if use_gensim: 
                paragraph_embedding_cosine_score += paragraph_embedding_cosine_similarity(expected_output, prediction,
                                                                                      doc_2_vec_model)
        if batch_idx > 9 and batch_idx % 10 == 0:
            print("Batch", batch_idx, "Examples seen:", batch_idx*BATCH_SIZE)
            
    num_examples = generator.__testlen__() * generator.batch_size

    results = []
    results.append(("Number of examples:", num_examples))
    results.append(("Location f measure", loc_a/num_examples))
    # results.append(("Location f measure recall", loc_r/num_examples))
    results.append(("Keyword f measure", kw_a / num_examples))
    results.append(("Keyword recall", kw_r / num_examples))
    results.append(("Severity f measure", s_a / num_examples))
    results.append(("Severity recall", s_r / num_examples))
    results.append(("Sentence Embedding Cosine Similarity", sentence_embedding_score / num_examples))
    results.append(("ROGUE precision", rouge_precision / num_examples))
    results.append(("ROGUE recall", rouge_recall / num_examples))
    results.append(("ROGUE f measure", rouge_fmeasure / num_examples))
    results.append(("METEOR SCORE", meteor_score / num_examples))
    results.append(("BLEU 1", bleu_1_score / num_examples))
    results.append(("2", bleu_2_score / num_examples))
    results.append(("3", bleu_3_score / num_examples))
    results.append(("4", bleu_4_score / num_examples))
    if use_gensim: results.append(("Paragraph Embedding Cosine Similarity", paragraph_embedding_cosine_score / num_examples))
    result_filename = "Combined Model Results.txt"
    if END_TO_END:
        result_filename = "End2End Model Results.txt"
    with open(result_filename, "w+") as f:
        for r in results:
            f.write(r[0] + ": " + str(r[1]) + "\n")
    print("DONE, wrote results to %s" % result_filename)
