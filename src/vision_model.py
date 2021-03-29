from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet169

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

def get_model(class_names,
              weights_path,
              input_shape=(224, 224, 3)):

    img_input = Input(shape=input_shape)
    base_model = DenseNet169(
        include_top=False,
        input_tensor=img_input,
        input_shape=input_shape,
        weights=None,
        pooling="avg")
    x = base_model.output
    predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)

    model.load_weights(weights_path)
    return model

#model = get_model(class_names, "outs/output4/best_weights.h5")
#model.summary()
