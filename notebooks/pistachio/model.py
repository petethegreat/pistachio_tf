'''
model.py
utils for building models
'''

from typing import List, Dict
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Normalization
from tensorflow.keras import Model
# from tensorflow.keras.metrics import Accuracy, AUC, Recall, Precision
import tensorflow as tf

def get_pistachio_model(
    feature_columns: List[str],
    train_dataset: tf.data.Dataset,
    units: int=10,
    layer_l1_reg: float=0.0,
    layer_l2_reg: float=0.0
    ):
    """build a pistachio model using functional api"""
    def _get_feature_normalizers():
        """initialise and adapt the feature normalisers"""
        print(f'preprocessing - initialising normalisers')
        normalizers = {}
        for feature in feature_columns:
            normaliser =  Normalization(axis=None, name=f'normalizer_{feature}')
            just_this_feature_ds = train_dataset.map(lambda x,y: x[feature])
            normaliser.adapt(just_this_feature_ds)
            normalizers[feature] = normaliser
        return normalizers
        
    def _build_model(normalizers: Dict):
        normalized_inputs = []
        raw_inputs = []
        for feature in feature_columns:
            feature_input = tf.keras.Input(shape=(1,), name=feature)
            raw_inputs.append(feature_input)
            normalized_input = normalizers[feature](feature_input)
            normalized_inputs.append(normalized_input)

        input_layer = tf.keras.layers.concatenate(normalized_inputs)

        # densely connected layers
        d1 = Dense(
            units,
            activation='relu',
            name='dense_1',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=layer_l1_reg, l2=layer_l2_reg))
        
        d2 = Dense(
            units,
            activation='relu',
            name='dense_2',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=layer_l1_reg, l2=layer_l2_reg))
        

        # output layer
        output_layer = Dense(1, activation='sigmoid', name='predicted_probability')
        # http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines

        # define graph
        x = d1(input_layer)
        x = d2(x)
        output_probability = output_layer(x)
        
        model = tf.keras.Model(raw_inputs, output_probability)
        return model
    normalizers = _get_feature_normalizers()
    model = _build_model(normalizers)
    return model
#######################################################


def get_pistachio_fc_model(
    feature_columns: List[str],
    feature_crosses: list[tuple[tr,str]],
    train_dataset: tf.data.Dataset,
    n_layers: int=2,
    units: int=10,
    layer_l1_reg: float=0.0,
    layer_l2_reg: float=0.0
    ):
    """
    build a pistachio model using functional api
    includes some crossed features
    features_crosses is a list of tuples of feature names. feature pairs in this list will be multiplied at input

    """
    def _get_feature_normalizers():
        """initialise and adapt the feature normalisers"""
        print(f'preprocessing - initialising normalisers')
        normalizers = {}
        for feature in feature_columns:
            normaliser =  Normalization(axis=None, name=f'normalizer_{feature}')
            just_this_feature_ds = train_dataset.map(lambda x,y: x[feature])
            normaliser.adapt(just_this_feature_ds)
            normalizers[feature] = normaliser
        return normalizers
        
    def _build_model(normalizers: Dict):
        normalized_inputs = []
        raw_inputs = []
        feature_indices = {}
        for ii, feature in enumerate(feature_columns):
            feature_indices[feature] = ii
        
            feature_input = tf.keras.Input(shape=(1,), name=feature)
            raw_inputs.append(feature_input)
            normalized_input = normalizers[feature](feature_input)
            normalized_inputs.append(normalized_input)

        # crossed features
        crossed_features = []
        for k in feature_crosses:
            crossed_features.append(
                tf.math.multiply(
                    normalized_inputs[feature_indices[k[0]]],
                    normalized_inputs[feature_indices[k[1]]],
                    name=f'cross_normed_{k[0]}_{k[1]}'
                )
            )

        input_layer = tf.keras.layers.concatenate(normalized_inputs + crossed_features)
        # alias this here for convenience
        x = input_layer

        # densely connected layers - number specified by n_layers
        for dd in n_layers:
            dense_layer = Dense(
                units,
                activation='relu',
                name=f'dense_{dd}',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=layer_l1_reg, l2=layer_l2_reg))
            x = dense_layer(x)

        # output layer
        output_layer = Dense(1, activation='sigmoid', name='predicted_probability')
        # http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines

        output_probability = output_layer(x)
        
        model = tf.keras.Model(raw_inputs, output_probability)
        return model
    normalizers = _get_feature_normalizers()
    model = _build_model(normalizers)
    return model
#######################################################

