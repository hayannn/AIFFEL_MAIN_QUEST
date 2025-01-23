import numpy as np

import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dropout, Dense, Flatten, Input, BatchNormalization
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from collections import defaultdict

class FeaturesEmbedding(Layer):  
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.embedding = tf.keras.layers.Embedding(input_dim=self.total_dim, output_dim=self.embed_dim)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = x + tf.constant(self.offsets)
        return self.embedding(x)

class MultiLayerPerceptron(Layer):  
    def __init__(self, input_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        hidden_units = [input_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]
        self.linears = [Dense(units, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for units in hidden_units[1:]]
        self.activation = tf.keras.layers.Activation(activation)
        if self.use_bn:
            self.bn = [BatchNormalization() for _ in hidden_units[1:]]
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x

class MultiHeadSelfAttention(Layer):  
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))

        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))  
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num), axis=-1)
        result = tf.squeeze(result, axis=0) 

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res, 'seed': self.seed}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        base_config.update(config)
        return base_config

class AutoIntMLP(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, dnn_hidden_units=(32, 32), **kwargs):
        super(AutoIntMLP, self).__init__(**kwargs)
        self.embedding_layer = FeaturesEmbedding(field_dims, embedding_size)
        self.att_layers = [MultiHeadSelfAttention(embedding_size, att_head_num) for _ in range(att_layer_num)]
        self.dnn = MultiLayerPerceptron(embedding_size * len(field_dims), dnn_hidden_units)
        self.final_dense = Dense(1, activation=None)
    
    def call(self, inputs):
        embed_x = self.embedding_layer(inputs)
        att_input = embed_x
        for layer in self.att_layers:
            att_input = layer(att_input)
        att_output = Flatten()(att_input)

        dnn_input = Flatten()(embed_x)
        dnn_output = self.dnn(dnn_input)

        combined_output = tf.add(att_output, dnn_output)

        y_pred = self.final_dense(combined_output)
        y_pred = tf.sigmoid(y_pred)
        return y_pred

def predict_model(model, pred_df):
    batch_size = 2048
    top = 10
    user_pred_info = []
    total_rows = len(pred_df)
    for i in range(0, total_rows, batch_size):
        features = pred_df.iloc[i:i + batch_size, :].values
        y_pred = model.predict(features, verbose=False)
        for feature, p in zip(features, y_pred):
            u_i = feature[:2]
            user_pred_info.append((int(u_i[1]), float(p)))

    return sorted(user_pred_info, key=lambda s: s[1], reverse=True)[:top]

def test_model(model, test_df):
    batch_size = 2048
    top = 10
    user_pred_info = defaultdict(list)
    total_rows = len(test_df)

    for i in range(0, total_rows, batch_size):
        features = test_df.iloc[i:i + batch_size, :-1].values
        y_pred = model.predict(features, verbose=False)

        y_pred = y_pred[:, 0]  # Flatten y_pred to be 1D if it's 2D

        for feature, p in zip(features, y_pred):
            u_i = feature[:2]
            user_pred_info[int(u_i[0])].append((int(u_i[1]), float(p)))

    sorted_user_pred_info = {k: sorted(v, key=lambda s: s[1], reverse=True) for k, v in user_pred_info.items()}

    top_recommendations = {}
    for user_id, pred_info in sorted_user_pred_info.items():
        top_recommendations[user_id] = pred_info[:top]

    return top_recommendations
