from keras.layers import Layer
from keras.layers import Concatenate, Flatten, Dense, Dropout, Multiply
from keras.layers import MultiHeadAttention, LayerNormalization, Add, TimeDistributed
import tensorflow as tf
from math import sqrt

class CPMP_Masking(Layer):
    def __init__(self, H: int) -> None:
        super(CPMP_Masking, self).__init__()

        self.__dense_1__ = Dense(H, activation= 'sigmoid')
        self.__dense_2__ = Dense(H * 3, activation= 'sigmoid')
        self.__dense_3__ = Dense(H * 2, activation= 'sigmoid')
        self.__dense_4__ = Dense(H, activation= 'sigmoid')
    
    def call(self, arr: tf.TensorArray) -> tf.TensorArray:
        dense_1 = self.__dense_1__(arr)
        dense_2 = self.__dense_2__(dense_1)
        dense_3 = self.__dense_3__(dense_2)
        dense_4 = self.__dense_4__(dense_3)

        return dense_4

class Reduction(Layer):
    def __init__(self) -> None:
        super(Reduction, self).__init__(trainable=False)

    def call(self, arr: tf.Tensor) -> tf.Tensor:
        S = tf.shape(arr)[1]
        S = tf.cast(tf.round(tf.sqrt(tf.cast(S, dtype=tf.float32))), dtype=tf.int32)

        aux = tf.math.logical_not(tf.eye(S, dtype=tf.bool))
        mask = tf.reshape(aux, [-1])
        
        output = tf.boolean_mask(arr, mask, axis=1)

        return output

class ConcatenationLayer(Layer):
    def __init__(self, **kwargs) -> None:
        super(ConcatenationLayer, self).__init__(**kwargs)
    def call(self, inputs: tf.TensorArray) -> None:
        labels = tf.ones(tf.shape(inputs)[1])
        labels = tf.expand_dims(labels, axis= 0)
        labels = tf.repeat(labels, repeats= tf.shape(inputs)[0], axis= 0)

        matriz_identidad = tf.eye(tf.shape(labels)[-1], dtype=tf.float32)
        matrices_diagonales = labels[:, :, tf.newaxis] * matriz_identidad

        test = tf.expand_dims(matrices_diagonales, axis= -1)
        
        matrices_copiadas = tf.expand_dims(inputs, axis= 1)
        matrices_copiadas = tf.repeat(matrices_copiadas, repeats= tf.shape(labels)[1], axis= 1)

        results = Concatenate(axis= 3)([matrices_copiadas, test])

        return results

class StackWiseProcessing(Layer):
    def __init__(self, units: int, activation: str | None) -> None:
        super(StackWiseProcessing, self).__init__()

        self.__dense_1__ = Dense(units * 5, activation= activation)
        self.__dense_2__ = Dense(units * 4, activation= activation)
        self.__dense_3__ = Dense(units * 4, activation= activation)
        self.__dense_4__ = Dense(units * 3, activation= activation)
        self.__dense_5__ = Dense(1, activation= activation)
        self.__dropout_1__ = Dropout(0.1)
        self.__dropout_2__ = Dropout(0.2)
        self.__dropout_3__ = Dropout(0.2)

    def call(self, arr: tf.TensorArray) -> tf.TensorArray:
        dense_1 = self.__dense_1__(arr)
        dense_2 = self.__dense_2__(dense_1)
        dropout_1 = self.__dropout_1__(dense_2)
        dense_3 = self.__dense_3__(dropout_1)
        dropout_2 = self.__dropout_2__(dense_3)
        dense_4 = self.__dense_4__(dropout_2)
        dropout_3 = self.__dropout_3__(dense_4)
        dense_5 = self.__dense_5__(dropout_3)

        return dense_5


class LayerExpandOutput(Layer):
    def __init__(self, **kwargs) -> None:
        super(LayerExpandOutput, self).__init__(**kwargs)

    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats=dim, axis=1)

        return expanded

class FeedForward(Layer):
    def __init__(self, units: int, activation: str | None):
        super(FeedForward, self).__init__()

        self.__dense_1__ = Dense(units * 4, activation= activation)
        self.__dense_2__ = Dense(units * 6, activation= activation)
        self.__dense_3__ = Dense(units * 5, activation= activation)
        self.__dense_4__ = Dense(units, activation= activation)
        self.__dropout_1__ = Dropout(0.2)
        self.__dropout_2__ = Dropout(0.2)

    def call(self, arr: tf.Tensor) -> tf.Tensor:
        dense_1 = self.__dense_1__(arr)
        dropout_1 = self.__dropout_1__(dense_1)

        dense_2 = self.__dense_2__(dropout_1)
        dropout_2 = self.__dropout_2__(dense_2)
        
        dense_3 = self.__dense_3__(dropout_2)
        dense_4 = self.__dense_4__(dense_3)

        return dense_4

class Transformer(Layer):
    def __init__(self, num_heads: int, key_dim: int) -> None:
        super(Transformer, self).__init__()

        self.__multihead__ = MultiHeadAttention(num_heads= num_heads, key_dim= key_dim)
        self.__add_layer__ = Add()
        self.__norm__ = LayerNormalization()
        self.__feed__ = FeedForward(units= key_dim, activation= 'sigmoid')

    def call(self, query: tf.TensorArray, key: tf.TensorArray, value: tf.TensorArray | None) -> tf.Tensor:
        multihead = self.__multihead__(query, key, value)
        add_1 = self.__add_layer__([query, multihead])
        norm_1 = self.__norm__(add_1)
        feed = self.__feed__(norm_1)
        add_2 = self.__add_layer__([norm_1, feed])
        norm_2 = self.__norm__(add_2)

        return norm_2

    
class Model_CPMP(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, key_dim: int, activation: str | None) -> None:
        super(Model_CPMP, self).__init__()

        self.__transformer_1__ = Transformer(num_heads= num_heads, key_dim= key_dim)
        self.__transformer_2__ = Transformer(num_heads= num_heads, key_dim= key_dim)
        self.__masking__ = CPMP_Masking(H= key_dim)
        self.__multiply__ = Multiply()
        self.__stackwise__ = StackWiseProcessing(units= key_dim, activation= activation)
        self.__flatten__ = Flatten()

    def call(self, state: tf.TensorArray) -> tf.TensorArray:
        masking = self.__masking__(state)
        multiply = self.__multiply__([state, masking])

        transformer_1 = self.__transformer_1__(state, state, state)
        transformer_2 = self.__transformer_2__(multiply, transformer_1, transformer_1)

        stackwise = self.__stackwise__(transformer_2)
        flatten = self.__flatten__(stackwise)

        return flatten