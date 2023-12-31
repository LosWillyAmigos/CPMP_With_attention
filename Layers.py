from keras.layers import Layer
from keras.layers import Concatenate, Flatten, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Input
import tensorflow as tf

class Model_CPMP(Layer):
    def __init__(self, num_layer_attention_add: int = 1,
                 heads: int = 5, S: int = 5, H: int = 5,
                 ) -> None:
        super(Model_CPMP, self).__init__()

        self.__num_layer_attention_add = num_layer_attention_add
        self.__flatten = Flatten()
        self.__dropout = Dropout(0.5)
        self.__dense_1 = Dense(H * 6, activation= 'sigmoid')
        self.__dense_5 = Dense(H * 9, activation= 'sigmoid')
        self.__dense_6 = Dense(H * 9, activation= 'sigmoid')
        self.__dense_2 = Dense(S, activation= 'sigmoid')
        self.__dense_3 = Dense(H + 1, activation= 'sigmoid')
        self.__dense_4 = Dense(H + 1)
        self.__multihead_atention = MultiHeadAttention(num_heads= heads, key_dim= H + 1)
        self.__normalization_layer = LayerNormalization(epsilon= 1e-6)
        self.__add = Add()

    @tf.autograph.experimental.do_not_convert
    def call(self, input: tf.TensorArray, training: bool = None) -> None:
        reshape = self.__multihead_atention(input, input)
        add = self.__add([input, reshape])
        normalization = self.__normalization_layer(add)
        dense_3 = self.__dense_3(normalization)
        dense_4 = self.__dense_4(dense_3)

        """for _ in range(self.__num_layer_attention_add):
            reshape = self.__multihead_atention(dense_4, input)
            add = self.__add([input, reshape])
            normalization = self.__normalization_layer(add)
            dense_3 = self.__dense_3(normalization)
            dense_4 = self.__dense_4(dense_3)"""

        flatten = self.__flatten(dense_4)
        dense_1 = self.__dense_1(flatten)
        dropout_1 = self.__dropout(dense_1)
        dense_5 = self.__dense_5(dropout_1, training= training)
        dense_6 = self.__dense_6(dense_5)
        dense_2 = self.__dense_2(dense_6)

        return dense_2

class ConcatenationLayer(Layer):
    def __init__(self, **kwargs) -> None:
        super(ConcatenationLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.TensorArray) -> None:
        labels = tf.ones(tf.shape(inputs)[1])
        labels = tf.expand_dims(labels, axis= 0)
        labels = tf.repeat(labels, repeats= tf.shape(inputs)[0], axis= 0)

        # Crear una matriz identidad de la misma forma que los arreglos
        matriz_identidad = tf.eye(tf.shape(labels)[-1], dtype=tf.float32)

        # Multiplicar cada arreglo por la matriz identidad para obtener la matriz diagonal
        matrices_diagonales = labels[:, :, tf.newaxis] * matriz_identidad

        test = tf.expand_dims(matrices_diagonales, axis= -1)

        matrices_copiadas = tf.expand_dims(inputs, axis= 1)
        matrices_copiadas = tf.repeat(matrices_copiadas, repeats= labels.shape[1], axis= 1)

        results = Concatenate(axis= 3)([matrices_copiadas, test])

        return results

    
class LayerExpandOutput(Layer):
    def __init__(self, **kwargs) -> None:
        super(LayerExpandOutput, self).__init__(**kwargs)

    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats=dim, axis=1)

        return expanded

class Reduction(Layer):
    def __init__(self) -> None:
        super(Reduction, self).__init__(trainable=False)

    def call(self, arr: tf.TensorArray, S) -> tf.TensorArray:
        aux = [True for n in range(S * S)]
        k = 0

        for i in range(S):
            for j in range(S):
                if i == j:
                    aux[k] = False
                k += 1

        mask = tf.constant(aux)
        output = tf.boolean_mask(arr, mask, axis= 1)
        output = tf.reshape(output, shape= (tf.shape(arr)[0], S * (S - 1)))

        return output

class OutputMultiplication(Layer):
    def __init__(self) -> None:
        super(OutputMultiplication,self).__init__(trainable=False)

    def call(self, arr1: tf.TensorArray, arr2: tf.TensorArray) -> tf.TensorArray:
        return arr1 * arr2
    

    
