from keras.layers import Layer, TimeDistributed, Dense, Dropout, Concatenate, Flatten
from keras.layers import Softmax, MultiHeadAttention, LayerNormalization, Add
import tensorflow as tf
from keras import backend as K

class N_Dense_Layer(Layer):
    def __init__(self, dim = 5, activation: str = 'sigmoid', dim_output=5) -> None:
        super(N_Dense_Layer,self).__init__()
        self.d1 = Dense(dim, activation=activation)
        self.d2 = Dense(dim * 4, activation=activation)
        self.dp1 = Dropout(0.5)
        self.d3 = Dense(dim * 6, activation=activation)
        self.dp2 = Dropout(0.5)
        self.d4 = Dense(dim * 4, activation=activation)
        self.d5 = Dense(dim_output, activation=activation)

    def call(self, inputs: tf.TensorArray):
        o1 = self.d1(inputs)
        o2 = self.d2(o1)
        d1 = self.dp1(o2)
        o3 = self.d3(d1)
        o4 = self.d4(o3)
        d2 = self.dp2(o4)
        o5 = self.d5(d2)

        return o5


class FeedForward(Layer):
    def __init__(self, dim: int = 5, activation: str = 'sigmoid', dim_output=5) -> None:
        super(FeedForward,self).__init__()
        self.n_dense = N_Dense_Layer(dim=dim, activation=activation, dim_output=dim_output)
        self.time = TimeDistributed(self.n_dense)

    def call(self, inputs: tf.TensorArray):
        result = self.time(inputs)
        return result
    
class Stack_Attention(Layer):
    def __init__(self, heads: int = 5, dim: int = 5,epsilon=1e-6, act = 'sigmoid') -> None:
        super(Stack_Attention,self).__init__()

        self.multihead = MultiHeadAttention(num_heads=heads,key_dim=dim)
        self.layer_n = LayerNormalization(epsilon=epsilon)
        self.feed_1 = Dense(dim, activation=act)
        self.feed_0 = Dense(dim, activation='linear')
        self.add = Add()
    
    def call(self, inputs_o: tf.TensorArray, inputs_att: tf.TensorArray, training=True):
        att = self.multihead(inputs_att,inputs_att, training=training)
        ad = self.add([inputs_o,att])
        norm = self.layer_n(ad)
        output = self.feed_0(norm)
        output_ = self.feed_1(output)
        return output_
    
class Model_CPMP(Layer):
    def __init__(self, heads: int = 5, H: int = 5,
                activation:str = 'sigmoid') -> None:
        super(Model_CPMP, self).__init__()

        self.att_0 = Stack_Attention(heads=5, dim=H+1, act=activation)
        self.att_1 = Stack_Attention(heads=5, dim=H+1, act=activation)
        self.feed_0 = FeedForward(dim=H+1,activation='sigmoid',dim_output=1)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_0: tf.TensorArray, training=True) -> None:
        at1 = self.att_0(input_0,input_0,training)
        at2 = self.att_1(at1,at1,training)

        dn0 = self.feed_0(at2)
        return dn0
    
class LayerExpandOutput(Layer):
    def __init__(self, **kwargs) -> None:
        super(LayerExpandOutput, self).__init__(**kwargs)

    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats=dim, axis=1)
        output = tf.reshape(expanded,shape=(tf.shape(inputs)[0], tf.shape(expanded)[1]))

        return output

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
        matrices_copiadas = tf.repeat(matrices_copiadas, repeats= tf.shape(labels)[1], axis= 1)

        results = Concatenate(axis= 3)([matrices_copiadas, test])

        return results

    
class OutputMultiplication(Layer):
    def __init__(self) -> None:
        super(OutputMultiplication,self).__init__(trainable=False)

    def call(self, arr1: tf.TensorArray, arr2: tf.TensorArray) -> tf.TensorArray:
        return arr1 * arr2
    
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

class UnificationLayer(Layer):
    def __init__(self,**kwargs):
        super(UnificationLayer,self).__init__(**kwargs)

    def call(self, inputs: tf.TensorArray):
        reshape = tf.reshape(inputs,shape=(tf.shape(inputs)[0],tf.shape(inputs)[1] * tf.shape(inputs)[2]))
        return reshape