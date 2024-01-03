from keras.layers import Layer, TimeDistributed, Dense, Dropout, Concatenate, Flatten
from keras.layers import Softmax, MultiHeadAttention, LayerNormalization, Add
import tensorflow as tf
from keras import backend as K

class N_Dense_Layer(Layer):
    def __init__(self, dim = 5, activation: str = 'sigmoid', dim_output=1) -> None:
        super(N_Dense_Layer,self).__init__()
        self.__d1 = Dense(dim, activation=activation)
        self.__d2 = Dense(dim * 2, activation=activation)
        self.__d3 = Dense(dim_output, activation=activation)

    def call(self, inputs: tf.TensorArray):
        o1 = self.__d1(inputs)
        o2 = self.__d2(o1)
        o3 = self.__d3(o2)

        return o3

class FeedForward(Layer):
    def __init__(self, dim: int = 5, activation: str = 'sigmoid', dim_output=5) -> None:
        super(FeedForward,self).__init__()
        self.n_dense = N_Dense_Layer(dim=dim, activation=activation, dim_output=dim_output)
        self.time = TimeDistributed(self.n_dense)

    def call(self, inputs: tf.TensorArray):
        result = self.time(inputs)
        return result
    
class Model_CPMP(Layer):
    def __init__(self, heads: int = 5, H: int = 5,
                activation:str = 'sigmoid', S:int = 5) -> None:
        super(Model_CPMP, self).__init__()

        self.att_0 = Stack_Attention(heads=5, dim=H+1, act=activation)
        self.att_1 = Stack_Attention(heads=5, dim=H+1, act=activation)
        self.flatten = Flatten()
        self.dense_0 = Dense(S*2,activation=activation)
        self.dense_1 = Dense(S,activation=activation)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_0: tf.TensorArray, training=True) -> None:
        at1 = self.att_0(input_0,input_0,training)
        at2 = self.att_1(input_0,at1,training)
        flt = self.flatten(at2)
        dn0 = self.dense_0(flt)
        dn1 = self.dense_1(dn0)
        return dn1
    
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

class UnificationLayer(Layer):
    def __init__(self,**kwargs):
        super(UnificationLayer,self).__init__(**kwargs)

    def call(self, inputs: tf.TensorArray):
        reshape = tf.reshape(inputs,shape=(tf.shape(inputs)[0],tf.shape(inputs)[1] * tf.shape(inputs)[2]))
        return reshape
    
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