from keras.layers import Layer, TimeDistributed, Dense, Dropout, Concatenate, Flatten
from keras.layers import Softmax, MultiHeadAttention, LayerNormalization, Add, Softmax
import tensorflow as tf
from keras import backend as K


class FeedForward(Layer):
    """
    FeedForward Neural Network Layer

    This class implements a simple feedforward neural network layer using TensorFlow's Keras API. The network consists of multiple dense layers with dropout regularization.

    Attributes:
        dim (int): The input dimension of the network. Default value is 5 for five column models.
        activation (str): The activation function applied to each dense layer. Default is 'sigmoid'.
        dim_output (int): The output dimension of the network. Default is 5.

    Methods:
        __init__(self, dim=5, activation='sigmoid', dim_output=5)
            Initializes the FeedForward layer with specified input dimension, activation function, and output dimension.

        call(self, inputs)
            Defines the forward pass of the network.

    Usage:
        # Create a FeedForward layer
        feedforward_layer = FeedForward(dim=10, activation='relu', dim_output=3)

        # Perform a forward pass
        output = feedforward_layer(inputs)
    """
    def __init__(self, dim_input: int, activation: str = 'sigmoid', dim_output: int = 1) -> None:
        # Verificar si los parámetros requeridos tienen valores
        if dim_input is None:
            raise ValueError("dim_input has no value.")
        super(FeedForward,self).__init__()
        self.d1 = Dense(dim_input, activation=activation)
        self.d2 = Dense(dim_input * 4, activation=activation)
        self.dp1 = Dropout(0.5)
        self.d3 = Dense(dim_input * 3, activation=activation)
        self.dp2 = Dropout(0.5)
        self.d4 = Dense(dim_input * 2, activation=activation)
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
    
class DenseLayer(Layer):
    """
    DenseLayer

    This class implements a custom layer using TensorFlow's Keras API. It consists of two dense layers with the same activation function.

    Attributes:
        dim (int): The output dimension of the dense layers.
        activation (str): Activation function applied to the dense layers. Default is 'sigmoid'.

    Methods:
        __init__(self, dim: int, activation: str = 'sigmoid')
            Initializes the DenseLayer with specified output dimension and activation function.

        call(self, inputs)
            Defines the forward pass of the DenseLayer.

    Usage:
        # Create a DenseLayer
        dense_layer = DenseLayer(dim=64, activation='relu')

        # Perform a forward pass
        output = dense_layer(inputs)
    """
    def __init__(self, dim: int, activation: str = 'sigmoid') -> None:
        super(DenseLayer,self).__init__()
        self.time = TimeDistributed(Dense(dim, activation))

    def call(self, inputs: tf.TensorArray):
        o1 = self.time(inputs)
        return o1
    

class Stack_Attention(Layer):
    """
    Stack Attention Layer

    This class implements a stack attention mechanism using TensorFlow's Keras API. It combines multi-head attention with layer normalization and dense layers.

    Attributes:
        heads (int): The number of attention heads.
        dim (int): The key dimension for multi-head attention.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.
        act (str): Activation function applied to the dense layers. Default is 'sigmoid'.

    Methods:
        __init__(self, heads: int, dim: int, epsilon=1e-6, act='sigmoid')
            Initializes the Stack Attention layer with specified parameters.

        call(self, inputs_o, inputs_att, training=True)
            Defines the forward pass of the stack attention layer.

    Usage:
        # Create a Stack Attention layer
        stack_attention_layer = Stack_Attention(heads=8, dim=64, epsilon=1e-6, act='relu')

        # Perform a forward pass
        output = stack_attention_layer(inputs_o, inputs_att, training=True)
    """
    def __init__(self, heads: int, dim: int,epsilon=1e-6, act = 'sigmoid') -> None:
        if heads is None or dim is None: 
            raise ValueError("heads or dim has no value.")
        super(Stack_Attention,self).__init__()
        self.multihead = MultiHeadAttention(num_heads=heads,key_dim=dim)
        self.layer_n = LayerNormalization(epsilon=epsilon)
        self.feed_1 = DenseLayer(dim, activation=act)
        self.feed_0 = DenseLayer(dim, activation='linear')
        self.add = Add()
    
    def call(self, inputs_o: tf.TensorArray, inputs_att: tf.TensorArray, training=True):
        att = self.multihead(inputs_att,inputs_att, training=training)
        ad = self.add([inputs_o,att])
        norm = self.layer_n(ad)
        output = self.feed_0(norm)
        output_ = self.feed_1(output)
        return output_
    
class Model_CPMP(Layer):
    """
    Model_CPMP Layer

    This class implements a custom neural network model named Model_CPMP using TensorFlow's Keras API. It combines stack attention and feedforward layers.

    Attributes:
        heads (int): The number of attention heads.
        H (int): The dimension parameter used in stack attention and feedforward layers.
        activation (str): Activation function applied to the inner layers. Default is 'sigmoid'.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.

    Methods:
        __init__(self, heads: int, H: int, activation='sigmoid', epsilon=1e-6)
            Initializes the Model_CPMP layer with specified parameters.

        call(self, input_0, training=True)
            Defines the forward pass of the Model_CPMP layer.

    Usage:
        # Create a Model_CPMP layer
        model_cpmp_layer = Model_CPMP(heads=8, H=64, activation='relu', epsilon=1e-6)

        # Perform a forward pass
        output = model_cpmp_layer(input_0, training=True)
    """
    def __init__(self, heads: int, H: int,
                activation:str = 'sigmoid', epsilon=1e-6) -> None:
        super(Model_CPMP, self).__init__()
        if heads is None or H is None:
            raise ValueError("heads or H has no value.")
        self.att_0 = Stack_Attention(heads=heads, dim=H+1, act=activation, epsilon=epsilon)
        self.att_1 = Stack_Attention(heads=heads, dim=H+1, act=activation,epsilon=epsilon)
        self.feed_0 = TimeDistributed(FeedForward(dim_input=H+1,activation='sigmoid',dim_output=1))

    @tf.autograph.experimental.do_not_convert
    def call(self, input_0: tf.TensorArray, training=True) -> None:
        at1 = self.att_0(input_0,input_0,training)
        at2 = self.att_1(at1,at1,training)

        dn0 = self.feed_0(at2)
        return dn0
 
class LayerExpandOutput(Layer):
    """
    LayerExpandOutput

    This class implements a custom layer using TensorFlow's Keras API. It expands the output by repeating each element along the second axis.

    Attributes:
        None

    Methods:
        __init__(self, **kwargs)
            Initializes the LayerExpandOutput layer.

        call(self, inputs)
            Defines the forward pass of the LayerExpandOutput layer.

    Usage:
        # Create a LayerExpandOutput layer
        expand_output_layer = LayerExpandOutput()

        # Perform a forward pass
        output = expand_output_layer(inputs)
    """

    def __init__(self, **kwargs) -> None:
        super(LayerExpandOutput, self).__init__(**kwargs)

    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats=dim, axis=1)
        output = tf.reshape(expanded,shape=(tf.shape(inputs)[0], tf.shape(expanded)[1]))

        return output

class ConcatenationLayer(Layer):
    """
    ConcatenationLayer

    This class implements a custom concatenation layer using TensorFlow's Keras API. It concatenates each input tensor with a corresponding diagonal matrix to assign a stack origin probability to each matrix.

    Attributes:
        None

    Methods:
        __init__(self, **kwargs)
            Initializes the ConcatenationLayer.

        call(self, inputs)
            Defines the forward pass of the ConcatenationLayer.

    Usage:
        # Create a ConcatenationLayer
        concatenation_layer = ConcatenationLayer()

        # Perform a forward pass
        output = concatenation_layer(inputs)
    """
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
    """
    OutputMultiplication Layer

    This class implements a custom layer using TensorFlow's Keras API. It performs element-wise multiplication between two input tensor arrays.

    Attributes:
        None

    Methods:
        __init__(self)
            Initializes the OutputMultiplication layer.

        call(self, arr1, arr2)
            Defines the forward pass of the OutputMultiplication layer.

    Usage:
        # Create an OutputMultiplication layer
        multiplication_layer = OutputMultiplication()

        # Perform a forward pass
        output = multiplication_layer(arr1, arr2)
    """
    def __init__(self) -> None:
        super(OutputMultiplication,self).__init__(trainable=False)

    def call(self, arr1: tf.TensorArray, arr2: tf.TensorArray) -> tf.TensorArray:
        return arr1 * arr2
    
class Reduction(Layer):
    """
    Reduction Layer

    This class implements a custom reduction layer using TensorFlow's Keras API. It reduces the input tensor by removing the diagonal elements.

    Attributes:
        None

    Methods:
        __init__(self)
            Initializes the Reduction layer.

        call(self, arr)
            Defines the forward pass of the Reduction layer.

    Usage:
        # Create a Reduction layer
        reduction_layer = Reduction()

        # Perform a forward pass
        output = reduction_layer(arr)
    """
    def __init__(self) -> None:
        super(Reduction, self).__init__(trainable=False)

    def call(self, arr: tf.Tensor) -> tf.Tensor:
        S = tf.shape(arr)[1]
        S = tf.cast(tf.round(tf.sqrt(tf.cast(S, dtype=tf.float32))), dtype=tf.int32)

        aux = tf.math.logical_not(tf.eye(S, dtype=tf.bool))
        mask = tf.reshape(aux, [-1])
        
        output = tf.boolean_mask(arr, mask, axis=1)

        return output
