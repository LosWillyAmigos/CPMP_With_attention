import keras
from keras.layers import Input, Flatten, Dense, Dropout, TimeDistributed
from keras.layers import TimeDistributed
from keras.models import Model
from CPMP_MODEL.Layers import OutputMultiplication, Model_CPMP, Reduction, DenseLayer
from CPMP_MODEL.Layers import ConcatenationLayer, LayerExpandOutput, FeedForward, Stack_Attention
import tensorflow as tf

def create_model(heads: int = 5, H: int = 5,
                optimizer: str | None = 'Adam', epsilon=1e-6
                ) -> Model:
    input_layer = Input(shape=(None,H+1))
    mask = DenseLayer(dim=H+1)(input_layer)
    layer_attention_so = Model_CPMP(H=H,heads=heads,activation='sigmoid',epsilon=epsilon)(mask)
    expand = LayerExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(mask)
    layer_attention_sd = Model_CPMP(H=H+1,heads=heads,activation='sigmoid')
    distributed = TimeDistributed(layer_attention_sd)(concatenation)
    unificate = Flatten()(distributed)
    m = OutputMultiplication()(unificate,expand)
    red = Reduction()(m)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics= ['mae', 'mse'])

    return model
    
def load_model(name: str) -> Model:
        c_o={'Model_CPMP': Model_CPMP, 
             'OutputMultiplication': OutputMultiplication,
             'LayerExpandOutput': LayerExpandOutput,
             'ConcatenationLayer': ConcatenationLayer,
             'Reduction': Reduction,
             'FeedForward' : FeedForward,
             'Stack_Attention' : Stack_Attention,
             'DenseLayer' : DenseLayer}
        model = tf.keras.models.load_model(name,custom_objects=c_o)
        
        return model

def save_model(name: str, model: Model | None) -> bool:
        if model is None:
            print('Model have not been initialized.')
            return False

        model.save(name)
        return True