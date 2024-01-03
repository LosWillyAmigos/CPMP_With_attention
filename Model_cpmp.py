import keras
from keras.layers import Input, Flatten, Dense, Dropout, TimeDistributed
from keras.layers import TimeDistributed, Concatenate
from keras.models import Model
from Layers import ConcatenationLayer, LayerExpandOutput, OutputMultiplication
from Layers import Stack_Attention, Model_CPMP, Reduction, UnificationLayer, N_Dense_Layer
import tensorflow as tf

def create_model(heads: int = 5, H: int = 5, S=5,
                optimizer: str | None = 'Adam',
                ) -> Model:
    input_layer = Input(shape=(S,H+1))
    layer_attention_so = Model_CPMP(heads=heads,H=H, S=S)(input_layer)
    expand = LayerExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(input_layer)
    distributed = TimeDistributed(Model_CPMP(heads=heads,H=H+1, S=S))(concatenation)
    unificate = Flatten()(distributed)
    m = OutputMultiplication()(unificate,expand)
    red = Reduction()(m,S=S)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics= ['mae', 'mse'])

    return model
    
def load_model(name: str) -> Model:
        c_o={'Model_CPMP': Model_CPMP, 
                        'OutputMultiplication': OutputMultiplication,
                        'LayerExpandOutput': LayerExpandOutput,
                        'ConcatenationLayer': ConcatenationLayer,
                        'Reduction': Reduction,
                        'UnificationLayer': UnificationLayer,
                        'N_Dense_Layer' : N_Dense_Layer
                        }
        model = tf.keras.models.load_model(name,custom_objects=c_o)
        
        return model

def save_model(name: str, model: Model) -> bool:
        if model is None:
            print('Model have not been initialized.')
            return False

        model.save(name)
        return True

def create_model_cpmp(heads: int = 5, H: int = 5, S=5,
                      optimizer: str | None = 'Adam') -> Model:
      
    input_ = Input(shape=(S, H+1))

    att_0 = Stack_Attention(heads=5, dim=H+1, act='sigmoid')(input_, input_)
    att_1 = Stack_Attention(heads=5, dim=H+1, act='sigmoid')(input_, att_0)
    att_2 = Stack_Attention(heads=5, dim=H+1, act='sigmoid')(input_, att_1)

    flat = Flatten()(att_2)
    h_ = Dense(32, activation='sigmoid')(flat)
    o_ = Dense(S, activation='sigmoid')(h_)

    model = Model(inputs=input_, outputs=o_)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae', 'mse'])

    return model
