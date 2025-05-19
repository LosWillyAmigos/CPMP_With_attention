from cpmp_ml.generators import generate_data_v3
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.validations import validate_model
from keras.api.models import Model
import numpy as np
import json
import os
from datetime import datetime

def reinforcement_training(model: Model, 
                          S: int, 
                          H: int, 
                          N: int, 
                          validate_optimizer: OptimizerStrategy, 
                          adapter: DataAdapter,
                          sample_size: int = 50000, 
                          iter: int = 5, 
                          max_steps: int = 30, 
                          epochs: int = 5, 
                          batch_size: int = 20, 
                          verbose: bool = True, 
                          perms_by_layout: int = 1,
                          save_history: bool = True,
                          history_file: str = None) -> None:
    """
    Realiza entrenamiento reforzado de un modelo de CPMP.
    
    Args:
        model: Modelo a entrenar
        S: Número de stacks
        H: Altura de los stacks
        N: Número máximo de prioridad
        validate_optimizer: Estrategia de optimización para validación
        adapter: Adaptador de datos
        sample_size: Tamaño de la muestra de datos a generar
        iter: Número de iteraciones de entrenamiento
        max_steps: Número máximo de pasos en la generación de datos
        epochs: Número de épocas de entrenamiento por iteración
        batch_size: Tamaño del lote para la generación de datos
        verbose: Si es True, muestra información durante el entrenamiento
        perms_by_layout: Número de permutaciones por layout
        save_history: Si es True, guarda el historial de entrenamiento
        history_file: Nombre del archivo para guardar el historial. Si es None, 
                      se genera un nombre basado en la fecha y configuración
    """
    optimizer = GreedyModel(model, adapter)
    
    # Crear historial de entrenamiento
    training_history = {
        "config": {
            "S": S,
            "H": H,
            "N": N,
            "sample_size": sample_size,
            "max_steps": max_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "perms_by_layout": perms_by_layout,
            "iterations": iter
        },
        "iterations": []
    }
    
    # Generar nombre de archivo para el historial si no se proporciona
    if save_history and history_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = f"training_history_S{S}H{H}N{N}_{timestamp}.json"
        # Crear directorio para el historial si no existe
        if not os.path.exists("./training_history"):
            os.makedirs("./training_history")
        history_file = f"./training_history/{history_file}"

    for i in range(iter):
        if verbose: 
            print(f"Paso {i + 1}")

        # Generar datos para entrenamiento
        data, labels = generate_data_v3(optimizer, adapter, S, H, N, sample_size, batch_size, 
                                       perms_by_layout=perms_by_layout, max_steps=max_steps)

        data = np.stack(data)
        labels = np.stack(labels)
        
        # Información sobre los datos generados
        data_info = {
            "shape_data": data.shape,
            "shape_labels": labels.shape,
            "data_sample": data[0].tolist() if len(data) > 0 else [],
            "labels_sample": labels[0].tolist() if len(labels) > 0 else []
        }

        # Entrenar el modelo
        history = model.fit(data, labels, epochs=epochs, verbose=verbose)
        
        # Validar el modelo
        results_model, results_greedy = validate_model(model, validate_optimizer, adapter, S, H, N, 1000, max_steps=max_steps)
        
        # Guardar resultados de esta iteración
        iteration_results = {
            "iteration": i + 1,
            "history": {k: [float(val) for val in v] for k, v in history.history.items()},
            "validation": {
                "model_success_rate": float(results_model),
                "greedy_success_rate": float(results_greedy)
            },
            "data_info": data_info
        }
        
        # Añadir al historial
        training_history["iterations"].append(iteration_results)
        
        # Guardar historial en cada iteración si se solicita
        if save_history:
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            if verbose:
                print(f"Historial guardado en: {history_file}")

        # Liberar memoria
        del data, labels

        if verbose: 
            print('')
        
        # Si el modelo alcanza buen rendimiento, terminar el entrenamiento
        if results_model > 96.0: 
            break
    
    # Guardar el historial final si no se ha hecho en cada iteración
    if save_history and not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        if verbose:
            print(f"Historial guardado en: {history_file}")
    
    return training_history

if __name__ == "__main__":
    from attentional_cpmp.model import create_model
    from cpmp_ml.optimizer import GreedyV2
    from cpmp_ml.utils.adapters import AttentionModel
    
    # Crear un nuevo modelo
    # Parámetros para un modelo de 7x7
    H = 7
    S = 10
    N = 50
    
    # Crear el modelo con configuración adecuada
    model = create_model(
        H=H,                          # Altura de los stacks
        key_dim=8,                    # Dimensión de las claves
        num_heads=5,                  # Número de cabezas de atención
        num_stacks=7,                 # Número de capas de atención
        list_neurons_hide=[32, 24, 16], # Capas ocultas
        list_neurons_feed=[32, 24, 16]  # Capas de alimentación
    )
    
    # Realizar entrenamiento reforzado
    reinforcement_training(
        model=model,
        S=S,                          # Número de stacks
        H=H,                          # Altura de los stacks  
        N=N,                          # Máxima prioridad
        validate_optimizer=GreedyV2(),
        adapter=AttentionModel(),
        iter=3,                       # Número de iteraciones
        max_steps=100,                # Pasos máximos
        epochs=10,                    # Épocas por iteración
        batch_size=32,                # Tamaño del lote
        save_history=True             # Guardar historial
    ) 