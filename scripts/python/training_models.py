import argparse
import json
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from attentional_cpmp.model import create_model
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.utils import get_data
from keras.api.callbacks import EarlyStopping
from typing import Any

def entrenar_y_guardar_modelos(models_config: list, 
                               H: int,
                               X: np.ndarray, 
                               Y: np.ndarray, 
                               epochs: int = 10,
                               dropout: float = 0.2,
                               rate: float = 0.0,
                               batch_size: int = 32,
                               validation_split: float = 0.2,
                               optimizer: Any | None = 'Adam',
                               loss: str = 'binary_crossentropy',
                               metrics: list[Any] = ['mae', 'mse', 'accuracy'],
                               monitor: str = 'val_loss',
                               patience: int = 3,
                               verbose: int = 1,
                               restore_best_weights: bool = True,
                               path_save_models: str = 'models/',
                               guardar_historial: bool = False,
                               nombre_archivo_excel: str = 'historial_entrenamiento.xlsx'):
    
    resumen_historial = []  # Aquí se guardan los datos de cada modelo
    
    for idx, config in enumerate(models_config):
        print(f"Entrenando modelo {idx + 1}...")
        modelo = create_model(H=H, 
                              rate=rate,
                              dropout=dropout,
                              optimizer=optimizer,
                              loss=loss,
                              metrics=metrics,
                              **config)

        early_stop = EarlyStopping(monitor=monitor,
                                   patience=patience,
                                   verbose=verbose,
                                   restore_best_weights=restore_best_weights)

        history = modelo.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        nombre_archivo = f"modelo_{idx + 1}"
        modelo.save(f"{path_save_models}{nombre_archivo}.h5")
        modelo.save(f"{path_save_models}{nombre_archivo}.keras")
        print(f"Modelo guardado como {nombre_archivo}.h5\n")

        if guardar_historial:

            val_monitor_history = history.history[monitor]

            best_epoch = val_monitor_history.index(min(val_monitor_history))
            
            # Últimos valores de cada métrica
            fila = {
                'modelo': nombre_archivo,
                'epocas_entrenadas': best_epoch + 1
            }
            for clave, valores in history.history.items():
                fila[f'Mejor {clave}'] = valores[best_epoch]
            
            resumen_historial.append(fila)

    # Guardar Excel con historial completo
    if guardar_historial and resumen_historial:
        df_resumen = pd.DataFrame(resumen_historial)
        ruta_excel = os.path.join(path_save_models, nombre_archivo_excel)
        df_resumen.to_excel(ruta_excel, index=False)
        print(f"\n✅ Historial de entrenamiento guardado en {ruta_excel}")


# ==== Argumentos desde la línea de comandos ====
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos desde configuración JSON")
    parser.add_argument('--model_config', required=True, help="Ruta al archivo JSON con configuración de modelos")
    parser.add_argument('--data_dimension', required=True, help="Dimensión de los datos a recuperar")
    parser.add_argument('--H', type=int, required=True, help="Dimensión de los datos a recuperar")
    parser.add_argument('--data', required=True, help="Ruta al archivo JSON con los datos")
    parser.add_argument('--train_config', required=True, help="Ruta al archivo JSON con configuración de entrenamiento")
    parser.add_argument('--path_save_models', type=str, required=True, help="Ruta donde se guardarán los modelos")
    parser.add_argument('--guardar_historial', type=bool, help="Guardar historial de entrenamiento en archivo Excel")
    parser.add_argument('--nombre_archivo_excel', type=str, default='historial_entrenamiento.xlsx', help="Nombre base del archivo Excel de historial")
    parser.add_argument('--max_data', type=int, default=0, help="Dantidad de datos de entrenamiento")
    
    
    args = parser.parse_args()

    data_Sx7 = load_data_from_json(args.data)
    x, y = get_data(data_Sx7, args.data_dimension)
    if args.max_data > 0:
        x, y = x[:args.max_data], y[:args.max_data]

    with open(args.model_config, 'r') as f:
        models_config = json.load(f)

    with open(args.train_config, 'r') as f:
        config_training = json.load(f)

    if not os.path.exists(args.path_save_models):
        os.makedirs(args.path_save_models)

    entrenar_y_guardar_modelos(models_config=models_config, 
                               H=args.H,
                               X=x, 
                               Y=y,
                               path_save_models=args.path_save_models,
                               guardar_historial=args.guardar_historial,
                               nombre_archivo_excel=args.nombre_archivo_excel,
                               **config_training)

if __name__ == '__main__':
    main()
