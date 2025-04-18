from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import MedianPruner

from attentional_cpmp.utils.hyperparameter_search import load_json, insert_trials
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.utils import get_data
from attentional_cpmp.validations import PercentageSolved
from attentional_cpmp.model import create_model

from typing import Any
from keras.api.callbacks import EarlyStopping
from keras.api.backend import clear_session
from keras.api.optimizers import Adam
from optuna import Trial
from optuna.integration import TFKerasPruningCallback
from optuna.exceptions import TrialPruned

import argparse
import tensorflow as tf
import os
import random
import numpy as np


def objective(trial: Trial,
              max_num_stacks: int,
              max_num_heads: int,
              max_key_dim: int,
              max_value_dim: int,
              max_n_dropout_hide: int,
              max_n_dropout_feed: int,
              max_epsilon: int,
              max_num_neurons_layers_feed: int,
              max_num_neurons_layers_hide: int,
              max_units_neurons_feed: int,
              max_units_neurons_hide: int,
              step:int,
              H: int,
              optimizer: Any | None,
              loss: Any | None,
              metrics: list[Any] | None,
              monitor: str,
              patience: int,
              verbose: int,
              restore_best_weights: bool,
              X_train: Any | np.ndarray,
              Y_train: Any | np.ndarray,
              epochs: int,
              batch_size: int,
              validation_split: float) -> float:

        X_train_copy = np.copy(X_train)
        Y_train_copy = np.copy(Y_train)
      
        num_stacks = trial.suggest_int('num_stacks', 1, max_num_stacks, step=step)
        num_heads = trial.suggest_int('num_heads', 1, max_num_heads, step=step)
        key_dim = trial.suggest_int('key_dim', 1, max_key_dim, step=step)

        value_dim = trial.suggest_categorical("value_dim", [None, *range(1, max_value_dim, step)])

        dropout = trial.suggest_float('dropout', 0.0, 0.4, step=0.1)
        rate = trial.suggest_float('rate', 0.0, 0.4, step=0.1)

        activation_hide = trial.suggest_categorical('activation_hide', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        activation_feed = trial.suggest_categorical('activation_feed', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        
        n_dropout_hide = trial.suggest_int('n_dropout_hide', 0, max_n_dropout_hide, step=step)
        n_dropout_feed = trial.suggest_int('n_dropout_feed', 0, max_n_dropout_feed, step=step)

        epsilon = trial.suggest_float('epsilon', 1e-9, max_epsilon, log=True)
        num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 0, max_num_neurons_layers_feed, step=step)
        num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 0, max_num_neurons_layers_hide, step=step)
        list_neurons_feed = [trial.suggest_int(f'list_neurons_feed_{i}', 1, max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
        list_neurons_hide = [trial.suggest_int(f'list_neurons_hide_{i}', 1, max_units_neurons_hide) for i in range(num_neurons_layers_hide)]

        learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])

        try:
            clear_session()

            model = create_model(H=H,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            num_heads=num_heads,
                            list_neurons_feed=list_neurons_feed,
                            list_neurons_hide=list_neurons_hide,
                            dropout=dropout,
                            rate=rate,
                            activation_hide=activation_hide,
                            activation_feed=activation_feed,
                            n_dropout_hide=n_dropout_hide,
                            n_dropout_feed=n_dropout_feed,
                            epsilon=epsilon,
                            num_stacks=num_stacks,
                            optimizer=Adam(learning_rate=learning_rate),
                            loss=loss,
                            metrics=metrics)
    
      
            callbacks = []
            
            pruning_callback = TFKerasPruningCallback(trial, monitor)
            early_stopping_callback = EarlyStopping(
                monitor= monitor,  
                patience=patience,         
                mode='min',          
                verbose=verbose,          
                restore_best_weights=restore_best_weights
            )
            
            callbacks.append(pruning_callback)
            callbacks.append(early_stopping_callback)
      
            if np.any(X_train_copy) == None or np.any(Y_train_copy) == None:
                raise ValueError("Something of the data has value None.")
        
            history = model.fit(X_train_copy, Y_train_copy, epochs=epochs, 
                                batch_size=batch_size, verbose=verbose, 
                                validation_split=validation_split, callbacks=callbacks)
            
            clear_session()
            val_monitor = history.history[monitor][-1]

            trial.set_user_attr("history", history.history)
            trial.set_user_attr("monitor", val_monitor)

            return val_monitor
        except (ValueError, 
                MemoryError, 
                RuntimeError, 
                tf.errors.ResourceExhaustedError,
                Exception) as e:      

            # Manejar errores
            print(f"Error en el ensayo {trial.number}: {e}")

            # Limpieza adicional
            clear_session()
            
            # Opcional: Propagar el error para que Optuna lo marque como fallo automáticamente
            raise TrialPruned(f"Ensayo fallido: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argumentos para cargar un backend de optuna")
    
    parser.add_argument('--study_name',
                        type=str,
                        help="Nombre del estudio a cargar")
    parser.add_argument('--storage_name',
                        type=str,
                        help="Ruta del backend")
    parser.add_argument('--path_data',
                        type=str,
                        help="Ruta de los datos")
    parser.add_argument('--dim_data',
                        type=str,
                        help="Dimensión de los datos a recuperar")
    parser.add_argument('--H',
                        type=int,
                        help="Dimensión del modelo")
    parser.add_argument('--path_config_model',
                        type=str,
                        help="Ruta de la configuración del modelo")
    parser.add_argument('--path_config_callbacks',
                        type=str,
                        help="Ruta de la configuración de los callbacks")
    parser.add_argument('--path_config_max_trials',
                        type=str,
                        help="Ruta de la configuración de los valores máximos de los trials")
    
    parser.add_argument('--path_good_params',
                        type=str,
                        required=False,
                        help="Ruta de los parametros buenos",
                        default=None)
    parser.add_argument('--n_trials',
                        type=int,
                        help="Cantidad de pruebas",
                        required=False,
                        default=1)
    parser.add_argument('--n_jobs',
                        type=int,
                        help="Cantidad de trabajos en paralelo",
                        required=False,
                        default=1)
    parser.add_argument('--max_samples',
                        type=int,
                        help="Cantidad máxima de datos",
                        required=False,
                        default=0)
    
    
    args = parser.parse_args()
    
    study = create_study(study_name=args.study_name, 
                         pruner=MedianPruner(),
                         storage=JournalStorage(JournalFileBackend(args.storage_name)),
                         direction="minimize",
                         load_if_exists=True)
    
    if args.path_good_params is not None:
        insert_trials(path_trials=args.path_good_params, study=study)
    
    data_Sx7 = load_data_from_json(args.path_data)
    X_train, Y_train = get_data(data_Sx7, args.dim_data)

    if args.max_samples > 0:
        start = random.randint(0, (X_train.shape[0] - args.max_samples))
        X_train, Y_train = X_train[start : start + args.max_samples], Y_train[start : start + args.max_samples]
    
    config_model = load_json(args.path_config_model)
    config_model["metrics"].append(PercentageSolved())
    config_max_trials = load_json(args.path_config_max_trials)
    config_callbacks = load_json(args.path_config_callbacks)

    os.makedirs(config_callbacks['dir_callbacks'], exist_ok=True)
    
    study.optimize(lambda trial: objective(trial,
                                        H=args.H,
                                        X_train=X_train,
                                        Y_train=Y_train,
                                        **config_callbacks,
                                        **config_max_trials,
                                        **config_model),
                   n_trials=args.n_trials, 
                   show_progress_bar=True,
                   gc_after_trial=True,
                   n_jobs=args.n_jobs,
                   catch=[ValueError,  
                          MemoryError, 
                          RuntimeError, 
                          tf.errors.ResourceExhaustedError])