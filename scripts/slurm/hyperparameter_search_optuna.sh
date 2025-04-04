#!/bin/bash
#SBATCH --job-name=ignacio_araya_job # Nombre del trabajo
#SBATCH --output=output_%j.log      # Archivo de salida
#SBATCH --error=error_%j.log        # Archivo de error
#SBATCH --time=24:00:00             # Tiempo máximo de ejecución (24 horas)
#SBATCH --partition=CPU             # Cola "CPU" (cola por defecto)
#SBATCH --nodes=1                   # Número de nodos
#SBATCH --ntasks=1                  # Número de tareas
#SBATCH --cpus-per-task=40          # Número de CPUs por tarea
#SBATCH --mem=32GB                  # Memoria total asignada
#SBATCH --qos=normal                # QoS normal

# Cargar el módulo de Python (si es necesario en tu entorno SLURM)
module load python/3.11.5            # Ajusta al módulo de Python disponible

# Activar el entorno virtual existente
source env/bin/activate

# Ejecutar el script con argumento
python ./../python/create_and_optimize_study_with_one_objective.py \
    --study_name hyperparameter_search_1 \
    --storage_name ./../storage/hyperparameter_search_1.json \
    --path_data ./../../data/CPMP_With_Attention.Sx7_v4.json \
    --dim_data 5 \
    --H 7 \
    --path_config_model ./../../data/config_model.json \
    --path_config_callbacks ./../../data/config_callbacks.json \
    --path_config_max_trials ./../../data/max_trials_setting.json \
    --path_good_params ./../../data/initial_hyperparameters.json \
    --n_trials 200 \
    --n_jobs 2 \

deactivate
