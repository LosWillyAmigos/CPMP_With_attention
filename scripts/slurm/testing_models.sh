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
source ./../../venv/bin/activate

# Ejecutar el script con argumento
python ./../python/testing_models.py \
    --S 5 \
    --H 7 \
    --N 25 \
    --n 2000 \
    --models_dir ./../models/ \
    --excel_path ./../../resultados_modelos.xlsx

deactivate

      