import os
import time
import numpy as np
import pandas as pd
import argparse
from cpmp_ml.utils import generate_random_layout
from attentional_cpmp.model import load_cpmp_model
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.optimizer import GreedyModel
from copy import deepcopy

# Asegúrate de que estas funciones y clases estén disponibles en tu entorno
# from tu_modulo import load_cpmp_model, generate_random_layout, GreedyModel, AttentionModel

def main(S: int,
         H: int,
         N: int,
         max_steps: int,
         n: int = 1000, 
         models_dir: str = 'models/',
         excel_path: str = 'resumen_modelos.xlsx',
         extension: str = '.h5'):
    # 1. Obtener los modelos .h5 en la carpeta
    model_files = [f for f in os.listdir(models_dir) if f.endswith(extension)]

    # 2. Generar n instancias del problema una sola vez
    instances = np.array([generate_random_layout(S=S, H=H, N=N) for _ in range(n)])

    # 3. Data para el DataFrame final
    summary_data = []

    for model_file in model_files:
        print(f"Calculando metricas modelo: {model_file}")
        model_path = os.path.join(models_dir, model_file)
        model = load_cpmp_model(model_path)
        
        solver = GreedyModel(model=model, data_adapter=AttentionModel())

        start_time = time.time()

        costs, _ = solver.solve(lays=deepcopy(instances), max_steps=max_steps)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Procesar resultados
        resolved_cases = [step for step in costs if step != -1]
        num_resolved = len(resolved_cases)
        resolved_percent = (num_resolved / n) * 100
        mean_steps = np.mean(resolved_cases) if resolved_cases else -1
        median_steps = np.median(resolved_cases) if resolved_cases else -1

        summary_data.append({
            'Modelo': model_file,
            'Porcentaje Resueltos (%)': resolved_percent,
            'Media de Pasos': mean_steps,
            'Mediana de Pasos': median_steps,
            'Tiempo Total (s)': elapsed_time,
            'Casos Resueltos': num_resolved,
            'Total Instancias': n
        })
        print(f"Metricas del modelo {model_file} calculadas")

    # 4. Crear archivo Excel
    df = pd.DataFrame(summary_data)
    df.to_excel(excel_path, index=False)
    print(f"Archivo {excel_path} creado con éxito.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluador de modelos cpmp.")
    parser.add_argument('--S', type=int, required=True, help="Parámetro S para generate_random_layout")
    parser.add_argument('--H', type=int, required=True, help="Parámetro H para generate_random_layout")
    parser.add_argument('--N', type=int, required=True, help="Parámetro N para generate_random_layout")
    parser.add_argument('--n', type=int, default=1000, help="Cantidad de instancias a generar")
    parser.add_argument('--models_dir', type=str, default='models/', help="Directorio donde están los modelos .h5")
    parser.add_argument('--excel_path', type=str, default='resumen_modelos.xlsx', help="Ruta de salida para el archivo Excel")
    parser.add_argument('--extension', type=str, default='.h5', help="Extensión de archivo de los modelos")

    args = parser.parse_args()
    
    main(S=args.S,
         H=args.H,
         N=args.N,
         max_steps=(args.S * (args.H) * 2),
         n=args.n,
         models_dir=args.models_dir,
         excel_path=args.excel_path,
         extension=args.extension)
