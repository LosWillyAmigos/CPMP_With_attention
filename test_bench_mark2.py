import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from attentional_cpmp.model import create_model as create_attention_model
from attentional_cpmp.model import load_cpmp_model
from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import load_data_json
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.utils.adapters import LinealModel
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils.generator import load_simbol
from cpmp_ml.model import create_cpmp_model as create_lineal_model
from cpmp_ml.model import generate_model2
from cpmp_ml.model import generate_model
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.optimizer import GreedyV1
from cpmp_ml.optimizer import GreedyV2
from urllib.parse import quote_plus
from keras.src.models import Model
from cpmp_ml.utils import Layout
from dotenv import load_dotenv
from typing import Callable
from copy import deepcopy
import pandas as pd
import numpy as np
import getpass
import pymongo
import json
import sys

SYSTEM = os.name

def delete_terminal_lines(lines: int):
    for _ in range(lines):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    sys.stdout.flush()

def clear_terminal() -> None:
    if SYSTEM == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def try_again() -> bool:
    act = input('Quiere volver a intentar? (S / N) ').lower()

    delete_terminal_lines(1)

    if act == 's': return True
    return False

def input_label(input_text: str, error_text: str, verify_input: Callable[[str], bool]) -> str:
    input_jump_line_count = input_text.count('\n') + 1
    error_jump_line_count = error_text.count('\n') + 1
    second_error = False

    while True:
        text = input(input_text)
        if verify_input(text):
            break
        
        if not second_error: delete_terminal_lines(input_jump_line_count)
        else: delete_terminal_lines(error_jump_line_count + input_jump_line_count)
        print(error_text)
        second_error = True

    return text

def install_data_route(route) -> None:
    os.makedirs(f'{route}attentional/')
    os.makedirs(f'{route}lineal/')

load_dotenv()
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

MONGO_URI = f'mongodb://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/?authSource={DB_NAME}'
MODEL_CONFIG_ROUTE = os.environ.get("MODEL_CONFIG_ROUTE")
JSON_DATA_ROUTE = os.environ.get("JSON_DATA_ROUTE")
MODEL_ROUTE = os.environ.get("MODEL_ROUTE")
BENCHMARK_ROUTE = os.environ.get("BENCHMARK_ROUTE")
RESULTS_BENCHMARK = os.environ.get("RESULTS_BENCHMARK")

def dis_menu_change_data_route() -> None:
    print('|*|**************| Ruta de los datos |**************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los datos generados.        |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_main_menu() -> None:
    print('|*|************ | Menú de benchmark | ************|*|')
    print('|*| 1) Pruebas con modelo almacenado              |*|')
    print('|*| 2) Pruebas con modelo desde cero              |*|')
    print('|*| 3) Visualizar pruebas                         |*|')
    print('|*| 4) Opciones                                   |*|')
    print('|*| 5) Salir                                      |*|')
    print('|*|*********** | CPMP_With_Attention | ***********|*|')
    print('')

def dis_saved_model_test() -> None:
    print('|*|******* | Pruebas con modelo almacenado | *******|*|')
    print('|*| Este menú le dará apoyo para realizar pruebas   |*|')
    print('|*| con un modelo que tenga almacenado.             |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')
    print('')

def dis_new_model_test() -> None:
    print('|*|******* | Pruebas con modelo desde cero | *******|*|')
    print('|*| Este menú le dará apoyo para realizar pruebas   |*|')
    print('|*| con un modelo que se creará desde cero.         |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')
    print('')

def dis_view_tests() -> None:
    print('|*|************* | Visualizar pruebas | ************|*|')
    print('|*| Este menú le permitirá visualizar los           |*|')
    print('|*| resultados de las pruebas realizadas a través   |*|')
    print('|*| de una tabla mostrada por terminal.             |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')

def dis_options() -> None:
    print('|*|*************** | Opciones | *****************|*|')
    print('|*| 1) Cambiar ruta de los casos de prueba       |*|')
    print('|*| 2) Cambiar ruta de los modelos               |*|')
    print('|*| 3) Cambiar ruta de los resultados            |*|')
    print('|*| 4) Volver al menú principal                  |*|')
    print('|*|********** | CPMP_With_Attention | ***********|*|')
    print('')

def dis_menu_change_model_route() -> None:
    print('|*|*************| Ruta de los modelos |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los modelos.                |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_menu_change_benchmark_route() -> None:
    print('|*|*************| Ruta de los casos de prueba |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta donde se   |*|')
    print('|*| encuentran los casos de prueba.                         |*|')
    print('|*|*****************| CPMP_With_Attention |*****************|*|')
    print('')

def dis_select_model() -> None:
    print('|*|***************| Selección de modelo |***************|*|')
    print('|*| Este menú le dará apoyo para seleccionar el modelo  |*|')
    print('|*| que desea utilizar para la generación de datos.     |*|')
    print('|*|***************| CPMP_With_Attention |***************|*|')
    print('')

def dis_show_directories() -> None:
    print('|*|********************| Directorios |*********************|*|')
    print('|*| Este menú le permitirá visualizar los directorios      |*|')
    print('|*| de los casos de prueba almacenados, escoja el a través |*|')
    print('|*| de un número los casos de prueba que desee utilizar.   |*|')
    print('|*|****************| CPMP_With_Attention |*****************|*|')
    print('')

def dis_laod_new_model() -> None:
    print('|*|***************| Cargar nuevo modelo |***************|*|')
    print('|*| Este menú le dará apoyo para cargar un nuevo modelo |*|')
    print('|*| que se creará desde cero.                           |*|')
    print('|*| Para esto tendrá que indicar todas los parámetros   |*|')
    print('|*| necesarios para la creación del modelo.             |*|')
    print('|*|***************| CPMP_With_Attention |***************|*|')
    print('')

def dis_train_new_model() -> None:
    print('|*|***************| Entrenar nuevo modelo |***************|*|')
    print('|*| Este menú le dará apoyo para entrenar un nuevo modelo |*|')
    print('|*| que se creará desde cero.                             |*|')
    print('|*| Elija una de las opciones que aparece a continuación. |*|')
    print('|*|                                                       |*|')
    print('|*| 1) Entrenar modelo con datos de MongoDB               |*|')
    print('|*| 2) Entrenar modelo con datos en JSON                  |*|')
    print('|*|****************| CPMP_With_Attention |****************|*|')
    print('')

def dis_menu_change_mongodb_uri() -> None:
    print('|*|*************| URI de MongoDB |***************|*|')
    print('|*| Este menú le dará apoyo para cambiar la URI  |*|')
    print('|*| de conexión a MongoDB.                       |*|')
    print('|*|***********| CPMP_With_Attention |************|*|')
    print('')

def read_benchmark_file(route: str, H: int | None = None) -> tuple[Layout, int, int, int]:
    with open(route) as file:
        S, N = (int(x) for x in next(file).split())
        if H is None: H = int((N / S) + 2)

        stacks = []
        for line in file:
            stack = [int(x) for x in line.split()[1::]]
            stacks.append(stack)
        
        lay = Layout(stacks, H)

    return lay, int(S), int(H), int(N)

def input_password() -> str:
    while True:
        temp = getpass.getpass('Introduce tu contraseña: ')
        temp_1 = getpass.getpass('Confirme tu contraseña: ')
        
        if temp != temp_1:
            print('Las contraseñas no coinciden.')
            input('Pulse Enter para continuar')
            
            delete_terminal_lines(4)
            continue
            
        break

    return temp

def change_data_route_menu() -> None:
    global JSON_DATA_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_data_route()

        print(f'Actual ruta de los datos: {JSON_DATA_ROUTE}\n')

        data_route = input('Indique la nueva ruta de los datos (puede ser la misma): ')
        if not os.path.exists(data_route): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los datos? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()

        if act == 's': 
            install_data_route(data_route)
            print('Ruta creada con éxito!')
        else: 
            print('Ruta no creada.')
            input('Pulse Enter para continuar')
            if try_again(): continue
        
        break

    input('Pulse Enter para continuar')
    JSON_DATA_ROUTE = data_route

def change_uri_mongo_menu() -> None:
    global MONGO_URI
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME

    while True:
        clear_terminal()
        dis_menu_change_mongodb_uri()

        print(f'Actual datos de MongoDB:')
        print(f'1) HOST: {DB_HOST}')
        print(f'2) USER: {DB_USER}')
        print(f"3) PASSWORD: {'*' * len(DB_PASSWORD)}")
        print(f'4) AUTENTICATION DATABASE: {DB_NAME}\n')

        act = input_label('Desea cambiar los datos de MongoDB? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n': break

        opt = int(input_label('Que dato desea cambiar? (1 - 4) ', 
                          'Por favor, coloque un número entero.', 
                          lambda x: x.isdigit() and 0 < int(x) <= 4))
        
        if opt == 1:
            DB_HOST = input('Indique la IP del servidor mongodb (IP:Port): ')
        elif opt == 2:
            DB_USER = input('Indique su nombre de usuario: ')
        elif opt == 3:
            DB_PASSWORD = input_password()
        elif opt == 4:
            DB_NAME = input('Ingrese el nombre de la base de datos de autenticación: ')

        MONGO_URI = f'mongodb://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/?authSource={DB_NAME}'
        print('Datos cambiados con éxito!')

        act = input_label('Desea cambiar otro dato? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n': break

def change_model_route_menu() -> None:
    global MODEL_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_model_route()

        print(f'Actual ruta de los modelos: {MODEL_ROUTE}\n')
        
        model_route = input('Indique la nueva ruta de los modelos (puede ser la misma): ').replace('\\', '/')
        if not model_route.endswith('/'): model_route += '/'
        if not os.path.exists(model_route): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los modelos? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()

        if act == 's': 
            install_data_route(model_route)
            print('Ruta creada con éxito!')
            break
        else: 
            print('Ruta no creada.')
            if try_again(): continue
            break

    input('Pulse Enter para continuar')
    MODEL_ROUTE = model_route

def change_benchmark_route() -> None:
    global BENCHMARK_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_benchmark_route()

        print(f'Actual ruta de los casos de prueba: {BENCHMARK_ROUTE}\n')

        benchmark_path = input('Indique la nueva ruta de los casos de prueba (puede ser la misma): ').replace('\\', '/')
        if not benchmark_path.endswith('/'): benchmark_path += '/'
        if not os.path.exists(benchmark_path): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los casos de prueba? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 's':
            os.mkdir(benchmark_path)
            break
        else:
            print('Ruta no creada.')
            if try_again(): continue
            break

    input('Pulse Enter para continuar')
    BENCHMARK_ROUTE = benchmark_path

def get_models(route: str) -> dict:
    models = dict()
    models_names = os.listdir(route)

    if len(models_names) == 0: return None

    for i, model in enumerate(models_names):
        models.update({i + 1: model})
        print(f'{i + 1}) {model}')
    print('')

    return models

def read_model_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = json.load(file)

    return config

def select_model(S: int, H: int, adapter: str) -> tuple[Model, dict] | None:
    while True:
        clear_terminal()
        dis_select_model()
        
        if not os.path.exists(MODEL_ROUTE):
            print('La ruta de los modelos no existe.')
            opt = input_label('Desea cambiar la ruta de los modelos? (S / N) ', 
                              'Por favor, coloque una opción válida.', 
                              lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_model_route_menu()
                continue

            return None, None

        if adapter == 'attentionmodel' and os.path.exists(f'{MODEL_ROUTE}attentional/Sx{H}/'): models = get_models(f'{MODEL_ROUTE}attentional/Sx{H}/')
        elif adapter == 'attentionmodel' and not os.path.exists(f'{MODEL_ROUTE}attentional/Sx{H}/'): 
            os.mkdir(f'{MODEL_ROUTE}attentional/Sx{H}/')
            models = get_models(f'{MODEL_ROUTE}attentional/Sx{H}/')

        if adapter == 'linealmodel' and os.path.exists(f'{MODEL_ROUTE}lineal/{S}x{H}/'): models = get_models(f'{MODEL_ROUTE}lineal/{S}x{H}/')
        elif adapter == 'linealmodel' and not os.path.exists(f'{MODEL_ROUTE}lineal/{S}x{H}/'): 
            os.mkdir(f'{MODEL_ROUTE}lineal/{S}x{H}/')
            models = get_models(f'{MODEL_ROUTE}lineal/{S}x{H}/')

        if models is None: 
            if adapter == 'attentionmodel': print(f'No hay modelos para seleccionar con altura {H}.')
            if adapter == 'linealmodel': print(f'No hay modelos para seleccionar con altura {H} y {S} stacks.')
            return None, None

        num_model = int(input_label('Seleccione el modelo que desea utilizar: ', 
                                'Por favor, coloque un número entero.', 
                                lambda x: x.isdigit() and 0 < int(x) <= len(models) + 1))

        if adapter == 'attentionmodel':
            model = load_cpmp_model(f'{MODEL_ROUTE}attentional/Sx{H}/{models[num_model]}')
            model_conf = read_model_config(f'{MODEL_CONFIG_ROUTE}attentional/Sx{H}/{models[num_model].replace(".keras", ".json")}')
        if adapter == 'linealmodel': 
            model = load_cpmp_model(f'{MODEL_ROUTE}lineal{S}x{H}/{models[num_model]}')
            model_conf = read_model_config(f'{MODEL_CONFIG_ROUTE}lineal{S}x{H}/{models[num_model].replace(".keras", ".json")}')
        
        model_conf.update({'total_params': model.count_params()})
        return model, model_conf
    
def show_benchmarks_directories() -> str:
    benchmarks = dict()
    temp = dict()
    benchmarks_direct = os.listdir(BENCHMARK_ROUTE)

    if len(benchmarks_direct) == 0: return None

    for i, direct in enumerate(benchmarks_direct):
        temp.update({i + 1: direct})
        print(f'{i + 1}) {direct}')
    print('')

    test_type = int(input_label('Ingrese el tipo de casos de prueba: ', 
                                'Por favor, coloque un número entero.', 
                                lambda x: x.isdigit() and int(x) > 0 and int(x) <= len(os.listdir(BENCHMARK_ROUTE))))
    
    for i, direct in enumerate(os.listdir(f'{BENCHMARK_ROUTE}{temp[int(test_type)]}')):
        benchmarks.update({i + 1: direct})
        print(f'{i + 1}) {direct}')

    if benchmarks == {}: return None

    opt = int(input_label('Seleccione el directorio de prueba que desea utilizar: ',
                          'Por favor, coloque un número entero.', 
                          lambda x: x.isdigit() and int(x) > 0 and int(x) <= len(os.listdir(f'{BENCHMARK_ROUTE}{temp[int(test_type)]}'))))

    
    return temp[test_type] + '/' + benchmarks[opt]

def read_optimal_solution(route: str) -> int | pd.DataFrame:
    if route.endswith('.txt'):
        with open(route, 'r') as file:
            optimal = int(file.readline())
    elif route.endswith('.xlsx'):
        optimal = pd.read_excel(route, usecols='A:B')
        
    return optimal

def load_problems(path: str, total_problems: int, H: int | None = None) -> tuple[list[Layout], int, int, int, int | pd.DataFrame]:
    problems = []
    cont = 0

    for file in os.scandir(path):
        if cont == total_problems: break
        if file.is_file() and (file.name.endswith('.txt') or file.name.endswith('.xlsx')):
            optimal = read_optimal_solution(file.path)
        if file.is_file() and (file.name.endswith('.dat') or file.name.endswith('.bay')):
            lay, S, H, N = read_benchmark_file(file.path, H)
            problems.append(lay)

            cont += 1

    return problems, S, H, N, optimal

def create_flags(lists_costs: list[np.ndarray[int]]) -> list:
    filter_costs = [True for _ in range(len(lists_costs[0]))]

    for costs in lists_costs:
        for i in range(len(costs)):
            if costs[i] == -1:
                filter_costs[i] = False

    return filter_costs

def filter_costs(lists_costs: list[np.ndarray[int]]) -> tuple[list[int]]:
    flags = create_flags(lists_costs)
    filter_opt = []

    for i in range(len(lists_costs)):
        filter_opt.append([])
        for j in range(len(lists_costs[i])):
            if flags[j]: filter_opt[i].append(lists_costs[i][j])
            elif not flags[j]: filter_opt[i].append(-1)

    return filter_opt

def create_report(
        problems_name: str,
        size_problems: int,
        S: int,
        H: int,
        N: int,
        optimal: int | pd.DataFrame,
        costs_optimizers: list[np.ndarray[int]],
        model_config: dict,
) -> pd.DataFrame | None:
    results = pd.DataFrame()
    
    try:
        results['Instance'] = [f'{problems_name}-{i + 1}' for i in range(size_problems)]
        results['S'] = [S for _ in range(size_problems)]
        results['H'] = [H for _ in range(size_problems)]
        results['N'] = [N for _ in range(size_problems)]
        results['Optimal'] = optimal['Cantidad de movimientos'][:size_problems]
        results['key_dim'] = [model_config['key_dim'] for _ in range(size_problems)]
        results['value_dim'] = [model_config['value_dim'] for _ in range(size_problems)]
        results['num_heads'] = [model_config['num_heads'] for _ in range(size_problems)]
        results['num_stacks'] = [model_config['num_stacks'] for _ in range(size_problems)]
        results['epsilon'] = [model_config['epsilon'] for _ in range(size_problems)]
        results['total_params'] = [model_config['total_params'] for _ in range(size_problems)]
        results['greedyv1'] = costs_optimizers[1]
        results['greedyv2'] = costs_optimizers[2]
        results['greedymodel'] = costs_optimizers[0]
    except KeyboardInterrupt:
        print('Se ha cancelado la ejecución del benchmark.')
        return None

    return results
def run_experiments(
        problems: list[Layout], 
        problems_name: str, 
        S: int, 
        H: int, 
        N: int, 
        optimal: int | pd.DataFrame,
        model: Model,
        adapter: DataAdapter,
        model_config: dict = None,
        verbose: bool = False
) -> None:
    if verbose: load_simbol(1, 6, 'Procesos: ')
    greedy1 = GreedyV1()
    greedy2 = GreedyV2()
    greedy_model = GreedyModel(model, adapter)

    if verbose: load_simbol(2, 6, 'Procesos: ')
    cost_g1 = greedy1.solve(np.array(deepcopy(problems)))[0]
    if verbose: load_simbol(3, 6, 'Procesos: ')
    cost_g2 = greedy2.solve(np.array(deepcopy(problems)), max_steps= N * 2)[0]
    if verbose: load_simbol(4, 6, 'Procesos: ')
    cost_gmodel = greedy_model.solve(np.array(deepcopy(problems)), max_steps= N * 2)[0]

    if verbose: load_simbol(5, 6, 'Procesos: ')
    report = create_report(
        problems_name,
        len(problems),
        S,
        H,
        N,
        optimal,
        [cost_gmodel, cost_g1, cost_g2],
        model_config
    )

    if verbose: load_simbol(6, 6, 'Procesos: ')
    return report

def saved_model_test() -> None:
    while True:
        clear_terminal()
        dis_saved_model_test()

        dis_show_directories()
        directory = show_benchmarks_directories()
        if directory is None: 
            print('No hay casos de prueba para seleccionar.')
            opt = input_label('Desea cambiar la ruta de los casos de prueba? (S / N) ',
                              'Por favor, coloque una opción válida.',
                              lambda x: x.lower() in ['s', 'n']).lower()
            
            if opt == 's':
                change_benchmark_route()
                continue
            else: break
        
        total_cases = len(os.listdir(f'{BENCHMARK_ROUTE}{directory}')) - 1

        instance_name = input('Indique el nombre de la instancia: ')
        
        size_problems = int(input_label(f'Ingrese la cantidad de problemas que desea probar (1 - {total_cases}): ',
                                        f'Por favor, coloque un número entero entre el 1 y el {total_cases}.',
                                        lambda x: x.isdigit() and 1 <= int(x) <= total_cases))
        
        selected_adapter = input_label('Que adaptador necesitas? (AttentionModel, LinealModel) ', 
                                      'Por favor, coloque un adaptador válido.', 
                                      lambda x: x.lower() in ['attentionmodel', 'linealmodel']).lower()

        verbose = input_label('Desea ver como progresa la ejecución del benchmark? (S / N) ', 
                             'Por favor, coloque una opción válida.', 
                             lambda x: x.lower() in ['s', 'n']).lower()
        
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n':
            clear_terminal()
            return
        
        if selected_adapter == 'attentionmodel': adapter = AttentionModel()
        elif selected_adapter == 'linealmodel': adapter = LinealModel()

        problems, S, H, N, optimal = load_problems(f'{BENCHMARK_ROUTE}{directory}/', size_problems)
        
        model, model_config = select_model(S, H, selected_adapter)
        if model is None: 
            if try_again(): continue
            else: break

        results = run_experiments(
            problems, 
            instance_name, S, H, N, 
            optimal, model, adapter,
            model_config, verbose
        )
        if results is not None: 
            results.to_excel(f'{RESULTS_BENCHMARK}benchmarks2.xlsx', index=False)
            print('Pruebas realizadas con éxito.\n')

        if try_again(): continue
        else: break

def view_tests() -> None:
    pass

def options() -> None:
    pass

def input_attention_model_params() -> dict | None:
    while True:
        H = int(input_label('Ingrese la altura de los porblemas que el modelo debe resolver (H ≥ 3): ',
                            'Por favor, coloque un número entero mayor 3.', 
                            lambda x: x.isdigit() and int(x) >= 3))
        key_dim = int(input_label('Ingrese el valor de key_dim: ', 
                                'Por favor, coloque un número entero.', 
                                lambda x: x.isdigit() and int(x) > 0))
        value_dim = int(input_label('Ingrese el valor de value_dim: ', 
                                    'Por favor, coloque un número entero.', 
                                    lambda x: x.isdigit() and int(x) > 0))
        num_heads = int(input_label('Ingrese el valor de num_heads: ', 
                                    'Por favor, coloque un número entero.', 
                                    lambda x: x.isdigit() and int(x) > 1))
        num_stacks = int(input_label('Ingrese el valor de num_stacks: ', 
                                    'Por favor, coloque un número entero.', 
                                    lambda x: x.isdigit() and int(x) > 1))
        epsilon = float(input_label('Ingrese el valor de epsilon: ', 
                                    'Por favor, coloque un número decimal.', 
                                    lambda x: x.replace('.', '', 1).isdigit()))
        list_neuron_hide = input_label('Ingrese la lista de neuronas ocultas en formato [1,2,...,n]: ', 
                                    'Por favor, coloque una lista de números enteros.', 
                                    lambda x: x[0] == '[' and x[-1] == ']'   \
                                    and all([i.isdigit() for i in x[1:-1].split(',')]))
        activation_hide = input_label('Ingrese la función de activación de las neuronas ocultas: ', 
                                    'Por favor, coloque una función de activación válida.', 
                                    lambda x: x.lower() in ['relu', 'tanh', 'sigmoid', 'linear'])
        list_neuron_feed = input_label('Ingrese la lista de neuronas feedforward en formato [1,2,...,n]: ', 
                                    'Por favor, coloque una lista de números enteros.', 
                                    lambda x: x[0] == '[' and x[-1] == ']'   \
                                    and all([i.isdigit() for i in x[1:-1].split(',')]))
        activation_feed = input_label('Ingrese la función de activación de las neuronas feedforward: ', 
                                    'Por favor, coloque una función de activación válida.', 
                                    lambda x: x.lower() in ['relu', 'tanh', 'sigmoid', 'linear'])
        dropout = float(input_label('Ingrese el valor de dropout (0 ≤ n ≤ 1): ', 
                                    'Por favor, coloque un número decimal.', 
                                    lambda x: x.replace('.', '', 1).isdigit()   \
                                    and 0 <= float(x) <= 1))
        n_dropout_hide = int(input_label('Ingrese cuantas capas dropout desea en las capas ocultas: ', 
                                        'Por favor, coloque un número entero mayor o igual a 0.', 
                                        lambda x: x.isdigit() and int(x) >= 0))
        n_dropout_feed = int(input_label('Ingrese cuantas capas dropout desea en las capas feedforward: ', 
                                        'Por favor, coloque un número entero mayor o igual a 0.', 
                                        lambda x: x.isdigit() and int(x) >= 0))
        optimizer = input_label('Ingrese el optimizador que desea utilizar: ', 
                                'Por favor, coloque un optimizador válido.', 
                                lambda x: x.lower() in ['adam', 'rmsprop', 'sgd'])
        loss = input_label('Ingrese la función de pérdida que desea utilizar: ', 
                        'Por favor, coloque una función de pérdida válida.', 
                        lambda x: x.lower() in ['mse', 'mae', 'mape', 'msle', 'binary_crossentropy'])
        metrics = input_label('Ingrese las métricas que desea utilizar en formato [metrica1, metrica2,..., metricaN]: ', 
                            'Por favor, coloque una métrica válida.', 
                                lambda x: x[0] == '[' and x[-1] == ']'   \
                                and all([i.lower() in ['mse', 'mae', 'mape', 'msle', 'binary_crossentropy'] for i in x[1:-1].split(',')]))
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                            'Por favor, coloque una opción válida.', 
                            lambda x: x.lower() in ['s', 'n']).lower()
            
        if act == 'n':
            if try_again(): continue
            return None
        elif act == 's': break
    
    return {
        'H': H, 
        'key_dim': key_dim,
        'value_dim': value_dim,
        'num_heads': num_heads,
        'num_stacks': num_stacks,
        'epsilon': epsilon,
        'list_neuron_hide': [int(i) for i in list_neuron_hide[1:-1].split(',')],
        'activation_hide': activation_hide,
        'list_neuron_feed': [int(i) for i in list_neuron_feed[1:-1].split(',')],
        'activation_feed': activation_feed,
        'dropout': dropout,
        'n_dropout_hide': n_dropout_hide,
        'n_dropout_feed': n_dropout_feed,
        'optimizer': optimizer,
        'loss': loss,
        'metrics': [i for i in metrics[1:-1].split(',')]
    }

def input_lineal_model_params() -> dict | None:
    while True:
        S = int(input_label('Ingrese la cantidad de stacks que desea utilizar: ', 
                            'Por favor, coloque un número entero mayor a 3.', 
                            lambda x: x.isdigit() and int(x) > 3))
        H = int(input_label('Ingrese la altura de los porblemas que el modelo debe resolver (H ≥ 3): ',
                            'Por favor, coloque un número entero mayor 3.', 
                            lambda x: x.isdigit() and int(x) >= 3))
        generator = input_label('Ingrese el generador que desea utilizar (generate_model, generate_model2): ', 
                                'Por favor, coloque un generador válido.', 
                                lambda x: x.lower() in ['generate_model', 'generate_model2'])
        act = input_label('Está seguro de sus elecciones? (S / N) ',
                            'Por favor, coloque una opción válida.', 
                            lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n':
            if try_again(): continue
            return None
        elif act == 's': break
    
    return {
        'S': S,
        'H': H,
        'generator': generate_model if generator == 'generate_model' else generate_model2
    }

def show_collections(client: pymongo.MongoClient, data_base_name: str) -> list:
    collections = dict()
    i = 1

    print('|*|**********| Colecciones disponibles |**********|*|')
    for collection in client[data_base_name].list_collection_names():
        if len(collection) > 39: print(f'|*| {i}) {collection[:39]}... |*|')
        else: print(f'|*| {i}) {collection}{" " * (42 - len(collection))} |*|')

        collections.update({i: collection})
        i += 1
    print('|*|************| CPMP_With_Attention |************|*|')
    print('')
    if len(collections): return collections
    return None

def show_data_bases(client: pymongo.MongoClient) -> dict:
    data_bases = dict()
    i = 1

    print('|*|********| Bases de datos disponibles |*********|*|')
    for data_base in client.list_database_names():
        if data_base == 'admin' or data_base == 'config' or data_base == 'local': continue

        if len(data_base) > 39:print(f'|*| {i}) {data_base[:39]}... |*|')
        else: print(f'|*| {i}) {data_base}{" " * (42 - len(data_base))} |*|')

        data_bases.update({i: data_base})
        i += 1
    print('|*|***********| CPMP_With_Attention |*************|*|')
    print('')

    if len(data_bases): return data_bases
    return None

def show_data_json(path: str) -> list:
    json_files = dict()

    print('|*|***************| Datos en JSON |***************|*|')
    for i, file in enumerate(os.listdir(path)):
        if not file.endswith('.json'): continue

        if len(file) > 39:print(f'|*| {i}) {file[:39]}... |*|')
        else: print(f'|*| {i + 1}) {file}{" " * (42 - len(file))} |*|')

        json_files.update({i + 1: file})
    print('|*|***********| CPMP_With_Attention |*************|*|')
    print('')

    if len(json_files): return json_files
    return None

def data_mongodb() -> dict | None:
    while True:
        client = pymongo.MongoClient(MONGO_URI)
        if client is None and try_again():
            print('La conexión no se pudo realizar.')
            opt = input_label('Desea cambiar la URI de MongoDB? (S / N) ',
                                'Por favor, coloque una opción válida.', 
                                lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_uri_mongo_menu()
                continue

            return
        
        data_bases = show_data_bases(client)
        if data_bases is None: 
            print('No hay bases de datos disponibles.')
            input('Pulse Enter para continuar')
            
            client.close()
            
            return False
        
        data_base_num = input_label(f'Indique el número de la base de datos: ', 
                                     'La base de datos no existe o el valor ingresado es invalido.',
                                     lambda x: x.isdigit() and 0 < int(x) <= len(data_bases))

        collections = show_collections(client, data_bases[int(data_base_num)])
        if collections is None:
            print('No hay colecciones disponibles.')
            input('Pulse Enter para continuar')
            
            client.close()
            
            return False
        
        collection_num = input_label('Indique el número de la colección: ', 
                                     'La colección no existe o el valor ingresado es invalido.',
                                     lambda x: x.isdigit() and 0 < int(x) <= len(collections))
        data_base = client[data_bases[int(data_base_num)]]

        print('Iniciando Carga de datos...')
        data = load_data_mongo(data_base[collections[int(collection_num)]], True)
        
        if data is not None: break
        elif try_again(): continue   

    return data

def data_json(adapter: str) -> dict:
    while True:
        if not os.path.exists(JSON_DATA_ROUTE):
            print('La ruta de los datos no existe.')
            opt = input_label('Desea cambiar la ruta? (S / N) ', 
                            'Por favor, coloque una opción válida.', 
                            lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_data_route_menu()
                continue

            input('Pulse Enter para continuar')
            return False

        if adapter == 'attentionmodel': base_path = JSON_DATA_ROUTE + 'attentional/'
        elif adapter == 'linealmodel': base_path = JSON_DATA_ROUTE + 'lineal/'

        files = show_data_json(base_path)
        if files is None:
            print('No hay archivos JSON disponibles.')
            input('Pulse Enter para continuar')
            return False
        
        file_num = int(input_label('Indique el número del archivo JSON: ', 
                                'El archivo no existe o el valor ingresado es invalido.',
                                lambda x: x.isdigit() and 0 < int(x) <= len(files)))

        print('Iniciando Carga de datos...')
        data = load_data_json(os.path.join(base_path + files[file_num]))
        
        return data
        
def train_new_model(model: Model, verbose: bool = False) -> None:
    while True:
        clear_terminal()
        dis_train_new_model()

        opt = int(input_label('Seleccione una opción: ', 
                              'Por favor, coloque un número entero.', 
                              lambda x: x.isdigit() and 0 < int(x) <= 2))
        
        if opt == 1:
            data = data_mongodb()
            if data is None:
                if try_again(): continue
                else: break
        elif opt == 2:
            data = data_json()
            if data is None:
                if try_again(): continue
                else: break

        model.train(data, verbose)
        
def load_new_model(adapter: str) -> Model | None:
    if adapter == 'attentionmodel': config_model = input_attention_model_params()
    elif adapter == 'linealmodel': config_model = input_lineal_model_params()

    if config_model is None: return None, None

    if adapter == 'attentionmodel': 
        model = create_attention_model(
            H= config_model['H'],
            key_dim= config_model['key_dim'],
            value_dim= config_model['value_dim'],
            num_heads= config_model['num_heads'],
            num_stacks= config_model['num_stacks'],
            epsilon= config_model['epsilon'],
            list_neurons_hide= config_model['list_neuron_hide'],
            activation_hide= config_model['activation_hide'],
            list_neurons_feed= config_model['list_neuron_feed'],
            activation_feed= config_model['activation_feed'],
            dropout= config_model['dropout'],
            n_dropout_hide= config_model['n_dropout_hide'],
            n_dropout_feed= config_model['n_dropout_feed'],
            optimizer= config_model['optimizer'],
            loss= config_model['loss'],
            metrics= config_model['metrics']
        )
    elif adapter == 'linealmodel':
        model = create_lineal_model(
            S= config_model['S'],
            H= config_model['H'],
            generate_model= config_model['generator']
        )

    return model, config_model

def new_model_test() -> None:
    while True:
        clear_terminal()
        dis_new_model_test()

        dis_show_directories()
        directory = show_benchmarks_directories()
        if directory is None: 
            print('No hay casos de prueba para seleccionar.')
            opt = input_label('Desea cambiar la ruta de los casos de prueba? (S / N) ',
                              'Por favor, coloque una opción válida.',
                              lambda x: x.lower() in ['s', 'n']).lower()
            
            if opt == 's':
                change_benchmark_route()
                continue
            else: break
        
        total_cases = len(os.listdir(f'{BENCHMARK_ROUTE}{directory}')) - 1

        instance_name = input('Indique el nombre de la instancia: ')
        
        size_problems = int(input_label(f'Ingrese la cantidad de problemas que desea probar (1 - {total_cases}): ',
                                        f'Por favor, coloque un número entero entre el 1 y el {total_cases}.',
                                        lambda x: x.isdigit() and 1 <= int(x) <= total_cases))
        
        selected_adapter = input_label('Que adaptador necesitas? (AttentionModel, LinealModel) ', 
                                       'Por favor, coloque un adaptador válido.', 
                                       lambda x: x.lower() in ['attentionmodel', 'linealmodel']).lower()

        verbose = input_label('Desea ver como progresa la ejecución del benchmark? (S / N) ', 
                              'Por favor, coloque una opción válida.', 
                              lambda x: x.lower() in ['s', 'n']).lower()
        
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n':
            clear_terminal()
            return
        
        if selected_adapter == 'attentionmodel': adapter = AttentionModel()
        elif selected_adapter == 'linealmodel': adapter = LinealModel()

        model, model_config = load_new_model(selected_adapter)
        if model is None:
            print('No se ha podido cargar el modelo.')
            if try_again(): continue
            else: break

        problems, S, H, N, optimal = load_problems(f'{BENCHMARK_ROUTE}{directory}/', size_problems)

        results = run_experiments(
            problems, 
            instance_name, S, H, N, 
            optimal, model, adapter, 
            model_config, verbose
        )
        if results is not None: 
            results.to_excel(f'{RESULTS_BENCHMARK}benchmarks2.xlsx', index=False)
            print('Pruebas realizadas con éxito.\n')

        if try_again(): continue
        else: break

def main_menu():
    try:
        while True:
            clear_terminal()
            dis_main_menu()
    
            option = int(input_label('Ingrese una opción: ', 
                                     'Coloque una opción valida (1 - 5)',
                                      lambda x: x.isdigit() and 1 <= int(x) <= 5))
            
            if option == 1:
                saved_model_test()
            elif option == 2:
                new_model_test()
            elif option == 3:
                continue
            elif option == 4:
                continue
            elif option == 5:
                print('\nSaliendo del programa...')
                break
    except KeyboardInterrupt:
        print('\nSaliendo del programa...')
        exit()

if __name__ == '__main__':
    saved_model_test()
