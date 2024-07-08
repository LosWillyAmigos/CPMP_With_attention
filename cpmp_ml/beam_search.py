import numpy as np
from keras.models import Model
from keras import backend as K
from CPMP.cpmp_ml import Layout
from copy import deepcopy
import time

def get_ann_state(layout: Layout) -> np.ndarray:
    S=len(layout.stacks) # Cantidad de stacks

    #matriz de stacks
    b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
    for i,j in enumerate(layout.stacks):
        b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
        b[i][0] = layout.is_sorted_stack(i)
    b.shape=(S,(layout.H + 1))
    
    return np.stack(b)

def init_session(lays: list[Layout]) -> list[tuple]:
    lays_size = len(lays)
    new_lays = []

    for i in range(lays_size):
        new_lays.append((lays[i], i))

    return new_lays

def search_solution(state, solutions):
    for case in solutions:
        if state == case.stacks: return True

    return False

def add_solution(solutions: dict, lay_solution: list[tuple]) -> dict:
    solutions_size = len(solutions)

    if solutions_size == 0 or not str(lay_solution[1]) in solutions:
        solutions.update({str(lay_solution[1]): [lay_solution[0]]})
    elif not search_solution(lay_solution[0].stacks, solutions[str(lay_solution[1])]):
        solutions[str(lay_solution[1])].append(lay_solution[0])
    
    return solutions

def prune_tree(index: list[int], lays: list[tuple], predict: list[np.array]):
    new_lays = [lay for lay in lays if lay[1] not in index]
    new_pred = None

    if not predict is None:
        new_pred = []
        for case in predict:
            if not int(case) in index:
                new_pred.append(np.array(predict[case]))

    return new_lays, new_pred

def init_pred_with_index(lays, pred):
    pred_size = len(pred)
    new_pred = dict()

    k = 0
    for i in range(pred_size):
        stack_size = pred[i].shape[0]

        for j in range(stack_size):
            if len(new_pred) == 0 or not str(lays[k][1]) in new_pred:
                new_pred.update({str(lays[k][1]): []})
            
            new_pred[str(lays[k][1])].append(pred[i][j])
            k += 1

    return new_pred

def verify_solution(lays: list[tuple], solutions: dict, predict: list[np.array] = None) -> dict:
    if not predict is None: predict = init_pred_with_index(lays, predict)
    lays_size = len(lays)
    index = []

    for i in range(lays_size):
        if lays[i][0].unsorted_stacks == 0:
            solutions = add_solution(solutions, lays[i])
            if lays[i][1] not in index: index.append(lays[i][1])

    new_lays, new_predict = prune_tree(index, lays, predict)

    if not predict is None: return new_lays, new_predict, solutions
    return new_lays, solutions

def get_model_lays(lays: list[tuple]) -> list:
    model_lays = [get_ann_state(lay[0]) for lay in lays]

    return np.stack(model_lays)

def expanded_lays(model: Model, lays: list[tuple]):
    lays_size = len(lays)
    child_lays = []
    new_pred_lays = []

    model_lays = get_model_lays(lays)
    pred_lays = model.predict(model_lays, verbose= False)
    K.clear_session()

    for k in range(lays_size):
        stack_size = len(lays[k][0].stacks)
        new_pred_lays.append([])

        z = 0
        for i in range(stack_size):
            for j in range(stack_size):
                if i == j: continue

                new_lay = deepcopy(lays[k][0])
                if new_lay.move((i, j)) is None: continue
                child_lays.append((new_lay, lays[k][1]))
                new_pred_lays[k].append(pred_lays[k][z])
                z += 1
        
        new_pred_lays[k] = np.array(new_pred_lays[k])

    return child_lays, new_pred_lays

def get_dimensions(lst: list) -> list:
    dimensions = []

    while isinstance(lst, list) or isinstance(lst, np.ndarray):  # Mientras el elemento sea una lista
        if isinstance(lst, list): dimensions.append(len(lst))
        elif isinstance(lst, np.ndarray):  # Agregar la longitud de la lista actual a las dimensiones
            lst = lst.tolist()
            dimensions.append(len(lst))

        if len(lst) == 0:
            # Si la lista está vacía, detener el bucle
            break
        lst = lst[0]  # Continuar con el primer elemento para revisar más niveles de anidamiento
    
    return dimensions 

def get_lays_and_preds(cases):
    new_lays = []
    new_pred = []

    for i in cases:
        lay_case = cases[str(i)]

        for lay in lay_case['new_lays']:
            new_lays.append(lay)

        new_pred.append(np.array(lay_case['new_pred']))

    return new_lays, new_pred 

def sort_indices(lst, n):
    # Crea una lista de índices ordenados de forma descendente basándose en los valores de la lista
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse= True)
    
    # Devuelve los n primeros índices
    return sorted_indices[:n]

def init_best_move(lays: list[tuple], child_pred_lays: list[np.array], multiply_pred_lays: list[np.array]):
    lays_pred_size = len(multiply_pred_lays)
    session = dict()

    k = 0
    for i in range(lays_pred_size):
        pred_size = multiply_pred_lays[i].shape[0]

        for j in range(pred_size):
            if len(session) == 0 or not str(lays[k][1]) in session:
                session.update({str(lays[k][1]): {'lays': [], 'multiply_pred': [], 'child_pred': []}})

            session[str(lays[k][1])]['lays'].append(lays[k])
            session[str(lays[k][1])]['multiply_pred'].append(multiply_pred_lays[i][j])
            session[str(lays[k][1])]['child_pred'].append(child_pred_lays[i][j])
            k += 1

    return session

def best_moves_v1(lays: list[tuple], child_pred_lays: list[np.ndarray], multiply_pred_lays: list[np.ndarray], w: int, threshold: int = 0.01) -> tuple:
    session = init_best_move(lays, child_pred_lays, multiply_pred_lays)
    cases = dict()

    for problem in session:
        pred_size = len(session[problem]['multiply_pred'])
        temp_pred = sort_indices(deepcopy(session[problem]['multiply_pred']), w)

        for i in range(pred_size):
            if len(cases) == 0 or not problem in cases:
                cases.update({problem: {'new_lays': [], 'new_pred': []}})
            
            if session[problem]['multiply_pred'][i] >= threshold and i in temp_pred:
                cases[problem]['new_lays'].append(session[problem]['lays'][i])
                cases[problem]['new_pred'].append(session[problem]['child_pred'][i])

    return get_lays_and_preds(cases)

def multiply_predictions(pred_lays: list[np.ndarray], pred_child_lays: list[np.ndarray]) -> list[np.ndarray]:
    pred_lays_size = len(pred_lays)
    multiply_pred = []

    j = 0
    for k in range(pred_lays_size):
        stack_size = pred_lays[k].shape[0]
        
        for i in range(stack_size):
            multiply_pred.append(pred_child_lays[j] * pred_lays[k][i])
            j += 1
    
    return multiply_pred

def count_unsorted_elements(matrix):
    # Cuenta el número de elementos desordenados en cada fila de una matriz
    return sum(any(row[i] < row[i+1] for i in range(len(row)-1)) for row in matrix)

def init_lower_bound_case(lays):
    lays_size = len(lays)
    lb = dict()

    for i in range(lays_size):
        if len(lb) == 0 or not str(lays[i][1]) in lb:
            lb.update({str(lays[i][1]): []})

        lb[str(lays[i][1])].append(lays[i][0])
    
    return lb

def find_lower_bound(lays):
    cases = init_lower_bound_case(lays)
    lb = dict()
    
    for case in cases:
        unsorted_counts = [count_unsorted_elements(matrix.stacks) for matrix in cases[case]]
        lb[case] = min(unsorted_counts)
    
    return lb

def filter_lower_bound(lays: list[tuple], pred_lays: list[np.ndarray], lb: dict, H: int):
    pred_size = len(pred_lays)
    new_lays = []
    new_pred = []

    k = 0
    for i in range(pred_size):
        lay_lb = count_unsorted_elements(lays[k][0].stacks)
        stack_size = pred_lays[i].shape[0]
        flag = True

        for j in range(stack_size):
            if flag:
                new_pred.append([])
                flag = False

            if lb[str(lays[k][1])] + H > lay_lb:
                new_lays.append(lays[k])
                new_pred[i].append(pred_lays[i][j])
            
            k += 1
        
        new_pred[i] = np.array(new_pred[i])

    return new_lays, new_pred

def show_lays(lays):
    lays_size = len(lays)

    for i in range(lays_size):
        print(f'case {lays[i][1]}: {lays[i][0].stacks}')

def verify_states_whitout_solution(solutions: dict, session_lays: list[tuple]):
    session_size = len(session_lays)

    for i in range(session_size):
        if not str(session_lays[i][1]) in solutions:
            solutions.update({str(session_lays[i][1]): -1})
    
    return solutions

def beam_search(model: Model, lays: list[Layout] = None, H: int = 5, 
                w: int = 1000, threshold: float = 0.01, max_steps: int = 30) -> list:
    
    start = time.time()
    session_lays = init_session(lays)
    solutions = dict()
    steps = 0
    
    session_lays, solutions = verify_solution(session_lays, solutions)
    if len(session_lays) == 0: return solutions
    
    session_lays, pred_lays = expanded_lays(model, session_lays)
    session_lays, pred_lays = best_moves_v1(session_lays, pred_lays, pred_lays, w, threshold)

    while True:
        steps += 1 
        session_lays, pred_lays, solutions = verify_solution(session_lays, solutions, pred_lays)
        if len(session_lays) == 0 or steps >= max_steps: break

        child_lays, pred_child_lays = expanded_lays(model, session_lays)
        multiply_pred = multiply_predictions(pred_lays, pred_child_lays)
        session_lays, pred_lays = best_moves_v1(child_lays, pred_child_lays, multiply_pred, w, threshold)

        lb = find_lower_bound(session_lays)
        session_lays, pred_lays = filter_lower_bound(session_lays, pred_lays, lb, H)
        print(get_dimensions(session_lays), get_dimensions(pred_lays))
        print(f'Lower bound: {lb.values()}')

    solutions = verify_states_whitout_solution(solutions, session_lays)
    end = time.time()
    print(f'\nExecution time: {round((end - start) / 60, 3)} minutes')

    return solutions

def show_results(cases: list[Layout], solutions: dict):
    cases_size = len(cases)

    for i in range(cases_size):
        if solutions[str(i)] != -1:
            print(f'Case {i + 1}:')
            print(f'State: {cases[i].stacks}')

            case_solution = solutions[str(i)]
            case_solution_size = len(case_solution)

            for j in range(case_solution_size):
                print(f'    solution {j}, steps {case_solution[j].steps}:')
                print(f'        {case_solution[j].stacks}, ')
                print('')
        else:
            print(f'Case {i + 1}:')
            print(f'State: {cases[i].stacks}')
            print("The problem could\'nt be resolved in the maximum number of steps")
    
        