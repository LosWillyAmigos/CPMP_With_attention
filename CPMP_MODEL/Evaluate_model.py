import numpy as np
import importlib
from copy import deepcopy

import os
current_directory = os.getcwd()
os.chdir(current_directory+"/CPMP-ML")

import cpmp_ml
from cpmp_ml import generate_random_layout, greedy, get_ann_state, generate_data
os.chdir(current_directory)

def get_ann_state(layout: cpmp_ml.Layout) -> np.ndarray:
  S=len(layout.stacks) # Cantidad de stacks
  #matriz de stacks
  b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
  for i,j in enumerate(layout.stacks):
     b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
     b[i][0] = layout.is_sorted_stack(i)
  b.shape=(S,(layout.H + 1))
  return b

#overriding the function in the module
cpmp_ml.get_ann_state = get_ann_state

def cosine_Similarity(y_predict, y_test):
    """
    The puspuse of this function is to verify if the values
    predicted by a multiclass classification deep learning
    mechanism are correct or not.

    Input:

        y_predict (list): Values predicted by the machine
                          learning.
        y_test (list): Actual values for each case.

    Return:
        float: Proportion of correctly predicted values over
        the total number of cases.
    """
    size = len(y_predict)
    suma = 0

    for i in range(size):
        result = np.dot(y_predict[i], y_test[i]) / (np.linalg.norm(y_predict[i]) * np.linalg.norm(y_test[i]))
        suma += result

    return suma / size



def best_move(model, state, S=5):
    predict = model.predict(np.stack([state]),verbose=0)
    idx = np.argmax(predict)
    aux=0
    for i in range(S):
        for k in range(S):
            if i != k:
                if idx == aux:
                    return (i,k)
                aux+=1
    return None

def greedy_dinamic_model(layout, model, max_steps=20):
    steps = 0
    while layout.unsorted_stacks>0 and steps < max_steps:
        bg_move=best_move(model, get_ann_state(layout))
        if bg_move is not None:
            layout.move(bg_move)
        else:
            return -1 # no lo resuelve
        steps +=1

    if layout.unsorted_stacks==0:
        return steps
    return -1

def generate_random_state(sample_size,S,H,N):
  l = []
  for i in range(sample_size):
    l.append(generate_random_layout(S=S,H=H,N=N))

  return l, deepcopy(l)

def evaluate(sample_size=10000, model = None, Verbose=True, Greedy=True, S=5, H=5, N=15):
    
    if Greedy:
       win_greedy = 0
       steps_greedy = 0
    
    win_model = 0
    steps_model = 0
    
    ls, la = generate_random_state(sample_size,S,H,N)

    for i in range(sample_size):
        if Verbose: print(f'iter: {i+1}')
        if Greedy:
            sg = greedy(ls[i])
            if sg != -1:
                steps_greedy+=sg
                win_greedy+=1
        sm = greedy_dinamic_model(la[i], model)
        if sm != -1:
          steps_model+=sm
          win_model+=1

    if Greedy:
       print(f'Casos resueltos por greedy : {(win_greedy/sample_size)*100}% - {steps_greedy/sample_size}')
    print(f'Casos resueltos por model: {(win_model/sample_size)*100}% - {steps_model/sample_size}')

def validate_model(model, S, H, N, verbose: bool = True, cvs_class=None):
    n = 1000

    lays = []
    if cvs_class is None:
        for i in range(n):
            lays.append(generate_random_layout(S,H,N))
    else:
        n=40
        for i in range(1,n+1):
            lay = read_file(f"benchmarks/CVS/{cvs_class}/data{cvs_class}-{i}.dat",5)
            lays.append(lay)

    lays1 = deepcopy(lays)
    costs1 = greedy_model(model, lays1, S= S, H= H, max_steps=N*2)
    costs2 = greedys(lays, max_steps= 80)

    valid_costs1 = [v for v in costs1 if v!=-1]
    valid_costs2 = [v for v in costs2 if v!=-1]

    results_model = len(valid_costs1) / n * 100.
    results_greedy = len(valid_costs2) / n * 100.

    if len(valid_costs1)>0:
        print(f"success ann model (%): {results_model}")
        print(f"mean steps: {mean(valid_costs1)}")
        print(f"median steps: {median(valid_costs1)}")
        #print(f"stdesv steps: {stdev(valid_costs1)}")
        print(f"min steps: {min(valid_costs1)}")
        print(f"max steps: {max(valid_costs1)}")
        print('')
    if len(valid_costs2)==0:
        print("success heuristic (%):", results_greedy)
    else:
        print("success heuristic (%):", results_greedy, mean(valid_costs2))
        print(f"mean steps: {mean(valid_costs2)}")
        print(f"median steps: {median(valid_costs2)}")
        #print(f"stdesv steps: {stdev(valid_costs2)}")
        print(f"min steps: {min(valid_costs2)}")
        print(f"max steps: {max(valid_costs2)}")
        print('')

    return results_model, results_greedy
