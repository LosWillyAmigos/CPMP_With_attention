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