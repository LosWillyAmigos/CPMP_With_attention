import json
import numpy as np

def save_data(states:list, labels:list, source:str, name_file:str) -> bool:

    matrix_list = [matrix.tolist() for matrix in states]
    vector_list = [vector.tolist() for vector in labels]

    data = {}
    for i, m_l in enumerate(matrix_list, 1):
        key = f"matrix_{i}"
        data[key] = m_l

    for i, v_l in enumerate(vector_list, 1):
        key = f"vector_{i}"
        data[key] = v_l
    
    
    with open(source+"/"+name_file, 'w') as file:
        json.dump(data, file)

    return True

def load_data_from_json(name_file:str):
    with open(name_file, 'r') as archivo_json:
        data = json.load(archivo_json)

    states = []
    labels = []

    for input in data:
        states.append(np.array(input.get("State", [])))
        labels.append(np.array(input.get("Labels", [])))


    return np.stack(states), np.stack(labels)


def generate_a_lot_of_data(model,
                      source:str,
                      name_file:str,
                      n_files:int,
                      S=5,
                      H=5,
                      N=15,
                      sample_size=1000,
                      max_steps=20,
                      batch_size=55,
                      perms_by_layout=1):
    
    for i in range(n_files):
        states = []
        labels = []
        states, labels = generate_data2(model=model,S=S,H=H,N=N,sample_size=sample_size,
                        max_steps=max_steps,batch_size=batch_size,
                        perms_by_layout=perms_by_layout)
        
        name = name_file + "_" + str(i) +"_" + str(S) + "x" + str(H) + ".json"

        save_data(states=states,labels=labels,source=source,name_file=name)


def get_a_lot_of_data(S:int,H:int,n_files:int,source:str,name_file=str):
    data_states = []
    data_labels = []
    for i in range(n_files):
        n = name_file + "_" + str(i) +"_" + str(S) + "x" + str(H) + ".json"
        s, l = load_data(source=source, name_file=n)
        data_states.append(s)
        data_labels.append(l)

    return data_labels,data_states