from typing import List
import pandas as pd
import json
import numpy as np
import os
from tensorflow import keras
import re

from params import PRIMITIVES, PARKINSONS, PRIMITIVES_LIST, PrimitiveType

data = []
#C:/thesis/data/'
#'20191012-141556-21001-multicube-MS.json'

LABELS_N = 4

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    test_data_frame = pd.DataFrame()

    if not data:
        print("Data was not loaded!!!")
        return test_data_frame

    keys = data.keys()
    drawing = data['data']
    extra = {
        'patient' : data['patientId'],
        'time' : data['time'],
        'hand' :data['hand']
    }
    
    #type = data['type']
    #i = 0
    for stroke_data in drawing:
        stroke = pd.DataFrame(stroke_data)
        #stroke['stroke_index'] = i
        test_data_frame = pd.concat([test_data_frame, stroke])
        #i += 1
    
    return test_data_frame, extra

def load_train_data_primitives(primitives_list: List[PrimitiveType]):
    X = []
    Y = []
    for filename in os.listdir(f'data/train/{PRIMITIVES}'):
        if re.match(r"\d+-\d+.npy", filename):
            ids = [x.id for x in primitives_list]
            ids.sort()
            f = os.path.join(f'data/train/{PRIMITIVES}', filename)
            if os.path.isfile(f):
                split = filename.rstrip('.npy').split('-')
                primitive_type = int(split[0])
                if primitive_type in ids:
                    #print(filename)
                    arr = np.load(f)
                    labels = np.zeros((len(primitives_list)))
                    labels[ids.index(primitive_type)] = 1 
                    X.append(arr)
                    Y.append(labels)
    return (np.asarray(X), np.asarray(Y))

def load_train_data_parkinsons(primitive: PrimitiveType, load_all=False):
    X = []
    Y = []
    for filename in os.listdir(f'data/train/{PARKINSONS}'):
        f = os.path.join(f'data/train/{PARKINSONS}', filename)
        if os.path.isfile(f):
            split = filename.rstrip('.npy').split('-')
            parkinsons = int(split[0])
            primitive_type = int(split[1])
            #print(filename)
            #print(parkinsons)
            #print(primitive_type)
            #print(primitive.id)
            if load_all or primitive_type == primitive.id:
                temp = np.load(f, allow_pickle=True)
                arr = list(temp.item().values())
                #print(arr)
                label = [parkinsons]
                #label[parkinsons] = 1
                X.append(arr)
                Y.append(label)
    return (np.asarray(X), np.asarray(Y))

def load_model(name):
    model = keras.models.load_model('data/models/' + name)
    return model
    