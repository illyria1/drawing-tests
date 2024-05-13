import sys
from typing import List
import tensorflow as tf
import keras
from keras import layers
import argparse
import importlib

from load import *
from core import WIDTH, HEIGHT, get_last_index_in_folder, get_model_primitives
from params import PRIMITIVES_LIST, PrimitiveType

def save_model(model, name):
    model.save(name)

#history, model = test_model()

#save_model(model, 'test')

# model = load_model('test')
# X, Y = load_train_data()
# x_test = X[:3]
# predictions = model.predict(x_test)
# print(predictions)

def main(train_model_features, train_model_primitives,
         train_parkinsons: bool, parkinsons_id,  primitives_list: list[PrimitiveType], 
         general=False):
    #print(train_parkinsons)
    #print(primitives_list)
    if general:
        parkinsons_x, parkinsons_y = load_train_data_parkinsons(None, True)
        #print(len(parkinsons_y))
        if len(parkinsons_x) > 0:
            model_name = f"general_model"
            #print(model_name)
            history, model = train_model_features(parkinsons_x, parkinsons_y, model_name)
            save_model(model, f"data/models/{PARKINSONS}/{model_name}")
    elif train_parkinsons and parkinsons_id is not None:
        primitives = list(filter(lambda x : x.id in get_model_primitives(parkinsons_id), PRIMITIVES_LIST))
        #print(primitives)
        if primitives:
            for primitive in primitives:
                if primitive.id == 0:
                    parkinsons_x, parkinsons_y = load_train_data_parkinsons(None, True)
                else:
                    parkinsons_x, parkinsons_y = load_train_data_parkinsons(primitive)
                if len(parkinsons_x) > 0:
                    model_name = f"parkinsons_model-{primitive.id}-{parkinsons_id}"
                    #print(model_name)
                    history, model = train_model_features(parkinsons_x, parkinsons_y, model_name)
                    save_model(model, f"data/models/{PARKINSONS}/{model_name}")
    elif primitives_list:
        primitives_data_x, primitives_data_y  = load_train_data_primitives(primitives_list)
        if len(primitives_data_x) > 0:
            get_model_last_index = get_last_index_in_folder(f'data/models/{PRIMITIVES}', '')
            get_model_last_index = get_model_last_index + 1 
            primitives_indicies = ''.join(list(map(lambda x: str(x.id) + '-', primitives_list)))
            model_name = f"primitives_model-{primitives_indicies}{get_model_last_index}"
            #print(model_name)
            history, model = train_model_primitives(primitives_data_x, primitives_data_y, WIDTH, HEIGHT, model_name, primitives_list, len(primitives_list))
            save_model(model, f'data/models/{PRIMITIVES}/{model_name}')
        #else:
            #print("No data samples found!")

parser = argparse.ArgumentParser(description='Train neural network.', add_help=True)
parser.add_argument('--primitives', dest='primitive_str', help="Specify primitives (comma separated names) for new model (primitives.conf)")
parser.add_argument('--parkinsons', type=int, dest='model_id', help="Specify id of a primitives model (data/models/primitives last id in filename)")
parser.add_argument('--general', action=argparse.BooleanOptionalAction, help="Train a model on all samples irrespective of primitive type")
parser.add_argument('--custom', help="Specify custom model definition (data/models_src/)")
parser.add_argument('--evaluate', help="Evaluate model instead of training")

args = parser.parse_args(sys.argv[1:])

try:
    if args.custom is not None:
        module = importlib.import_module(f'data.models_src.{args.custom}')
    else:
        module = importlib.import_module('data.models_src.default')
except Exception as e:
    #print(str(e))
    #print("Could not import the module")
    exit()

if args.general is True:
    main(module.train_model_features, module.train_model_primitives, True, None, [PRIMITIVES_LIST[0]], True)
if args.model_id is not None:
    main(module.train_model_features, module.train_model_primitives, True, args.model_id, None)
elif args.primitive_str:
    primitives = args.primitive_str.split(',')
    #print(module.train_model_features, module.train_model_primitives, primitives)
    main(module.train_model_features, module.train_model_primitives, False, None, list(filter(lambda x : x.name in primitives, PRIMITIVES_LIST)))


