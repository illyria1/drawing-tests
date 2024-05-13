from enum import Enum
import os
import pandas as pd
import math
import itertools
import params as Params
from typing import Dict, Optional, List
import numpy as np
from params import PRIMITIVES, PARKINSONS, PRIMITIVES_LIST, PrimitiveType
from os import listdir
from os.path import isfile, join
from tensorflow import keras
DEV = False

PRIMITIVE_THRESHOLD = 0.6


RED = 'red'
GREEN = 'green'
BLUE = 'blue'

COLORS = [RED, GREEN, BLUE]

SCALE = 4
RADIUS = 5

FULL_WIDTH = 128
FULL_HEIGHT = 128

WIDTH = int(FULL_WIDTH / SCALE)
HEIGHT = int(FULL_HEIGHT / SCALE)

from params import PRIMITIVES_LIST

class FeatureType(Enum):
    DISTANCE = 1
    DURATION = 2
    MASS = 3
    MEAN = 4
    MEDIAN = 5
    STD = 6
    MAX = 7
    MIN = 8

class Feature:
    def __init__(self, name: str, value: float, unit_of_measurement: str):
        self.name = name
        self.value = round(value,2)
        self.unit_of_measurement = unit_of_measurement

class FeatureFactory:
    param_names = {
        Params.VELOCITY: 'Velocity',
        Params.ACCELERATION: 'Acceleration',
        Params.JERK: 'Jerk',
        Params.PDIFF: 'Pressure difference',
        Params.YANK: 'Yank',
        Params.TUG: 'Tug',
        Params.SNATCH: 'Snatch',
        Params.ALPHA: 'Alpha angle',
        Params.PHI: 'Rotational angle',
        Params.YAW: 'Yaw angle',
        Params.LDIFF: 'Latitude difference',
        Params.ADIFF: 'Azimuth difference'
    }
    units_of_measurement = {
        'duration': 's',
        Params.DISTANCE: 'mm',
        Params.VELOCITY : 'mm/s',
        Params.ACCELERATION : 'mm/{}\u00b2'.format('s'),
        Params.JERK : 'mm/{}\u00b3'.format('s'),
        Params.PDIFF : 'units',
        Params.YANK : 'units/s',
        Params.TUG : 'units/{}\u00b2'.format('s'),
        Params.SNATCH : 'units/s{}\u00b4'.format(4),
        Params.ALPHA: 'rad',
        Params.PHI: 'rad',
        Params.YAW: 'rad',
        Params.LDIFF: 'rad',
        Params.ADIFF: 'rad'
    }
    def build(self, df: pd.DataFrame, feature_type: FeatureType, param:str=None):
        if feature_type == FeatureType.DISTANCE:
            return Feature(
                'Distance',
                df['dis'].sum(),
                FeatureFactory.units_of_measurement['dis']
            )
        elif feature_type == FeatureType.DURATION:
            return Feature(
                'Duration',
                df.iloc[-1]['t'] - df.iloc[0]['t'],
                FeatureFactory.units_of_measurement['duration']
            )
        elif feature_type == FeatureType.MASS:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'mass',
                df[param].sum(),
                FeatureFactory.units_of_measurement[param]
            )
        elif feature_type == FeatureType.MEAN:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'mean',
                df[param].mean(),
                FeatureFactory.units_of_measurement[param]
            )
        elif feature_type == FeatureType.MEDIAN:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'median',
                df[param].median(),
                FeatureFactory.units_of_measurement[param]
            )
        elif feature_type == FeatureType.STD:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'std',
                df[param].std(),
                FeatureFactory.units_of_measurement[param]
            )
        elif feature_type == FeatureType.MAX:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'max',
                df[param].max(),
                FeatureFactory.units_of_measurement[param]
            )
        elif feature_type == FeatureType.MIN:
            return Feature(
                FeatureFactory.param_names[param] + ' ' + 'min',
                df[param].min(),
                FeatureFactory.units_of_measurement[param]
            )


class IndexedDF:
    def __init__(self, df : pd.DataFrame, begin : int = None, end : int = None):
        self._df = df
        self._indexed_df = None
        if begin is None or end is None:
            self.begin = 0
            self.end = len(self._df.index) - 1
        else:
            self.begin = begin
            self.end = end
    @property
    def df(self):
        if self.begin == 0 and self.end == len(self._df.index) - 1:
            return self._df
        else:
            if self._indexed_df is None:
                self._indexed_df = self._df.iloc[self.begin:self.end+1]
                return self._indexed_df
            else:
                return self._indexed_df

class Stroke:
    id_segment = itertools.count()
    def __init__(self, main_indexed_df: IndexedDF, color : str ='BLACK', merged=False, original=False, **kwargs):
        self.main_indexed_df : IndexedDF = main_indexed_df
        self._filtered_df : pd.DataFrame = None
        self.color : str = color
        self.features = None
        self.original = original
        self.forced_type = None
        self.type = None
        type_ = kwargs.get('type', None)
        if type_ is not None:
            for primitive_type in PRIMITIVES_LIST:
                if type_ == primitive_type.name:
                    self.type = primitive_type
        if self.type is None:
            self.type = PRIMITIVES_LIST[0]
        self.note = None
        self.prediction = None
        self.used_model = None
        self.id = next(self.id_segment)
        self.patient = kwargs.get('patient', None)
        self.time = kwargs.get('time', None)
        self.hand = kwargs.get('hand', None)
        if self.hand is not None:
            if self.hand == 'M':
                self.hand = 'Right'
            else:
                self.hand = 'Left'
        if merged:
            self.name : str = "Merged segment #" + str(self.id)
        else:
            self.name : str = "Segment #" + str(self.id)
    @property
    def filtered_df(self) -> pd.DataFrame:
        if self._filtered_df is not None:
            return self._filtered_df
        else:
            return self.main_indexed_df.df
    @property
    def main_df(self):
        return self.main_indexed_df.df
    def assign_stroke(self, other):
        self.main_indexed_df = other.main_indexed_df
        self._filtered_df = other._filtered_df
    def update_extra(self, other):
        if other.patient is not None:
            self.patient = other.patient
        if other.time is not None:
            self.time = other.time
        if other.hand is not None:
            self.hand = other.hand
    def restore(self, original_df: pd.DataFrame):
        self.main_indexed_df = IndexedDF(original_df)
        self._filtered_df = None
    def get_features_as_plain_dict(self, for_training=True):
        features = self.calculate_single_value_features()
        features_dict = {}
        for category in features:
            for column in features[category]:
                for feature in column:
                    if not for_training or (feature.name != "Duration" and feature.name != "Distance"):
                        features_dict[feature.name] = feature.value
                        #print(len(features_dict))
        return features_dict
    def get_features_as_table(self):
        features = self.calculate_single_value_features()
        features_table : List[List[str]]= []
        height = 0
        width = 0
        for category in features:
            height += len(features[category][0])
            width = max(len(features[category]), width)
        for i in range(height):
            features_table.append([])
            for j in range(width):
                features_table[i].append('')
        row_index = 0
        column_index_prev = 0
        column_index = 0
        for category in features:
            row_index = 0
            for column in features[category]:
                column_index = column_index_prev
                for feature in column:
                    features_table[column_index][row_index] = "{}: {:.2e} {}".format(
                                                feature.name,
                                                feature.value,
                                                feature.unit_of_measurement
                                            )
                    column_index += 1
                row_index += 1
            column_index_prev = column_index
        # for i in range(len(features_table)):
        #     if len(features_table[i]) < max_row:
        #         features_table[i] = features_table[i] + ([''] * (max_row - len(features_table[i])))
        return features_table
    def calculate_single_value_features(self):
        # duration = self.filtered_df.iloc[0]['t'] - self.filtered_df.iloc[-1]['t']
        # distance = self.filtered_df['dis'].sum()

        # velocity_mean = self.filtered_df.loc[:, 'velocity'].mean()
        # velocity_median = self.filtered_df['velocity'].median()
        # velocity_mass = self.filtered_df['velocity'].sum()
        # velocity_std = self.filtered_df['velocity'].std()
        # velocity_max = self.filtered_df['velocity'].max()
        # velocity_min = self.filtered_df['velocity'].min()

        # acceleration_mean = self.filtered_df.loc[:, 'acceleration'].mean()
        # acceleration_median = self.filtered_df['acceleration'].median()
        # acceleration_mass = self.filtered_df['acceleration'].sum()
        # acceleration_std = self.filtered_df['acceleration'].std()
        # acceleration_max = self.filtered_df['acceleration'].max()
        # acceleration_min = self.filtered_df['acceleration'].min()

        # jerk_mean = self.filtered_df.loc[:, 'jerk'].mean()
        # jerk_median = self.filtered_df['jerk'].median()
        # jerk_mass = self.filtered_df['jerk'].sum()
        # jerk_std = self.filtered_df['jerk'].std()
        # jerk_max = self.filtered_df['jerk'].max()
        # jerk_min = self.filtered_df['jerk'].min()

        # pressure_diff_mean = self.filtered_df.loc[:, 'pdif'].mean()
        # pressure_diff_median = self.filtered_df.loc[:, 'pdif'].median()
        # pressure_diff_mass = self.filtered_df.loc[:, 'pdif'].sum()
        # pressure_diff_std = self.filtered_df.loc[:, 'pdif'].std()
        # pressure_diff_max = self.filtered_df.loc[:, 'pdif'].max()
        # pressure_diff_min = self.filtered_df.loc[:, 'pdif'].min()

        # yank_mean = self.filtered_df.loc[:, 'yank'].mean()
        # yank_median = self.filtered_df.loc[:, 'yank'].median()
        # yank_mass = self.filtered_df.loc[:, 'yank'].sum()
        # yank_std = self.filtered_df.loc[:, 'yank'].std()
        # yank_max = self.filtered_df.loc[:, 'yank'].max()
        # yank_min = self.filtered_df.loc[:, 'yank'].min()

        # tug_mean = self.filtered_df.loc[:, 'tug'].mean()
        # tug_median = self.filtered_df.loc[:, 'tug'].median()
        # tug_mass = self.filtered_df.loc[:, 'tug'].sum()
        # tug_std = self.filtered_df.loc[:, 'tug'].std()
        # tug_max = self.filtered_df.loc[:, 'tug'].max()
        # tug_min = self.filtered_df.loc[:, 'tug'].min()

        # snatch_mean = self.filtered_df.loc[:, 'snatch'].mean()
        # snatch_median = self.filtered_df.loc[:, 'snatch'].median()
        # snatch_mass = self.filtered_df.loc[:, 'snatch'].sum()
        # snatch_std = self.filtered_df.loc[:, 'snatch'].std()
        # snatch_max = self.filtered_df.loc[:, 'snatch'].max()
        # snatch_min = self.filtered_df.loc[:, 'snatch'].min()
        factory = FeatureFactory()
        return {
            'Spatial-time features': [[factory.build(self.filtered_df, FeatureType.DISTANCE)],
                                      [factory.build(self.filtered_df, FeatureType.DURATION)]],
            'Kinematic features': [[factory.build(self.filtered_df, FeatureType.MASS, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.JERK),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.JERK)],
                                   [
                                    factory.build(self.filtered_df, FeatureType.STD, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.JERK),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.JERK),
                                   ],
                                   [
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.VELOCITY),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.ACCELERATION),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.JERK),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.JERK),
                                   ]],
            'Pressure features': [[factory.build(self.filtered_df, FeatureType.MASS, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.TUG),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.TUG)],
                                   [
                                    factory.build(self.filtered_df, FeatureType.STD, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.TUG),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.TUG),
                                   ],
                                   [
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.PDIFF),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.YANK),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.TUG),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.TUG),
                                   ]],
            'Geometric features': [[factory.build(self.filtered_df, FeatureType.MASS, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.MASS, Params.YAW),
                                    factory.build(self.filtered_df, FeatureType.MEAN, Params.YAW)],
                                   [factory.build(self.filtered_df, FeatureType.STD, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.STD, Params.YAW),
                                    factory.build(self.filtered_df, FeatureType.MEDIAN, Params.YAW)],
                                   [factory.build(self.filtered_df, FeatureType.MIN, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.ADIFF),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.LDIFF),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.ALPHA),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.PHI),
                                    factory.build(self.filtered_df, FeatureType.MIN, Params.YAW),
                                    factory.build(self.filtered_df, FeatureType.MAX, Params.YAW)]]
        }


class Primitive(Stroke):
    def __init__(self, df, begin, end, type, predictions, color='BLACK'):
        super().__init__(df, begin, end, color)
        self.type = type
        self.predictions = predictions

class Square:
    def __init__(self, x, y, width, height):
        self.startx = x
        self.starty = y
        x2 = x + width
        y2 = y + height
        self.points = [[self.startx, self.starty], [x2, self.starty], [x2, y2], [self.startx, y2], [0, 0]]
    @property
    def x1(self):
        return self.points[0][0]
    @property
    def y1(self):
        return self.points[0][1]
    @property
    def x2(self):
        return self.points[2][0]
    @property
    def y2(self):
        return self.points[2][1]
    def set_bottom_right(self, x, y):
        if x > self.startx:
            self.points[1][0] = x
            self.points[2][0] = x
        elif x < self.startx:
            self.points[0][0] = x
            self.points[3][0] = x
        if y > self.starty:
            self.points[2][1] = y
            self.points[3][1] = y
        elif y < self.starty:
            self.points[0][1] = y
            self.points[1][1] = y

            
class ModelDefinition:
    def __init__(self, primitives_model_name : str = None, default=False):
        self.default = default
        if default:
            self.primitives_model_name = "general_model"
            self.model_id = 0
            self.primitives = [PRIMITIVES_LIST[0]]
            self.parkinsons = {}
            self.parkinsons[0] = f"general_model"
        else:
            self.primitives_model_name = primitives_model_name
            model_id = int(primitives_model_name.split('-')[-1])
            primitive_ids = list(map(lambda x: int(x), primitives_model_name.split('-')[1:-1]))
            primitives : List[PrimitiveType] = list(filter(lambda x : x.id in primitive_ids, PRIMITIVES_LIST))
            self.model_id : int = model_id
            self.primitives = primitives
            self.parkinsons = {}
            for primitive in self.primitives:
                self.parkinsons[primitive.id] = f"parkinsons_model-{primitive.id}-{model_id}"

class ModelsObject:
    def __init__(self, model_definition : ModelDefinition = None):
        self.model_definition = model_definition
        if model_definition.default:
            self.primitives_model = None
            self.parkinsons : Dict[int, keras.models.Model] = {}
            self.parkinsons[0] = keras.models.load_model(f'data/models/{PARKINSONS}/general_model')
        else:
            self.primitives_model : keras.models.Model = keras.models.load_model(f'data/models/{PRIMITIVES}/{model_definition.primitives_model_name}')
            self.parkinsons : Dict[int, keras.models.Model] = {}
            for primitive_id in model_definition.parkinsons:
                if primitive_id == 0:
                    if not os.path.isdir(f'data/models/{PARKINSONS}/{model_definition.parkinsons[primitive_id]}'):
                        continue
                self.parkinsons[primitive_id] = keras.models.load_model(f'data/models/{PARKINSONS}/{model_definition.parkinsons[primitive_id]}')
    def get_type_prediction(self, stroke: Stroke):
        if self.primitives_model is not None:
            converted_stroke = np.asarray([convert_stroke(stroke)])
            predictions = self.primitives_model.predict(converted_stroke)
            predictions_as_dict = {}
            for index, prediction in enumerate(predictions[0]):
                self.model_definition.primitives.sort(key=lambda x: x.id)
                predictions_as_dict[self.model_definition.primitives[index]] = prediction
            #print(predictions_as_dict)
            return predictions_as_dict
        else:
            raise RuntimeError("This is a general model which cant predict type!")
    def get_parkinsons_prediction(self, primitive_id: int, stroke: Stroke):
        if primitive_id not in self.parkinsons:
            raise RuntimeError(f"No corresponding model for the primitive {primitive_id} found!")
        else:
            model = self.parkinsons[primitive_id]
            features = np.asarray([list(stroke.get_features_as_plain_dict().values())])
            #print(features)
            prediction = model.predict(features)
            return prediction[0][0]

class ModelsObjectHolder:
    model_object : ModelsObject = None

def is_point_within_square(square : Square, x, y):
    if x <= square.x2 and x >= square.x1 and y <= square.y2 and y >= square.y1:
        return True
    return False

def distance(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def filter_by_pressure(data_frame):
    #data_frame.sort_values(by=['p'])
    return data_frame.query('p > 0.3')


def calculate_base_params(data_frame: pd.DataFrame):
    displacements = []
    pressure_diffs = []
    alphas = []
    phis = []
    yaws = []
    a_diffs = []
    l_diffs = []
    for index in range(0, len(data_frame)):
        if index == len(data_frame.index) - 1:
            row1 = data_frame.iloc[index - 1]
            row2 = data_frame.iloc[index]
        else:
            row1 = data_frame.iloc[index]
            row2 = data_frame.iloc[index + 1]
        dis = distance((row2['x'], row2['y']), (row1['x'], row1['y']))
        pressure_diff = row2['p'] - row1['p']
        latitude_diff = row2['l'] - row1['l']
        l_diffs.append(latitude_diff)
        azimuth_diff = row2['a'] - row1['a']
        a_diffs.append(azimuth_diff)
        k = (row2['y'] - row1['y']) / (row2['x'] - row1['x'])
        alpha = math.atan(k)
        if index == 1:
            phi = math.pi + alphas[index-1] - alpha
            yaw = alpha - alphas[index-1]
            phis.append(phi)
            phis.append(phi)
            yaws.append(yaw)
            yaws.append(yaw)
        elif index > 1:
            phi = math.pi + alphas[index-1] - alpha
            yaw = alpha - alphas[index-1]
            phis.append(phi)
            yaws.append(yaw)
        displacements.append(dis)
        pressure_diffs.append(pressure_diff)
        alphas.append(alpha)

    data_frame[Params.DISTANCE] = displacements
    data_frame[Params.PDIFF] = pressure_diffs
    data_frame[Params.ADIFF] = a_diffs
    data_frame[Params.LDIFF] = l_diffs
    data_frame[Params.ALPHA] = alphas
    data_frame[Params.PHI] = phis
    data_frame[Params.YAW] = yaws
    
def calculate_derivative(data_frame : pd.DataFrame, next_derivative_name: str, arg: str):
    derivative_vals = []
    for index in range(0, len(data_frame)):
        if index == len(data_frame.index) - 1:
            row1 = data_frame.iloc[index - 1]
            row2 = data_frame.iloc[index]
        else:
            row1 = data_frame.iloc[index]
            row2 = data_frame.iloc[index + 1]
        if arg == Params.DISTANCE:
            val = row1[arg]
        else:
            val = row2[arg] - row1[arg]
        time_delta = abs(row2['t'] - row1['t'])
        if time_delta < 0.001:
            time_delta = 0.001
        derivative_vals.append(val / time_delta)
        #if index == 157:
            #print(val / time_delta)
            #print(time_delta)
    data_frame[next_derivative_name] = derivative_vals

def calculate_vector_features(data_frame: pd.DataFrame):
    data_frame.sort_values(by=[Params.TIME], inplace=True, ascending=True)
    calculate_base_params(data_frame)
    calculate_derivative(data_frame, Params.VELOCITY, Params.DISTANCE)
    calculate_derivative(data_frame, Params.ACCELERATION, Params.VELOCITY)
    calculate_derivative(data_frame, Params.JERK, Params.ACCELERATION)
    calculate_derivative(data_frame, Params.YANK, Params.PDIFF)
    calculate_derivative(data_frame, Params.TUG, Params.YANK)
    calculate_derivative(data_frame, Params.SNATCH, Params.TUG)



def get_acceleration_vectors(data_frame):
    if 'v' in data_frame.columns:
        data_frame = data_frame.sort_values(by=['t'])
        acceleration_vectors = []
        for index, row in data_frame.iterrows():
            if index < data_frame.size - 2:
                row1 = row
                row2 = data_frame.iloc[index + 1]
                row3 = data_frame.iloc[index + 2]

                vector = ((((row3['x'] - row2['x']) / (row3['t'] - row2['t'])) - ((row2['x'] - row1['x']) / (row2['t'] - row1['t']))),
                (((row3['y'] - row2['y']) / (row3['t'] - row2['t'])) - ((row2['y'] - row1['y']) / (row2['t'] - row1['t']))))
                if (math.isnan(vector[0]) or math.isnan(vector[1]) or math.isinf(vector[0]) or math.isinf(vector[1])):
                    if index == 0:
                        acceleration_vectors.append((0, 0))
                    else:
                        acceleration_vectors.append(acceleration_vectors[index - 1])
                else:
                    acceleration_vectors.append(vector)

        return acceleration_vectors
    else:
        #print("Data frame has no velocity")
        return []
    
def convert_stroke(stroke: Stroke):
        res = np.zeros((WIDTH, HEIGHT))
        half_width = int(WIDTH / 2)
        half_height = int(HEIGHT / 2)
        df = stroke.filtered_df
        xmin = df['x'].min()
        xmax = df['x'].max()
        ymin = df['y'].min()
        ymax = df['y'].max()

        xmid = df['x'].mean()
        ymid = df['y'].mean()

        #diff_x = point_x_mid - square_x_mid
        diff_x = xmid #xmin
        diff_y = ymid #ymin
        #print("x:" + str(xmax - xmin))
        #print("y:" + str(ymax - ymin))
        
        for index in range(0, len(df)):
            x_orig = df.iloc[index]['x']
            y_orig = df.iloc[index]['y']
            
            x = int(x_orig - diff_x) // SCALE + half_width
            y = int(y_orig - diff_y) // SCALE + half_height
            angles = np.arange(0, 2*np.pi, 0.1)
            radii = np.arange(SCALE, RADIUS, 0.2)
            for angle in angles:
                for dr in radii:
                    x_circle = int(((x_orig - diff_x) + dr * np.cos(angle))) // SCALE + half_width
                    y_circle = int(((y_orig - diff_y) + dr * np.sin(angle))) // SCALE + half_height
                    if x_circle >= 0 and x_circle < WIDTH and y_circle >= 0 and y_circle < HEIGHT:
                        res[y_circle][x_circle] = 1
            if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT:
                res[y][x] = 1
            #else:
                #print("out of bounds")
        return res

def enumerate_ids(ids):
    return list(range(0, len(ids)))

def get_last_index_in_folder(dir_path, extension):
    files = [f.rstrip(extension) for f in listdir(dir_path)]
    #print(files)
    index = 0
    try:
        if files:
            index = max(map(lambda x: int(x.split('-')[-1]), files))
    except Exception as e:
        #print("Could not find last index!!!")
        #print(str(e))
        return
    return index

def save_stroke(converted_stroke, prefix: str, txt=False):
    dir_path = f'data/train/{PRIMITIVES}'
    index = get_last_index_in_folder(dir_path, '.npy')
    filename = f'{prefix}-{index + 1}'
    if txt:
        with open('strokes_txt/' + filename + '.txt', 'w') as file:
            for line in converted_stroke.tolist():
                file.write(str(line) + '\n')
    np.save(f'data/train/{PRIMITIVES}/{filename}', converted_stroke)

def save_features(features: dict, prefix: str, txt=False):
    dir_path = f'data/train/{PARKINSONS}'
    index = get_last_index_in_folder(dir_path, '.npy')
    filename = f'{prefix}-{index + 1}'
    if txt:
        with open('parkinsons_txt/' + filename + '.txt', 'w') as file:
            for key in features:
                file.write(str(key) + ' : ' + str(features[key]) + '\n')
    np.save(f'data/train/{PARKINSONS}/{filename}', features)

def get_model_primitives(index):
    dir_path = f'data/models/{PRIMITIVES}'
    extension = ''
    files = [f.rstrip(extension) for f in listdir(dir_path)]
    try:
        for filename in files:
            file_index = int(filename.split('-')[-1])
            if index == file_index:
                #print(filename.split('-')[1:-1])
                return list(map(lambda x: int(x), filename.split('-')[1:-1]))
        raise ValueError()
    except ValueError:
        raise RuntimeError(f"Could not find model with index {index}")
    
def load_model_definitions():
    model_definitions = []
    dir_path = f'data/models/{PRIMITIVES}'
    extension = ''
    files = [f.rstrip(extension) for f in listdir(dir_path)]
    for filename in files:
        model_definitions.append(ModelDefinition(filename))
    return model_definitions
