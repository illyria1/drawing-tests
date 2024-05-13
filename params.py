from typing import List

MODE_2D = 0
MODE_3D = 1

VELOCITY = 'velocity'
TIME = 't'
ACCELERATION = 'acceleration'
JERK = 'jerk'
DISTANCE = 'dis'
PDIFF = 'pdiff'
YANK = 'yank'
TUG = 'tug'
SNATCH = 'snatch'
ALPHA = 'alpha'
PHI = 'phi'
YAW = 'yaw'
AZIMUTH = 'a'
LATITUDE = 'l'
ADIFF = 'adiff'
LDIFF = 'ldiff'

PARK_NEGATIVE = 0
PARK_POSITIVE = 1

PRIMITIVES = 'primitives'
PARKINSONS = 'parkinsons'

class PrimitiveType:
    def __init__(self, id, name):
        self.id = id
        self.name = name
    def __hash__(self):
        return hash((self.id, self.name))

PRIMITIVES_LIST : List[PrimitiveType] = []

with open("primitives.conf", 'r') as file:
    PRIMITIVES_LIST.clear()
    lines = file.readlines()
    for line in lines:
        params = line.split(',')
        if int(params[0]) < 0:
            raise RuntimeError("Primitive id must be greater than 0!!!")
        PRIMITIVES_LIST.append(PrimitiveType(int(params[0]), str(params[1]).rstrip('\n').lstrip(' ')))

