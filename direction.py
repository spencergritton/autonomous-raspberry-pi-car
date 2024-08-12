from enum import Enum, auto

class Direction(Enum):
    '''
	Represents a direction on a 2D map.
	'''
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()