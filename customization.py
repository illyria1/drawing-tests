from typing import List

from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from core import Square


class FilterSettings():
    def __init__(self, filter_value, parameter):
        self.filter_value = filter_value
        self.parameter = parameter

class Customization():
    def __init__(self, size):
        self.color = None
        self.size = size
        self.sortParameter = None
        self.colorParameter = None
        self.lims = None
        self.addColorbar = False
        self.deleteColorbar = False
        self.colorbar = None
        self.ax : Axes = None
        self.canvas : FigureCanvasBase = None
        self.toggleAxis = True
        self.invertY = True
        self.show_filtered_df : bool = False
        self.filter_settings : FilterSettings = None
        self.drawSelectionSquare : bool = False
        self.selectionSquareToDraw : Square = None
        self.selectionSquareSize = 50
        self.selectionSquareLeftClicked = False
        self.show_all_selection_boxes = False
        self.selection_box_history : List[Square] = []
    def setColor(self, color):
        self.color = color
    def setSize(self, size):
        self.size = size
    def setColor(self, parameter):
        self.colorParameter = parameter
    def setSortParameter(self, parameter):
        self.sortParameter = parameter
    def setColorParameter(self, parameter):
        self.colorParameter = parameter
    def setLims(self, lims):
        self.lims = lims
    def setColorbar(self, colorbar):
        self.colorbar = colorbar