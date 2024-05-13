import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import json
import numpy as np

from core import distance, Stroke
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from customization import Customization, FilterSettings

codes_selection_square = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]

COLOR_MAP = plt.cm.get_cmap("coolwarm")

def plot_vector(point, vector):
    #print(point['x'])
    #print(point['y'])
    #print(point['x'] + vector[0])
    #print(point['y'] + vector[1])
    #if point['x'] + vector[0] > 1000 or point['x'] + vector[0] < 0:
        #return
    #if point['y'] + vector[0] > 1000 or point['y'] + vector[0] < 0:
        #return

    lenght = distance((0, 0), (vector[0], vector[1]))
    X = [point['x']]
    Y = [point['y']]
    U = [vector[0]]
    V = [vector[1]]
    #if (vector[1] < 0):
    plt.quiver(X, Y, U, V, color=['r'], units='xy', scale=4)

def save_plot(test_data_frame, width, height, dpi, filename):
    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.scatter(test_data_frame['x'], test_data_frame['y'], s=1, c='#000000')
    plt.savefig(filename, dpi=dpi)

def plot_2d(stroke: Stroke, customization: Customization, ignore_filtered=False, **kwargs):
    test_data_frame = stroke.main_df
    fig = customization.canvas.figure
    if not ignore_filtered and customization.show_filtered_df:
        test_data_frame = stroke.filtered_df
    if customization.color is not None:
        color = customization.color
    else:
        color = '#000000'
    if customization.deleteColorbar:
        fig.clear()
    if customization.ax and not customization.deleteColorbar:
        ax = customization.ax
    else:
        ax = fig.subplots()
    if customization.deleteColorbar:
        #print("delete colorbar")
        #fig.delaxes(fig.axes[1])
        #fig.subplots_adjust(right=0.90)
        #pos = ax.get_position(original=True)
        #ax._set_position(pos)
        customization.deleteColorbar = False
        customization.colorbar = None
    if customization.lims:
        lims = customization.lims
        x0 = lims[0]
        xmax = lims[1]
        y0 = lims[2]
        ymax = lims[3]
        ax.set_xlim([x0 - 10, xmax + 10])
        ax.set_ylim([y0 - 10, ymax + 10])
    if "i" in kwargs:
        if "i2" in kwargs:
            if kwargs["i2"] <= kwargs["i"]:
                test_data_frame = test_data_frame[kwargs["i2"]:kwargs["i"]]
            else:
                test_data_frame = test_data_frame[kwargs["i"]:kwargs["i2"]]
        else:
            test_data_frame = test_data_frame[:kwargs["i"]]
    if not customization.toggleAxis:
        ax.axis('off')
    else:
        ax.axis('on')
    if customization.invertY:
        ax.invert_yaxis()
    if customization.colorParameter:
        if customization.sortParameter:
            test_data_frame = test_data_frame.sort_values(by=[customization.sortParameter])
        if customization.lims:
            lims = customization.lims
            vmin = lims[4]
            vmax = lims[5]
        else:
            vmin = min(test_data_frame[customization.colorParameter])
            vmax = max(test_data_frame[customization.colorParameter])
        sc = ax.scatter(test_data_frame['x'], test_data_frame['y'], c=test_data_frame[customization.colorParameter], vmin=vmin, vmax=vmax, cmap=COLOR_MAP, s=customization.size)
        if customization.addColorbar:
            #print("add colorbar")
            customization.colorbar = fig.colorbar(sc)
            customization.addColorbar = False
    else:        
        sc = ax.scatter(test_data_frame['x'], test_data_frame['y'], s=customization.size,c=color)
    if customization.show_all_selection_boxes:
            for box in customization.selection_box_history:
                path = Path(box.points, codes_selection_square)
                pathpatch = PathPatch(path, facecolor='none', edgecolor='red')
                ax.add_patch(pathpatch)
    if customization.drawSelectionSquare and customization.selectionSquareToDraw is not None:
        path = Path((customization.selectionSquareToDraw.points), codes_selection_square)
        pathpatch = PathPatch(path, facecolor='none', edgecolor='red')
        ax.add_patch(pathpatch)
    return ax

def plot_3d(stroke: Stroke, customization: Customization, ignore_filtered=False, **kwargs):
    test_data_frame = stroke.main_df
    fig = customization.canvas.figure
    if not ignore_filtered and customization.show_filtered_df is True:
        test_data_frame = stroke.filtered_df
    if customization.color is not None:
        color = customization.color
    else:
        color = '#000000'
    if customization.ax:
        ax = customization.ax
    else:
        ax = fig.add_subplot(projection='3d')
    if customization.lims:
        lims = customization.lims
        x0 = lims[0]
        xmax = lims[1]
        y0 = lims[2]
        ymax = lims[3]
        ax.set_xlim([x0 - 10, xmax + 10])
        ax.set_ylim([y0 - 10, ymax + 10])
        z0 = lims[6]
        zmax = lims[7]
        ax.set_zlim([z0 - 10, zmax + 10])
    if "i" in kwargs:
        test_data_frame = test_data_frame[:kwargs["i"]]
    if customization.colorParameter:
        if customization.sortParameter:
            test_data_frame = test_data_frame.sort_values(by=[customization.sortParameter])
        cm = plt.cm.get_cmap("coolwarm")
        if customization.lims:
            lims = customization.lims
            vmin = lims[4]
            vmax = lims[5]
        else:
            vmin = min(test_data_frame[customization.colorParameter])
            vmax = max(test_data_frame[customization.colorParameter])
        sc = ax.scatter(test_data_frame['x'], test_data_frame['y'], test_data_frame['t'], s=customization.size, c=test_data_frame[customization.colorParameter], vmin=vmin, vmax=vmax, cmap=cm)
        if "add_colorbar" in kwargs and kwargs["add_colorbar"] == True:
            fig.colorbar(sc)
    else:
        ax.scatter(test_data_frame['x'], test_data_frame['y'], test_data_frame['t'], c=color,s=customization.size)
    return ax

#plot_2d(test_data_frame, color="pressure")
#plot_3d(test_data_frame)
#plot_3d(test_data_frame, color="pressure")
#input()