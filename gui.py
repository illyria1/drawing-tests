import pickle
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QLabel, \
QWidget, QHBoxLayout,QPushButton, QFileDialog, \
QSpacerItem,QSizePolicy,QSlider, \
QButtonGroup,QCheckBox,QColorDialog, \
QListWidget, QAbstractItemView, QFrame, QLineEdit, QScrollArea, QMessageBox, QPlainTextEdit

from PyQt5.QtCore import Qt, QSize, QCoreApplication
from matplotlib.backend_bases import MouseButton

import numpy as np
from model_select import QSelectModel, QSelectType
import params as Params
from load import *
from plot import *
from core import *
from info import *

from matplotlib import animation
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import sys
import copy
from reports import QGenerateReports
from save import QSaveSegments, SaveInfo

from train_ui import QSaveTrainDataSample
from customization import Customization, FilterSettings

import logging

#how many frames animation jumps forward for one animate call
PLAY_SPEED_SLOW = 0.5
PLAY_SPEED_NORMAL = 1
PLAY_SPEED_FAST = 2

ANIMATION_SPEED_SECONDS = 0.033


debug_logger = logging.getLogger('debug')
debug_logger.write = debug_logger.debug    #consider all prints as debug information
debug_logger.flush = lambda: None   # this may be called when printing
#debug_logger.setLevel(logging.DEBUG)      #activate debug logger output
sys.stdout = debug_logger


class QLabelWithValue(QLabel):
    def __init__(self, f_str: str, value):
        super().__init__(f_str.format(value=str(value)))
        self.setContentsMargins(5, 0, 5, 5)
        self.value = str(value)
        self.f_str = f_str
    def update(self, value):
        self.value = str(value)
        super().setText(self.f_str.format(value=str(value)))
        
    
def refresh(customization: Customization, stroke: Stroke, plot_func, ignore_filtered=True):
    if customization.canvas is not None:
        customization.ax.cla()
        ax = plot_func(stroke, customization, ignore_filtered)
        customization.ax = ax
        customization.canvas.draw()

class QGraph3D(QWidget):
    def __init__(self, customization_3d : Customization, current_stroke: Stroke):
        super().__init__()
        self.current_stroke = current_stroke
        self.layout = QVBoxLayout()
        self.canvas_3d = FigureCanvas(Figure())
        self.customization_3d = customization_3d
        self.customization_3d.canvas = self.canvas_3d
        self.ax_3d = plot_3d(self.current_stroke, self.customization_3d)
        self.customization_3d.ax = self.ax_3d
        self.layout.addWidget(self.canvas_3d)
        self.setLayout(self.layout)
        #self.setGeometry(200, 200, 800, 800)

class Graph3DHolder():
    window : QGraph3D = None

class PlayButtons(QWidget):
    def __init__(self, current_stroke: Stroke, customization : Customization):
        super().__init__()
        self.layout = QVBoxLayout()
        self.spacer = QSpacerItem(100,10,QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setLayout(self.layout)
        self.hbox = QHBoxLayout()
        self.hbox_speed_buttons = QHBoxLayout()
        self.is3d = False
        self.canvas_3d = None
        self.moving_slider = False
        self.current_play_speed = PLAY_SPEED_NORMAL
        self.customization_3d = None
        self.customization = customization
        self.current_time_in_s = 0
        self.anim = None
        self.anim_3d = None
        self.paused = False
        self.currentFrame = 0
        self.current_stroke : Stroke = current_stroke
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.current_stroke.main_indexed_df.df) - 1)
        self.slider.valueChanged.connect(self.set_frame)
        self.second_slider = QSlider(Qt.Horizontal)
        self.second_slider.setMinimum(0)
        self.second_slider.setMaximum(len(self.current_stroke.main_indexed_df.df) - 1)
        self.second_slider.valueChanged.connect(self.set_frame)
        self.second_frame = 0
        self.stopButton = QPushButton()
        self.stopButton.setText("Stop")
        self.stopButton.clicked.connect(self.stop_animation)
        self.pauseButton = QPushButton()
        self.pauseButton.setText("Pause")
        self.pauseButton.clicked.connect(self.pause_animation)
        self.playButton = QPushButton()
        self.playButton.setText("Play")
        self.playButton.clicked.connect(self.play_animation)
        self.frameCounter = QLabel("Frame: 0")
        self.frameCounter.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.timeCounter = QLabel("Animation time: 0")
        self.timeCounter.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hbox.addWidget(self.playButton)
        self.hbox.addWidget(self.pauseButton)
        self.hbox.addWidget(self.stopButton)
        self.hbox.addWidget(self.frameCounter)
        self.hbox.addWidget(self.timeCounter)
        self.hbox.addItem(self.spacer)

        self.slowButton = QPushButton()
        self.slowButton.setText(">")
        self.slowButton.setCheckable(True)
        self.normalButton = QPushButton()
        self.normalButton.setText(">>")
        self.normalButton.setCheckable(True)
        self.normalButton.setChecked(True)
        self.fastButton = QPushButton()
        self.fastButton.setText(">>>")
        self.fastButton.setCheckable(True)

        self.btn_grp_speed = QButtonGroup()
        self.btn_grp_speed.setExclusive(True)
        self.btn_grp_speed.addButton(self.slowButton)
        self.btn_grp_speed.addButton(self.normalButton)
        self.btn_grp_speed.addButton(self.fastButton)

        self.hbox_speed_buttons.addWidget(self.slowButton)
        self.hbox_speed_buttons.addWidget(self.normalButton)
        self.hbox_speed_buttons.addWidget(self.fastButton)
        self.hbox_speed_buttons.addItem(self.spacer)
        self.btn_grp_speed.buttonClicked.connect(self.change_speed)

        self.layout.addLayout(self.hbox)
        self.layout.addLayout(self.hbox_speed_buttons)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.second_slider)
        self.infoLabel = QLabel("") #QLabel("Info: ")
        self.infoLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.graph3dholder = None
        self.layout.addWidget(self.infoLabel)
    def update_stroke(self, current_stroke: Stroke):
        self.current_stroke = current_stroke
        self.current_time_in_s = self.current_stroke.filtered_df['t'].min()
        self.refresh_slider()
    def init3D(self, customization_3d : Customization, graph3dholder : Graph3DHolder):
        self.graph3dholder = graph3dholder
        self.customization_3d = customization_3d
        self.enable3DButton = QPushButton()
        self.enable3DButton.setText("Enable 3D")
        self.enable3DButton.clicked.connect(self.enable_3d)
        self.hbox.insertWidget(3, self.enable3DButton)
    def updateFrameCounter(self, frame):
        self.currentFrame = frame
        self.frameCounter.setText(f"Current frame: {frame}")
        self.timeCounter.setText(f"Animation time: {self.current_time_in_s}s")
    def animate(self, i, draw_manually=False):
        if self.current_stroke is not None and self.customization is not None:
            stroke_len = len(self.current_stroke.filtered_df.index)
            if self.currentFrame >= stroke_len:
                self.currentFrame = 0
                self.current_time_in_s = self.current_stroke.filtered_df['t'].min()
                self.stop_animation(True)
                return
            if not draw_manually:
                next_timestamp = self.current_time_in_s + (ANIMATION_SPEED_SECONDS * self.current_play_speed)
                index = self.currentFrame
                while True:
                    index = index + 1
                    if index >= stroke_len:
                        break
                    next_point = self.current_stroke.filtered_df.iloc[index]
                    #print(next_point)
                    if next_point['t'] > next_timestamp:
                        index = index - 1
                        break
                self.current_time_in_s = next_timestamp
                self.currentFrame = index
            else:
                self.current_time_in_s = self.current_stroke.filtered_df.iloc[self.currentFrame]['t']
            #self.update_info()
            self.updateFrameCounter(self.currentFrame)
            self.move_slider(self.currentFrame)
            self.customization.ax.cla()
            if self.currentFrame == 0:
                drawFrame = len(self.current_stroke.filtered_df) - 1
            else:
                drawFrame = self.currentFrame
            plot_2d(self.current_stroke, self.customization, i=drawFrame, i2=self.second_frame)
            if draw_manually:
                self.customization.canvas.draw()
            if self.is3d and self.customization_3d is not None and self.graph3dholder is not None:
                if not self.graph3dholder.window.isHidden():
                    self.customization_3d.ax.cla()
                    plot_3d(self.current_stroke, self.customization_3d, i=drawFrame)
                    self.customization_3d.canvas.draw()
    def update_info(self):
        next_frame = self.currentFrame + 1
        if next_frame >= len(self.current_stroke.filtered_df.index):
            next_frame = 0
        dt = self.current_stroke.filtered_df.iloc[self.currentFrame]['t'] - self.current_stroke.filtered_df.iloc[next_frame]['t']
        new_info_str = f'''DEBUG Info:
x0: {self.current_stroke.filtered_df.iloc[self.currentFrame]['x']}
y0: {self.current_stroke.filtered_df.iloc[self.currentFrame]['y']}
x1: {self.current_stroke.filtered_df.iloc[next_frame]['x']} 
y1: {self.current_stroke.filtered_df.iloc[next_frame]['y']} 
dis: {self.current_stroke.filtered_df.iloc[next_frame]['dis']}
t0: {self.current_stroke.filtered_df.iloc[self.currentFrame]['t']}
t1: {self.current_stroke.filtered_df.iloc[next_frame]['t']} 
dt: {dt}
v0: {self.current_stroke.filtered_df.iloc[self.currentFrame]['velocity']}
v1: {self.current_stroke.filtered_df.iloc[next_frame]['velocity']}
a0: {self.current_stroke.filtered_df.iloc[self.currentFrame]['acceleration']}
a1: {self.current_stroke.filtered_df.iloc[next_frame]['acceleration']}
d3_1: {self.current_stroke.filtered_df.iloc[self.currentFrame]['jerk']}
d3_2: {self.current_stroke.filtered_df.iloc[next_frame]['jerk']}'''
        self.infoLabel.setText(new_info_str)
    def pause_animation(self):
        if self.anim is not None:
            if not self.paused:
                self.paused = True
                self.anim.pause()
    def stop_animation(self, do_not_update_counter=False):
        if self.anim is not None:
            self.current_time_in_s = self.current_stroke.filtered_df['t'].min()
            if not do_not_update_counter:
                self.updateFrameCounter(0)
            self.anim.event_source.stop()
            del self.anim
            self.anim = None
            self.paused = False
            self.customization.ax.cla()
            plot_2d(self.current_stroke, self.customization)
            self.customization.canvas.draw()
            if self.is3d:
                self.customization_3d.ax.cla()
                plot_3d(self.current_stroke, self.customization_3d)
                self.customization_3d.canvas.draw()

    def play_animation(self):
        if self.anim is None:
            if self.current_stroke is not None and self.customization.canvas is not None and self.customization.ax is not None:
                #frames = range(1, len(self.current_data_frame.index), 20)
                self.anim = animation.FuncAnimation(self.customization.canvas.figure, self.animate,frames=1,interval=int(1000*ANIMATION_SPEED_SECONDS),blit=False)
                self.customization.canvas.draw()
        else:
            if self.paused:
                self.paused = False
                self.anim.resume()
    def enable_3d(self):
        if self.customization_3d is not None:
            if self.is3d:
                self.enable3DButton.setText("Enable 3D")
                self.is3d = False
                plot_3d(self.current_stroke, self.customization_3d)
                self.customization_3d.canvas.draw()
            else:
                self.enable3DButton.setText("Disable 3D")
                self.is3d = True
    def move_slider(self, frame):
        self.moving_slider = True
        self.slider.setValue(frame)
        self.moving_slider = False
    def set_frame(self):
        if not self.moving_slider:
            frame_first = self.slider.value()
            self.second_frame = self.second_slider.value()
            self.updateFrameCounter(frame_first)
            self.pause_animation()
            self.animate(frame_first, True)
    def change_speed(self):
        if self.btn_grp_speed.checkedButton() == self.slowButton:
            self.current_play_speed = PLAY_SPEED_SLOW
        elif self.btn_grp_speed.checkedButton() == self.normalButton:
            self.current_play_speed = PLAY_SPEED_NORMAL
        elif self.btn_grp_speed.checkedButton() == self.fastButton:
            self.current_play_speed = PLAY_SPEED_FAST
    def refresh_slider(self):
        length = len(self.current_stroke.main_indexed_df.df) - 1
        self.slider.setMaximum(length)
        self.second_slider.setMaximum(length)
        self.second_slider.setValue(0)

class QSegmentInfo(QWidget):
    def __init__(self, current_segment: Stroke, list_of_segments: QListWidget, model_object_holder: ModelsObjectHolder):
        super().__init__()
        self.model_object_holder = model_object_holder
        self.list_of_segments = list_of_segments
        self.spacer = QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spacer_name = QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spacer_type = QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        #self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame = QFrame()
        #self.frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setStyleSheet("background-color: white")
        self.frame.setLineWidth(4)
        self.frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frameLayout = QVBoxLayout(self.frame)
        self.frameLayout.setContentsMargins(0, 0, 0, 0)
        self.layout = QVBoxLayout()
        self.current_segment = current_segment
        
        self.title = QLabel("Segment info")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title.setContentsMargins(0, 10, 0, 20)
        self.frameLayout.addWidget(self.title)
        
        self.setLayout(self.layout)
        self.layout.addWidget(self.scroll_area)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.select_type_window = None

        self.name_layout = QHBoxLayout()
        self.type_layout = QHBoxLayout()
        self.frameLayout.addLayout(self.name_layout)
        self.frameLayout.addLayout(self.type_layout)

        self.name = QLabelWithValue("Name: {value}", "")
        self.name.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.name_field = QLineEdit()
        self.name_field.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.name_field.setText(current_segment.name)
        self.type = QLabelWithValue("Type: {value}", self.current_segment.type.name)
        self.type.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.patient = QLabelWithValue("Patient ID: {value}", self.current_segment.patient)
        self.patient.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.time= QLabelWithValue("Time: {value}", self.current_segment.time)
        self.time.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hand = QLabelWithValue("Hand: {value}", self.current_segment.hand)
        self.hand.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prediction = QLabelWithValue("Neural network output: {value}", "No model selected")
        self.prediction.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        

        self.note_title = QLabel("Insert a note:")
        self.note_text = QPlainTextEdit()

        self.note_text.textChanged.connect(self.note_changed)

        self.info = QInfo(current_segment.calculate_single_value_features())
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.change_type_button = QPushButton("Change")
        self.change_type_button.clicked.connect(self.activate_select_type_window)
        self.change_type_button.setContentsMargins(5, 0, 5, 0)
        self.change_type_button.setFixedSize(QSize(60, 20))
        self.change_type_button.setStyleSheet("background: lightgray")
        self.change_type_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.change_name_button = QPushButton("Change")
        self.change_name_button.clicked.connect(self.change_name)
        self.change_name_button.setContentsMargins(5, 0, 5, 0)
        self.change_name_button.setFixedSize(QSize(60, 20))
        self.change_name_button.setStyleSheet("background: lightgray")

        self.name_layout.addWidget(self.name)
        self.name_layout.addWidget(self.name_field)
        self.name_layout.addWidget(self.change_name_button)
        self.name_layout.addItem(self.spacer_name)
        self.type_layout.addWidget(self.type)
        self.type_layout.addWidget(self.change_type_button)
        self.type_layout.addItem(self.spacer_type)
        self.frameLayout.addWidget(self.patient)
        self.frameLayout.addWidget(self.time)
        self.frameLayout.addWidget(self.hand)
        self.frameLayout.addWidget(self.prediction)
        self.frameLayout.addWidget(self.note_title)
        self.frameLayout.addWidget(self.note_text)
        self.frameLayout.addWidget(self.info)
        self.frameLayout.addItem(self.spacer)
        self.scroll_area.setWidget(self.frame)
        self.update_type()
        self.update_prediction()
    def note_changed(self):
        text = self.note_text.toPlainText()
        if text:
            self.current_segment.note = text
        else:
            self.current_segment.note = None
    def destroy_select_type_window(self):
        if self.select_type_window is not None:
            self.select_type_window.close()
            self.select_type_window = None
    def activate_select_type_window(self):
        if self.select_type_window is None:
            if self.model_object_holder.model_object is not None:
                self.select_type_window = QSelectType(self.model_object_holder, self)
                self.select_type_window.show()
        else:
            self.select_type_window.close()
            del self.select_type_window
            self.select_type_window = QSelectType(self.model_object_holder, self)
            self.select_type_window.show()
    def update_prediction(self):
        if self.model_object_holder.model_object is not None:
            prediction = self.model_object_holder.model_object.get_parkinsons_prediction(self.current_segment.type.id, self.current_segment)
            self.prediction.update(str(prediction))
            self.current_segment.prediction = prediction
            self.current_segment.used_model = self.model_object_holder.model_object.model_definition.primitives_model_name
    def update_type(self):
        if self.current_segment.forced_type is not None:
            self.current_segment.type = self.current_segment.forced_type
            self.type.update(self.current_segment.forced_type.name)
        elif self.model_object_holder.model_object is not None:
            if self.model_object_holder.model_object.model_definition.default:
                self.type.update(PRIMITIVES_LIST[0].name)
                self.current_segment.type = PRIMITIVES_LIST[0]
            else:
                self.type_predictions = self.model_object_holder.model_object.get_type_prediction(self.current_segment)
                max_primitive : PrimitiveType = max(self.type_predictions, key=self.type_predictions.get)
                if self.type_predictions[max_primitive] >= PRIMITIVE_THRESHOLD:
                    self.current_segment.type = max_primitive
                    self.type.update(max_primitive.name)
                else:
                    self.current_segment.type = PRIMITIVES_LIST[0]
                    self.type.update(PRIMITIVES_LIST[0].name)
    def update_note(self):
        self.note_text.setPlainText(self.current_segment.note)
    def update_segment(self, new_segment : Stroke):
        self.current_segment = new_segment
        self.title.setText(new_segment.name)
        self.name_field.setText(new_segment.name)
        self.update_note()
        self.update_type()
        self.update_prediction()
        self.set_info()
    def set_info(self):
        self.info.update(self.current_segment.calculate_single_value_features())

    def change_name(self):
        item = self.list_of_segments.findItems(self.current_segment.name, Qt.MatchExactly)[0]
        new_name = self.name_field.text()
        if len(new_name) > 0:
            item.setText(new_name)
            self.current_segment.name = new_name
            self.title.setText(new_name)
        



class QAnalysis(QWidget):
    def __init__(self, current_stroke: Stroke, original_df: pd.DataFrame, play_buttons: PlayButtons, customization: Customization, loaded_segments : Dict[str, Stroke] = None, project_name = None):
        super().__init__()
        # self.sliderStrokesLabel = None
        # self.sliderStrokes = None
        # self.sliderStrokeSelectorLabel = None
        # self.sliderStrokeSelector = None
        # self.sliderSquareSizeLabel = None
        # self.sliderSquareSize = None
        # self.sliderPrimitives = None
        # self.sliderPrimitivesLabel = None
        # self.squareSize = FULL_WIDTH
        # self.strokesLabel = None
        # self.primitivePredictionsLabel = None
        # self.strokePredictionsLabel = None
        # self.currentSelectedPrimitiveDF = None
        # self.filteredDataFrame = None
        # self.model = load_model('test')
        self.customization = customization
        self.play_buttons = play_buttons
        self.current_stroke = current_stroke
        self.original_data_frame = original_df
        self.segments = {}
        self.strokes_detected = False
        self.project_name = project_name
        # self.startAnalysisButton = QPushButton("Start analysis")
        # self.predictPrimitiveButton = None
        # if DEV:
        #     self.saveStrokesAutomatically = QPushButton("Saves strokes automatically")
        #     self.saveStrokesAutomatically.clicked.connect(self.save_strokes_automatically)
        # self.startAnalysisButton.clicked.connect(self.start_analysis)
        
        self.model_object_holder = ModelsObjectHolder()
        #self.init_default_model()
        

        self.layout = QHBoxLayout()
        self.leftVBoxLayout = QVBoxLayout()
        self.leftBottomVBoxLayout = QVBoxLayout()
        self.rightVBoxLayout = QVBoxLayout()
        self.leftTopLayout = QVBoxLayout()
        # self.layout.addWidget(self.startAnalysisButton)
        # if DEV:
        #     self.layout.addWidget(self.saveStrokesAutomatically)
        self.setLayout(self.layout)
        self.layout.addLayout(self.leftVBoxLayout)
        self.layout.addLayout(self.rightVBoxLayout)
        self.leftVBoxLayout.addLayout(self.leftTopLayout)
        self.leftVBoxLayout.addLayout(self.leftBottomVBoxLayout)

        self.segmentByHandButton = QPushButton("Enable segmenting by hand")
        self.segmentByHandButton.clicked.connect(self.enableSegmentingByHand)
        self.leftTopLayout.addWidget(self.segmentByHandButton)

        self.saveSelectedSegment = QPushButton("Save selected segment")
        self.saveSelectedSegment.clicked.connect(self.save_selected_segment)
        self.leftTopLayout.addWidget(self.saveSelectedSegment)

        self.strokes_as_segments_button = QPushButton("Cut individual strokes as segments")
        self.strokes_as_segments_button.clicked.connect(self.detect_strokes)
        self.leftTopLayout.addWidget(self.strokes_as_segments_button)

        self.save_training_sample_window = None
        self.save_segments_window = None
        self.select_model_window = None
        self.generate_report_window = None

        self.save_training_sample_button = QPushButton("Save training sample")
        self.save_training_sample_button.clicked.connect(self.save_training_sample)
        self.leftTopLayout.addWidget(self.save_training_sample_button)

        self.save_segments_button = QPushButton("Save collection of segments")
        self.save_segments_button.clicked.connect(self.save_segments)
        self.leftTopLayout.addWidget(self.save_segments_button)

        self.choose_model_button = QPushButton("Select neural network model")
        self.choose_model_button.clicked.connect(self.select_model)
        self.leftTopLayout.addWidget(self.choose_model_button)

        self.generate_pdf_report_button = QPushButton("Generate a PDF report")
        self.generate_pdf_report_button.clicked.connect(self.generate_report)
        self.leftTopLayout.addWidget(self.generate_pdf_report_button)

        self.listOfSegments = QListWidget()
        self.listOfSegments.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listOfSegments.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.segment_info = QSegmentInfo(self.current_stroke, self.listOfSegments, self.model_object_holder)
        self.segment_info.setContentsMargins(0, 0, 0, 0)

        self.restore_button = QPushButton("Original drawing")
        self.restore_button.clicked.connect(self.restore_original_df)
        self.restore_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.select_button = QPushButton("Select segment(s)")
        self.select_button.clicked.connect(self.selectSegment)
        self.select_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        #self.listOfSegments.clicked.connect(self.selectSegment)

       # self.selection_box_size_label = QLabel(f"Select box size\nCurrent size {self.customization.selectionSquareSize}")
        #self.selection_box_size_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #self.selection_box_size_slider = QSlider(Qt.Horizontal)
        #self.selection_box_size_slider.setMinimum(20)
        #self.selection_box_size_slider.setMaximum(150)
        #self.selection_box_size_slider.valueChanged.connect(self.change_selection_box_size)

        self.check_box_show_selection_boxes = QCheckBox("Show all selection boxes")
        self.check_box_show_selection_boxes.setChecked(customization.show_all_selection_boxes)
        self.check_box_show_selection_boxes.stateChanged.connect(self.toggle_show_all_selection_boxes)

        self.rightVBoxLayout.addWidget(self.restore_button)
        self.rightVBoxLayout.addWidget(self.select_button)
        self.rightVBoxLayout.addWidget(self.listOfSegments)
        self.leftTopLayout.addWidget(self.check_box_show_selection_boxes)
        #self.leftTopLayout.addWidget(self.selection_box_size_label)
        #self.leftTopLayout.addWidget(self.selection_box_size_slider)        

        #self.spacer = QSpacerItem(100, 300)
        self.leftBottomVBoxLayout.addWidget(self.segment_info)
        self.leftBottomVBoxLayout.setContentsMargins(0, 10, 10, 0)
        #self.leftVBoxLayout.addItem(self.spacer)

        

        if loaded_segments is not None:
            for segment in loaded_segments.values():
                self.addNewSegment(segment)
        else:
            self.addNewSegment(self.current_stroke)



        # self.strokes = []
        # self.primitivesAutomatic = []
        self.setWindowTitle("Analysis")
    def generate_report(self):
        if self.generate_report_window is None:
            self.generate_report_window = QGenerateReports(self.customization, self.segments, self.original_data_frame)
            self.generate_report_window.show()
        else:
            self.generate_report_window.close()
            del self.generate_report_window
            self.generate_report_window = QGenerateReports(self.customization, self.segments, self.original_data_frame)
            self.generate_report_window.show()
    def save_segments(self):
        if self.save_segments_window is None:
            self.save_segments_window = QSaveSegments(self.customization, self.segments, self.strokes_detected, self.project_name, self.original_data_frame)
            self.save_segments_window.show()
        else:
            self.project_name = self.save_segments_window.loaded_project_name
            self.save_segments_window.close()
            del self.save_segments_window
            self.save_segments_window = QSaveSegments(self.customization, self.segments, self.strokes_detected, self.project_name, self.original_data_frame)
            self.save_segments_window.show()
    def save_selected_segment(self):
        index1 = self.play_buttons.currentFrame
        index2 = self.play_buttons.second_frame
        if index1 > index2:
            points = copy.deepcopy(self.current_stroke.filtered_df.iloc[index2:index1])
        elif index2 < index1:
            points = copy.deepcopy(self.current_stroke.filtered_df.iloc[index1:index2])
        else:
            return
        new_df = IndexedDF(points)
        segment = Stroke(new_df)
        self.addNewSegment(segment)
    def init_default_model(self):
        definitions = load_model_definitions()
        min_model_definition = min(definitions, key=lambda x: x.model_id)
        self.model_object_holder.model_object = ModelsObject(min_model_definition)
    def select_model(self):
        if self.select_model_window is None:
            self.select_model_window = QSelectModel(self.model_object_holder, self.segment_info)
            self.select_model_window.show()
        else:
            self.select_model_window.close()
            del self.select_model_window
            self.select_model_window = QSelectModel(self.model_object_holder, self.segment_info)
            self.select_model_window.show()
    def save_training_sample(self):
        if self.save_training_sample_window is None:
            self.save_training_sample_window = QSaveTrainDataSample(self.current_stroke)
            self.save_training_sample_window.show()
        else:
            self.save_training_sample_window.close()
            del self.save_training_sample_window
            self.save_training_sample_window = QSaveTrainDataSample(self.current_stroke)
            self.save_training_sample_window.show()
    def onSelectionSquareMove(self, event):
        if event.xdata is not None and event.ydata is not None \
            and self.customization.selectionSquareToDraw is not None \
            and self.customization.selectionSquareLeftClicked:
            self.customization.selectionSquareToDraw.set_bottom_right(event.xdata, event.ydata)
            # print(event.xdata)
            # print(event.ydata)
            refresh(self.customization, self.current_stroke, plot_2d, False)
    def onSelectionSquareClick(self, event):
        if event.xdata is not None and event.ydata is not None and event.button is MouseButton.LEFT:
            if not self.customization.selectionSquareLeftClicked:
                self.customization.selectionSquareToDraw = Square(event.xdata, event.ydata,
                                                                    0,
                                                                    0)
                refresh(self.customization, self.current_stroke, plot_2d, False)
                self.customization.selectionSquareLeftClicked = True
            else:
                self.update_selection_box_history(copy.deepcopy(self.customization.selectionSquareToDraw))
                self.saveSegmentOnClick()
                self.customization.selectionSquareToDraw = None
                self.customization.selectionSquareLeftClicked = False
                refresh(self.customization, self.current_stroke, plot_2d, False)
    def enableSegmentingByHand(self):
        if not self.customization.drawSelectionSquare:
            self.customization.drawSelectionSquare = True
            #self.customization.selectionSquareToDraw = Square(0, 0, 0, 0)
            self.segmentByHandButton.setText("Disable segmenting by hand")
            self.customization.canvas.mpl_connect("motion_notify_event", self.onSelectionSquareMove)
            self.customization.canvas.mpl_connect("button_press_event", self.onSelectionSquareClick)
        else:
            self.disableSegmentingByHand()
    def disableSegmentingByHand(self):
        self.segmentByHandButton.setText("Enable segmenting by hand")
        self.customization.drawSelectionSquare = False
        self.customization.selectionSquareToDraw = None
        self.customization.canvas.mpl_disconnect("motion_notify_event")
        self.customization.canvas.mpl_disconnect("button_press_event")
        refresh(self.customization, self.current_stroke, plot_2d, False)
    def saveSegmentOnClick(self):
        if self.customization.selectionSquareToDraw is not None:
            points = []
            for index in range(0, len(self.current_stroke.filtered_df)):
                point = self.current_stroke.filtered_df.iloc[index]
                if is_point_within_square(self.customization.selectionSquareToDraw, point['x'], point['y']):
                    points.append(point)
            if points:
                new_df = IndexedDF(pd.DataFrame(points))
                segment = Stroke(new_df)
                self.addNewSegment(segment)
    def addNewSegment(self, segment: Stroke):
        segment.update_extra(self.current_stroke)
        self.segments[segment.name] = segment
        self.listOfSegments.addItem(segment.name)
    def selectSegment(self):
        items = self.listOfSegments.selectedItems()
        if len(items) == 1:
            segment : Stroke = self.segments.get(items[0].text())
            if segment is not None:
                if segment.original:
                    self.restore_original_df()
                else:
                    self.segment_info.update_segment(segment)
                    self.current_stroke.assign_stroke(segment)
                    refresh(self.customization, self.current_stroke, plot_2d)
                    self.play_buttons.update_stroke(self.current_stroke)
        else:
            segments_dfs = []
            for item in items:
                segment : Stroke = self.segments.get(item.text())
                if segment is not None:
                    segments_dfs.append(segment.main_df)
            merged_df = pd.concat(segments_dfs, axis=0).drop_duplicates().reset_index(drop=True)
            segment = Stroke(IndexedDF(merged_df), merged=True)
            self.addNewSegment(segment)
            self.segment_info.update_segment(segment)
            self.current_stroke.assign_stroke(segment)
            refresh(self.customization, self.current_stroke, plot_2d)
            self.play_buttons.update_stroke(self.current_stroke)
            
        # segment = self.segments.get(self.listOfSegments.currentItem().text())
        # if segment is not None:
        #     self.current_stroke.assign_stroke(segment)
        #     refresh(self.customization, self.current_stroke, plot_2d)
        #     self.play_buttons.refresh_slider()
    def restore_original_df(self):
        self.current_stroke.restore(self.original_data_frame)
        self.segment_info.update_segment(self.current_stroke)
        self.play_buttons.refresh_slider()
        refresh(self.customization, self.current_stroke, plot_2d)
    def change_selection_box_size(self):
        new_size = self.selection_box_size_slider.value()
        self.customization.selectionSquareSize = new_size
        self.selection_box_size_label.setText(f"Select box size\nCurrent size {self.customization.selectionSquareSize}")
    def toggle_show_all_selection_boxes(self):
        self.customization.show_all_selection_boxes = self.check_box_show_selection_boxes.isChecked()
        refresh(self.customization, self.current_stroke, plot_2d)
    def update_selection_box_history(self, selection_box: Square):
        self.customization.selection_box_history.append(selection_box)

            
    # def start_analysis(self):
    #     self.detect_strokes()
    #     self.automatic_primitive_predict()
    #     self.initUI()
    # def automatic_primitive_predict(self):
    #     if self.strokes:
    #         for stroke in self.strokes:
    #             df = stroke.df[stroke.begin:stroke.end+1]
    #             square_index = 0
    #             while square_index < len(df) - 1:
    #                 index = square_index
    #                 square_point = df.iloc[index]
    #                 square = Square(square_point['x'], square_point['y'], self.squareSize, self.squareSize)
    #                 points = []
    #                 for index in range(0, len(df)):
    #                     point = df.iloc[index]
    #                     if is_point_within_square(square, point['x'], point['y']):
    #                         points.append(point)
    #                 if points:
    #                     new_df = pd.DataFrame(points)
    #                     primitive = Primitive(new_df, 0, len(points) - 1, None, None)
    #                     predictions = self.get_predictions(primitive)
    #                     prediction = predictions[0]
    #                     prediction_index = np.argmax(prediction)
    #                     if prediction[prediction_index] >= PRIMITIVE_THRESHOLD:
    #                         primitive.type = prediction_index + 1
    #                         primitive.predictions = predictions
    #                         self.primitivesAutomatic.append(primitive)
    #                         square_index = index + 1
    #                         continue
    #                 square_index += 10
    def detect_strokes(self):
        if not self.strokes_detected:
            last_index = 0
            self.detecting_strokes_filtered_df = self.current_stroke.main_df[self.current_stroke.main_df['p'] >= 0.1]
            l = len(self.detecting_strokes_filtered_df)
            prev_stroke_index = 0
            for index in range(0, len(self.detecting_strokes_filtered_df)):
                if index == 0:
                    continue
                # dt = self.detecting_strokes_filtered_df.iloc[index]['t'] - self.detecting_strokes_filtered_df.iloc[index - 1]['t']
                # # dis = distance((filtered_df.iloc[index]['x'], filtered_df.iloc[index]['y']),
                # #         (filtered_df.iloc[index - 1]['x'], filtered_df.iloc[index - 1]['y']))
                # if dt >= 0.2 or index == l-1:
                #     segment = Stroke(IndexedDF(self.detecting_strokes_filtered_df, last_index, index - 1))
                #     self.addNewSegment(segment)
                #     last_index = index
                stroke_index = self.detecting_strokes_filtered_df.iloc[index]['stroke_index']
                if stroke_index != prev_stroke_index or index == l-1:
                    segment = Stroke(IndexedDF(self.detecting_strokes_filtered_df, last_index, index - 1))
                    self.addNewSegment(segment)
                    last_index = index
                    prev_stroke_index = stroke_index
            self.strokes_detected = True
            self.strokes_as_segments_button.setDisabled(True)
            if self.save_segments_window is not None:
                self.save_segments_window.was_cut = True
    # def initUI(self):
    #     if self.strokesLabel is None:
    #         self.strokesLabel = QLabel("Current stroke: 0")
    #         self.layout.addWidget(self.strokesLabel)
    #     if self.strokePredictionsLabel is None:
    #         self.strokePredictionsLabel = QLabel("Stroke predicitons: ")
    #         self.layout.addWidget(self.strokePredictionsLabel)
    #     if self.sliderStrokes is None:
    #         self.sliderStrokes = QSlider(Qt.Horizontal)
    #         self.sliderStrokes.valueChanged.connect(self.show_stroke)
    #         self.sliderStrokesLabel = QLabel("Select stroke")
    #         self.layout.addWidget(self.sliderStrokesLabel)
    #         self.layout.addWidget(self.sliderStrokes)
    #     if self.sliderSquareSize is None:
    #         self.sliderSquareSize = QSlider(Qt.Horizontal)
    #         self.sliderSquareSize.valueChanged.connect(self.change_square_size)
    #         self.sliderSquareSize.setMinimum(32)
    #         self.sliderSquareSize.setMaximum(FULL_WIDTH)
    #         self.sliderSquareSizeLabel = QLabel("Select detection square size")
    #         self.layout.addWidget(self.sliderSquareSizeLabel)
    #         self.layout.addWidget(self.sliderSquareSize)
    #     if self.sliderStrokeSelector is None:
    #         self.sliderStrokeSelector = QSlider(Qt.Horizontal)
    #         self.sliderStrokeSelector.valueChanged.connect(self.show_stroke_within_square)
    #         self.sliderStrokeSelector.setMinimum(0)
    #         self.sliderStrokeSelector.setMaximum(len(self.filteredDataFrame.index) - 1)
    #         self.sliderStrokeSelectorLabel = QLabel("Move selection square")
    #         self.layout.addWidget(self.sliderStrokeSelectorLabel)
    #         self.layout.addWidget(self.sliderStrokeSelector)
    #     if self.predictPrimitiveButton is None:
    #         self.predictPrimitiveButton = QPushButton("Predict primitive")
    #         self.predictPrimitiveButton.clicked.connect(self.predict_primitive)
    #         self.layout.addWidget(self.predictPrimitiveButton)
    #     if self.primitivePredictionsLabel is None:
    #         self.primitivePredictionsLabel = QLabel("Primitive predictions: ")
    #         self.layout.addWidget(self.primitivePredictionsLabel)
    #     if self.primitivesAutomatic and self.sliderPrimitives is None:
    #         self.sliderPrimitives = QSlider(Qt.Horizontal)
    #         self.sliderPrimitives.valueChanged.connect(self.show_primitive)
    #         self.sliderPrimitives.setMinimum(0)
    #         self.sliderPrimitives.setMaximum(len(self.primitivesAutomatic) - 1)
    #         self.sliderPrimitivesLabel = QLabel("Select automatic primitives: ")
    #         self.layout.addWidget(self.sliderPrimitivesLabel)
    #         self.layout.addWidget(self.sliderPrimitives)
    #     self.sliderStrokes.setMinimum(0)
    #     self.sliderStrokes.setMaximum(len(self.strokes) - 1)
    # def predict_primitive(self):
    #     if self.currentSelectedPrimitiveDF is not None and self.primitivePredictionsLabel is not None:
    #         stroke = Stroke(self.currentSelectedPrimitiveDF, 0, len(self.currentSelectedPrimitiveDF) - 1, RED)
    #         predictions = self.get_predictions(stroke)
    #         self.primitivePredictionsLabel.setText(f"Primitive predictions: {str(predictions)}")
    # def get_predictions(self, stroke):
    #     arr = self.convert_stroke(stroke)
    #     arr = np.asarray([arr])
    #     predictions = self.model.predict(arr)
    #     return predictions
    # def change_square_size(self):
    #     if self.sliderSquareSize is not None:
    #         self.squareSize = self.sliderSquareSize.value()
    #         self.show_stroke_within_square()
    # def show_stroke(self):
    #     if self.strokes and self.sliderStrokes and self.strokesLabel:
    #         stroke_index = self.sliderStrokes.value()
    #         self.strokesLabel.setText(f"Current stroke: {stroke_index}")
    #         stroke = self.strokes[stroke_index]
    #         predictions = self.get_predictions(stroke)
    #         if self.strokePredictionsLabel is not None:
    #             self.strokePredictionsLabel.setText(f"Stroke predicitons: {str(predictions)}")
    #         df = stroke.df[stroke.begin:stroke.end+1]
    #         refresh(self.customization, df, plot_2d)
    #         self.save_stroke(stroke, "test", True)
    # def show_primitive(self):
    #     if self.primitivesAutomatic:
    #         primitiveIndex = self.sliderPrimitives.value()
    #         primitive = self.primitivesAutomatic[primitiveIndex]
    #         df = primitive.df[primitive.begin:primitive.end+1]
    #         self.primitivePredictionsLabel.setText(f"Primitive predictions: {str(primitive.predictions)}")
    #         refresh(self.customization, df, plot_2d)
    # def show_stroke_within_square(self):
    #     if self.filteredDataFrame is not None and self.sliderStrokeSelector is not None:
    #         points = []
    #         start_index = self.sliderStrokeSelector.value()
    #         start_point = self.filteredDataFrame.iloc[start_index]
    #         points.append(start_point)
    #         index = start_index
    #         square = Square(start_point['x'], start_point['y'], self.squareSize, self.squareSize)
    #         for index in range(0, len(self.filteredDataFrame)):
    #             point = self.filteredDataFrame.iloc[index]
    #             if is_point_within_square(square, point['x'], point['y']):
    #                 points.append(point)
    #         df_to_show = pd.DataFrame(points)
    #         self.currentSelectedPrimitiveDF = df_to_show
    #         refresh(self.customization, df_to_show, plot_2d)
    # #FOR DEV  -------------------------------------
    # def save_strokes_automatically(self):
    #     samples = [(0, 9, 1), (10, 20, 2), (21, 31, 2), (32, 37, 3), (38, 44, 3), (45, 52, 1), (53, 60, 4), (61, 69, 4)]
    #     picture = 1
    #     for sample in samples:
    #         stroke_type = sample[2]
    #         if self.strokes:
    #             start = sample[0]
    #             end = sample[1]
    #             for index in range(start, end + 1):
    #                 stroke = self.strokes[index]
    #                 self.save_stroke(stroke, str(picture) + '-' + str(index) + '-' + str(stroke_type), True)
    # #-----------------------------------------------
    # def convert_stroke(self, stroke):
    #     res = np.zeros((WIDTH, HEIGHT))
    #     half_width = int(WIDTH / 2)
    #     half_height = int(HEIGHT / 2)
    #     df = stroke.df[stroke.begin:stroke.end+1]
    #     xmin = df['x'].min()
    #     xmax = df['x'].max()
    #     ymin = df['y'].min()
    #     ymax = df['y'].max()

    #     xmid = df['x'].mean()
    #     ymid = df['y'].mean()

    #     #diff_x = point_x_mid - square_x_mid
    #     diff_x = xmid #xmin
    #     diff_y = ymid #ymin
    #     #print("x:" + str(xmax - xmin))
    #     #print("y:" + str(ymax - ymin))
        
    #     for index in range(0, len(df)):
    #         x_orig = df.iloc[index]['x']
    #         y_orig = df.iloc[index]['y']
            
    #         x = int(x_orig - diff_x) // SCALE + half_width
    #         y = int(y_orig - diff_y) // SCALE + half_height
    #         angles = np.arange(0, 2*np.pi, 0.1)
    #         radii = np.arange(SCALE, RADIUS, 0.2)
    #         for angle in angles:
    #             for dr in radii:
    #                 x_circle = int(((x_orig - diff_x) + dr * np.cos(angle))) // SCALE + half_width
    #                 y_circle = int(((y_orig - diff_y) + dr * np.sin(angle))) // SCALE + half_height
    #                 if x_circle >= 0 and x_circle < WIDTH and y_circle >= 0 and y_circle < HEIGHT:
    #                     res[y_circle][x_circle] = 1
    #         if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT:
    #             res[y][x] = 1
    #         else:
    #             print("out of bounds")
    #     return res
    # def save_stroke(self, stroke: Stroke, filename: str, txt=False):
    #     res = self.convert_stroke(stroke)
    #     if txt:
    #         with open('strokes_txt/' + filename + '.txt', 'w') as file:
    #             for line in res.tolist():
    #                 file.write(str(line) + '\n')
    #     np.save('data/train/' + filename, res)

class QCustomize(QWidget):
    def __init__(self, customization : Customization, playButtons : PlayButtons, current_stroke: Stroke, plot_func):
        super().__init__()
        self.colorDialog = QColorDialog(self)
        self.colorButton = QPushButton("Select color")
        self.colorButton.clicked.connect(self.select_color)
        self.playButtons = playButtons
        if (customization.colorParameter is not None):
            self.colorButton.setEnabled(False)
        else:
            self.colorButton.setEnabled(True)
        self.customization = customization
        self.current_stroke : Stroke = current_stroke
        layout = QVBoxLayout()
        self.plot_func = plot_func
        self.sizeLabel = QLabel(f"Point thickness: {self.customization.size}")
        self.sizeSlider = QSlider(Qt.Horizontal)
        self.sizeSlider.setMinimum(1)
        self.sizeSlider.setMaximum(50)
        self.sizeSlider.setValue(self.customization.size)
        self.colorParameterLabel = QLabel("Display variable as color: ")
        self.checkBox1 = QCheckBox("Pressure")
        self.checkBox1.setChecked(customization.colorParameter is not None)
        self.checkBox1.stateChanged.connect(lambda:self.change_color_parameter(self.checkBox1))
        self.checkBox2 = QCheckBox("Toggle axis")
        self.checkBox2.setChecked(customization.toggleAxis)
        self.checkBox2.stateChanged.connect(lambda:self.toggle_axis(self.checkBox2))
        layout.addWidget(self.sizeLabel)
        layout.addWidget(self.sizeSlider)
        layout.addWidget(self.colorParameterLabel)
        layout.addWidget(self.checkBox1)
        layout.addWidget(self.colorButton)
        layout.addWidget(self.checkBox2)
        if self.customization.lims is not None:
            if self.customization.filter_settings.filter_value is not None:
                self.filterLabel = QLabel(f"Filter points with less pressure: {self.customization.filter_settings.filter_value}")
            else:
                self.filterLabel = QLabel(f"Filter points with less pressure: 0")
            self.filterSlider = QSlider(Qt.Horizontal)
            self.filterSlider.setMinimum(0)
            self.filterSlider.setMaximum(int(self.customization.lims[5] * 100))
            self.filterSlider.setValue(0)
            self.filterSlider.valueChanged.connect(self.change_filter)
            layout.addWidget(self.filterLabel)
            layout.addWidget(self.filterSlider)
        self.sizeSlider.valueChanged.connect(self.change_size)
        self.setLayout(layout)
        self.setWindowTitle("Customize")
    def change_size(self):
        self.customization.setSize(self.sizeSlider.value())
        self.sizeLabel.setText(f"Point thickness: {self.customization.size}")
        self.refresh()
    def change_filter(self):
        self.customization.filter_settings.filter_value = self.filterSlider.value() / 100
        if self.customization.filter_settings.filter_value == 0:
            self.filterLabel.setText(f"Filter points with less pressure: 0")
            self.customization.filter_settings.filter_value = None
            self.current_stroke._filtered_df = None
            self.playButtons.slider.setMaximum(len(self.current_stroke.main_df) - 1)
            self.customization.show_filtered_df = False
        else:
            self.filterLabel.setText(f"Filter points with less pressure: {self.customization.filter_settings.filter_value}")
            self.current_stroke._filtered_df = self.current_stroke.main_indexed_df.df[self.current_stroke.main_indexed_df.df[self.customization.filter_settings.parameter] >= self.customization.filter_settings.filter_value]
            self.playButtons.slider.setMaximum(len(self.current_stroke.filtered_df) - 1)
            self.customization.show_filtered_df = True
            self.playButtons.stop_animation()

        self.refresh()
    def refresh(self):
        if self.customization.canvas is not None:
            self.customization.ax.cla()
            ax = self.plot_func(self.current_stroke, self.customization)
            self.customization.ax = ax
            self.customization.canvas.draw()
        #if self.canvas_3d:
            #plot_3d(self.current_data_frame, self.canvas_3d, self.customization, ax=self.ax_3d)
    def change_color_parameter(self, button):
        if button.text() == "Pressure":
            if button.isChecked() == True:
                self.customization.addColorbar = True
                self.customization.colorParameter = 'p'
                self.colorButton.setEnabled(False)
            else:
                self.customization.deleteColorbar = True
                self.customization.colorParameter = None
                self.colorButton.setEnabled(True)
        self.refresh()
    def select_color(self):
        qcolor = self.colorDialog.getColor()
        self.customization.color = qcolor.name()
        self.refresh()
    def toggle_axis(self, button):
        if button.text() == "Toggle axis":
            self.customization.toggleAxis = button.isChecked()
        self.refresh()

class QMain(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.canvas_2d = None
        self.ax_2d = None
        self.canvas_3d = None
        self.ax_3d = None

        self.current_mode = Params.MODE_2D

        self.customizeWindow = None
        self.customizeWindow3D = None
        self.analysisWindow = None

        self.playButtons = None

        self.customization = Customization(30)
        self.customization.setColorParameter('p')
        self.customization_3d = Customization(30)
        self.customization_3d.setColorParameter('p')
        self.customization.filter_settings = FilterSettings(None, 'p')
        self.customization_3d.filter_settings = self.customization.filter_settings

        self.current_stroke = None
        self.original_data_frame = None

        self.lims = None
        self.lims_3d = None

        self.label = QLabel()
        self.label.setText("No data loaded")
        self.label.setAlignment(Qt.AlignCenter)

        self.graph_3d_holder = Graph3DHolder()

        self.currentVelocityLabel = None
        self.currentAccelerationLabel = None

        self.load_button = QPushButton()
        self.load_saved_button = QPushButton()
        self.customizeButton = QPushButton()
        self.customize3DButton = QPushButton()
        self.analysisButton = QPushButton()
        self.open3DButton = QPushButton()
    
        self.open3DButton.setText("Open 3D graph")
        self.analysisButton.setText("Analysis")
        self.customizeButton.setText("Customize 2D")
        self.customize3DButton.setText("Customize 3D")
        self.load_button.setText("Load new sample")
        self.load_saved_button.setText("Load saved collection")
        self.load_button.clicked.connect(self.load_file)
        self.load_saved_button.clicked.connect(self.load_collection)
        self.customizeButton.clicked.connect(self.show_customize_menu)
        self.analysisButton.clicked.connect(self.show_analysis_menu)
        self.customize3DButton.clicked.connect(self.show_customize3d_menu)
        self.open3DButton.clicked.connect(self.open_3d_graph)
        self.spacer = QSpacerItem(100,10,QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.load_button_hbox = QHBoxLayout()
        self.load_button_hbox.addWidget(self.load_button)
        self.load_button_hbox.addWidget(self.load_saved_button)
        self.load_button_hbox.addWidget(self.open3DButton)
        self.load_button_hbox.addWidget(self.customizeButton)
        self.load_button_hbox.addWidget(self.customize3DButton)
        self.load_button_hbox.addWidget(self.analysisButton)
        self.load_button_hbox.addItem(self.spacer)

        self.project_name = None

        self.loaded_strokes_detected = None

        self.graphs = QHBoxLayout()

        self.loaded_segments = None

        self.plot_2d = QVBoxLayout()
        #self.plot_3d = QVBoxLayout()
        self.graphs.addLayout(self.plot_2d)
        #self.graphs.addLayout(self.plot_3d)

        self.main_vbox = QVBoxLayout()
        self.main_vbox.addLayout(self.load_button_hbox)
        self.main_vbox.addLayout(self.graphs)
        self.main_vbox.addWidget(self.label)

        self.setLayout(self.main_vbox)
        self.setGeometry(200,200,1000,800)
        self.setWindowTitle('View tests')
    def open_3d_graph(self):
        if self.current_stroke is not None:
            if self.graph_3d_holder.window is None:
                self.graph_3d_holder.window = QGraph3D(self.customization_3d, self.current_stroke)
                self.playButtons.init3D(self.customization_3d, self.graph_3d_holder)
            else:
                self.graph_3d_holder.window.activateWindow()
            self.graph_3d_holder.window.show()
    def close_child_windows(self):
        if self.customizeWindow is not None:
            self.customizeWindow.close()
            self.customizeWindow = None
        if self.customizeWindow3D is not None:
            self.customizeWindow3D.close()
            self.customizeWindow3D = None
        if self.analysisWindow is not None:
            self.analysisWindow.close()
            self.analysisWindow = None
        if self.graph_3d_holder.window is not None:
            self.graph_3d_holder.window.close()
            self.graph_3d_holder.window = None
    def load_collection(self):
        folderpath = QFileDialog.getExistingDirectory(self, 'Select Collection folder')
        if folderpath:
            if self.label is not None:
                self.label.setText("Loading...")
                self.label.update()
            self.close_child_windows()
            try:
                #loading
                with open(f"{folderpath}/properties", 'rb') as file:
                    save_info: SaveInfo = pickle.load(file)
                    self.customization = save_info.customization_2d
                    Stroke.id_segment = itertools.count(start=save_info.last_segment_id+1)
                    self.loaded_strokes_detected = save_info.was_cut
                    self.project_name = save_info.project_name
                    if self.analysisWindow is not None:
                        self.analysisWindow.project_name = self.project_name
                    if self.analysisWindow is not None and self.loaded_strokes_detected:
                        self.analysisWindow.strokes_detected = True
                        self.analysisWindow.strokes_as_segments_button.setDisabled(True)
                directory = os.fsencode(f"{folderpath}/segments")
                self.current_stroke = None
                self.original_data_frame = None
                self.loaded_segments = {}
                for segment_file in os.listdir(directory):
                    filename = os.fsdecode(segment_file)
                    with open(f"{folderpath}/segments/{filename}", "rb") as file:
                        segment : Stroke = pickle.load(file)
                        self.loaded_segments[segment.name] = segment
                        if segment.original:
                            self.original_data_frame = segment.main_df
                            self.current_stroke = segment
                if self.current_stroke is None:
                    raise RuntimeError("No main dataframe found!")
                #gui init
                if self.label is not None:
                    self.label.setParent(None)
                    self.label = None
                self.lims = self.customization.lims
                self.customization_3d.setLims(self.lims)
                self.customization.selection_box_history.clear()
                self.load_init_gui()
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('Error loading the collection')
                msg.setInformativeText(str(e))
                msg.setWindowTitle("Error")
                msg.exec_()
    def load_file(self):
        file , check = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()",
                                               "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        if check:
            self.customization.selection_box_history.clear()
            Stroke.id_segment = itertools.count(start=0)
            if self.label is not None:
                self.label.setText("Loading...")
                self.label.repaint()
            self.loaded_strokes_detected = None
            self.close_child_windows()
            if self.label is not None:
                self.label.setParent(None)
                self.label = None
            #self.clear_layout(self.plot_2d)
            #self.clear_layout(self.plot_3d)
            #if self.playButtons is not None:
                #self.playButtons.setParent(None)
                #self.playButtons = None
            
            self.lims = None
            self.customization.lims = None
            loaded_data_tuple = load_data(file)
            self.original_data_frame = loaded_data_tuple[0]
            self.update_time()


            #self.current_data_frame = filter_by_pressure(self.current_data_frame)
            x0 = self.original_data_frame['x'].min()
            xmax = self.original_data_frame['x'].max()
            y0 = self.original_data_frame['y'].min()
            ymax = self.original_data_frame['y'].max()
            t0 = self.original_data_frame['t'].min()
            tmax = self.original_data_frame['t'].max()

            p0 = self.original_data_frame['p'].min()
            pmax = self.original_data_frame['p'].max()

            self.lims = (x0, xmax, y0, ymax, p0, pmax, t0, tmax)
            self.customization.setLims(self.lims)
            self.customization_3d.setLims(self.lims)

            calculate_vector_features(self.original_data_frame)

            self.current_stroke = Stroke(
                IndexedDF(self.original_data_frame),
                original=True,
                **loaded_data_tuple[1]
                )
            self.load_init_gui()
    def load_init_gui(self):
            if self.canvas_2d is None:
                self.canvas_2d = FigureCanvas(Figure())
                self.customization.canvas = self.canvas_2d
                #self.plot_2d.addWidget(NavigationToolbar(self.canvas_2d, self))
                self.plot_2d.addWidget(self.canvas_2d)
                self.customization.addColorbar = True
                self.ax_2d = plot_2d(self.current_stroke, self.customization)
                self.customization.ax = self.ax_2d
            else:
                self.customization.addColorbar = True
                self.customization.canvas = self.canvas_2d
                self.customization.ax = None
                self.canvas_2d.figure.clear()
                self.ax_2d = plot_2d(self.current_stroke, self.customization)
                self.customization.ax = self.ax_2d
                self.customization.canvas.draw()
                

            if self.playButtons is None:
                self.playButtons = PlayButtons(self.current_stroke, self.customization)
                self.playButtons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.main_vbox.addWidget(self.playButtons)
            else:
                self.playButtons.stop_animation()
                self.playButtons.update_stroke(self.current_stroke)
                #self.customization.canvas.draw()

            # self.canvas_3d = FigureCanvas(Figure())
            # self.customization_3d.canvas = self.canvas_3d
            # #self.plot_3d.addWidget(NavigationToolbar(self.canvas_3d, self))
            # self.plot_3d.addWidget(self.canvas_3d)
            # self.ax_3d = plot_3d(self.current_stroke, self.customization_3d)
            # self.customization_3d.ax = self.ax_3d
            # self.playButtons.init3D(self.customization_3d)
    def show_customize_menu(self):
        if self.current_stroke is not None and self.playButtons is not None:
            if self.customizeWindow is None:
                self.customizeWindow = QCustomize(self.customization, self.playButtons, self.current_stroke, plot_2d)
            else:
                self.customizeWindow.activateWindow()
            self.customizeWindow.show()
    def show_customize3d_menu(self):
        if self.current_stroke is not None and self.playButtons is not None and self.customization_3d is not None:
            if self.customizeWindow3D is None:
                self.customizeWindow3D = QCustomize(self.customization_3d, self.playButtons, self.current_stroke, plot_3d)
            else:
                self.customizeWindow3D.activateWindow()
            self.customizeWindow3D.show()
    def show_analysis_menu(self):
        if self.current_stroke is not None:
            if self.analysisWindow is None:
                self.analysisWindow = QAnalysis(self.current_stroke, self.original_data_frame, self.playButtons, self.customization, self.loaded_segments, self.project_name)
                if self.loaded_strokes_detected:
                    self.analysisWindow.strokes_detected = True
                    self.analysisWindow.strokes_as_segments_button.setDisabled(True)
            else:
                self.analysisWindow.activateWindow()
            self.analysisWindow.show()
    def clear_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    def update_time(self):
        tmin = self.original_data_frame['t'].min()
        if self.lims is None and self.original_data_frame is not None:
            for index in range(0, len(self.original_data_frame)):
                self.original_data_frame.iloc[index]['t'] = (self.original_data_frame.iloc[index]['t'] - tmin)
    def closeEvent(self, event) -> None:
        QApplication.closeAllWindows()
        QApplication.quit()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QMain()  
    main.show()
    main.activateWindow()
    main.raise_()
    sys.exit(app.exec())