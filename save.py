from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QMessageBox

from typing import Dict, List

from pandas import DataFrame
from core import Stroke
from customization import Customization, FilterSettings

import os
import pickle
import copy
import shutil

class SaveInfo:
    def __init__(self, customization_2d : Customization, last_segment_id : int, was_cut: bool, project_name: str):
        self.customization_2d = customization_2d
        self.last_segment_id = last_segment_id
        self.was_cut = was_cut
        self.project_name = project_name

class QSaveSegments(QWidget):
    def __init__(self, customization_2d : Customization, segments : Dict[str, Stroke], was_cut: bool, project_name : str, original_dataframe : DataFrame):
        super().__init__(parent=None)
        self.original_dataframe = original_dataframe
        self.loaded_project_name = project_name
        self.customization_2d = customization_2d
        self.segments = segments
        self.title = QLabel("Enter project name:")
        self.project_name_line_edit = QLineEdit()
        if self.loaded_project_name is not None:
            self.project_name_line_edit.setText(self.loaded_project_name)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.was_cut = was_cut
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.project_name_line_edit)
        self.layout.addWidget(self.save_button)
        self.setLayout(self.layout)
    def save(self):
        project_name : str = self.project_name_line_edit.text()
        if self.loaded_project_name is not None and project_name == self.loaded_project_name:
            shutil.rmtree(f"data/collections/{self.loaded_project_name}")            
        if project_name.isalnum() and not os.path.isdir(f"data/collections/{project_name}"):
            os.mkdir(f"data/collections/{project_name}")
            os.mkdir(f"data/collections/{project_name}/segments")
            i = 1
            max_id = 0
            for segment_name, segment in self.segments.items():
                if segment.original:
                    segment.restore(self.original_dataframe)
                with open(f"data/collections/{project_name}/segments/{str(i)}", "wb") as file:
                    pickle.dump(segment, file)
                i += 1
                max_id = max(max_id, segment.id)
            with open(f"data/collections/{project_name}/properties", "wb") as file:
                canvas = self.customization_2d.canvas
                ax = self.customization_2d.ax
                self.customization_2d.canvas = None
                self.customization_2d.ax = None
                save_info = SaveInfo(self.customization_2d, max_id, self.was_cut, project_name)
                pickle.dump(save_info, file)
                self.customization_2d.canvas = canvas
                self.customization_2d.ax = ax
            self.loaded_project_name = project_name
            self.close()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Error saving the collection')
            msg.setInformativeText("Bad project name")
            msg.setWindowTitle("Error")
            msg.exec_()
           