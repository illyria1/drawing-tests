from typing import Dict, List
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QLabel, \
QWidget, QHBoxLayout,QPushButton, QFileDialog, \
QSpacerItem,QSizePolicy,QSlider, \
QButtonGroup,QCheckBox,QColorDialog, \
QListWidget, QAbstractItemView, QFrame, QLineEdit

from PyQt5.QtCore import Qt, QSize

from core import Feature

class QInfo(QWidget):
    def __init__(self, data : Dict[str, List[List[Feature]]]):
        super().__init__()
        self.data = data
        self.categories : List[QFeatureCategory] = []
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        for category in self.data:
            category_widget = QFeatureCategory(category, self.data[category])
            category_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.categories.append(category_widget)
            self.layout.addWidget(category_widget)
        self.update(data)
    def update(self, data):
        self.data = data
        for category in self.categories:
            category.update(self.data[category.category_name])


class QFeatureCategory(QWidget):
    def __init__(self, category: str, features: List[List[Feature]]):
        super().__init__()
        self.category_name = category
        self.features = features
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.title_hbox_layout = QHBoxLayout()
        self.features_hbox_layout = QHBoxLayout()
        self.layout.addLayout(self.title_hbox_layout)
        self.layout.addLayout(self.features_hbox_layout)
        self.title = QLabel(category)
        self.title_hbox_layout.addWidget(self.title)
        self.feature_column_vbox_layouts = []
        self.features_widgets : List[List[QFeature]] = []
        for feature_column in features:
            column_layout = QVBoxLayout()
            self.feature_column_vbox_layouts.append(column_layout)
            self.features_hbox_layout.addLayout(column_layout)
            feature_column_widgets = []
            for feature in feature_column:
                feature_widget = QFeature(feature)
                column_layout.addWidget(feature_widget)
                feature_column_widgets.append(feature_widget)
            self.features_widgets.append(feature_column_widgets)
    def update(self, features: List[List[Feature]]):
        self.features = features
        for column_index in range(len(features)):
            features_column = features[column_index]
            for row_index in range(len(features_column)):
                feature = features_column[row_index]
                self.features_widgets[column_index][row_index].update(feature)

class QFeature(QWidget):
    def __init__(self, feature: Feature):
        super().__init__()
        self.feature = feature
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.label = QLabel(self.get_feature_as_str())
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.label)
    def get_feature_as_str(self):
         return "{}: {:.2e} {}".format(
             self.feature.name,
             self.feature.value,
             self.feature.unit_of_measurement
         )
    def update(self, feature: Feature):
        self.feature = feature
        self.label.setText(self.get_feature_as_str())