from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QRadioButton, QHBoxLayout, QPushButton

from core import Stroke, convert_stroke, save_features, save_stroke
from params import PRIMITIVES_LIST, PARK_POSITIVE, PARK_NEGATIVE, PARKINSONS, PRIMITIVES

class QSaveTrainDataSample(QWidget):
    def __init__(self, current_stroke: Stroke):
        super().__init__(parent=None)
        
        self.layout = QVBoxLayout()
        self.current_stroke = current_stroke

        self.select_value_primitive = 0
        self.select_value_parkinsons = PARK_NEGATIVE

        self.select_sample_value_group_parkinsons = QGroupBox("Select value")
        self.select_sample_value_group_primitives = QGroupBox("Select value")

        self.select_sample_value_parkinsons_vbox = QVBoxLayout()
        self.select_sample_value_primitives_vbox = QVBoxLayout()

        self.positive_btn = QRadioButton("Positive")
        self.negative_btn = QRadioButton("Negative")
        self.negative_btn.setChecked(True)

        self.positive_btn.toggled.connect(self.change_value_parkinsons)
        self.negative_btn.toggled.connect(self.change_value_parkinsons)

        self.select_sample_value_parkinsons_vbox.addWidget(self.positive_btn)
        self.select_sample_value_parkinsons_vbox.addWidget(self.negative_btn)
        
        self.radio_buttons_primitives : dict[int, QRadioButton] = {}

        for primitive in PRIMITIVES_LIST:
            radio_btn = QRadioButton(primitive.name)
            radio_btn.toggled.connect(self.change_value_primitive)
            self.select_sample_value_primitives_vbox.addWidget(radio_btn)
            self.radio_buttons_primitives[primitive.id] = radio_btn
        
        self.radio_buttons_primitives[0].setChecked(True)

        self.select_sample_value_group_parkinsons.setLayout(self.select_sample_value_parkinsons_vbox)
        self.select_sample_value_group_primitives.setLayout(self.select_sample_value_primitives_vbox)

        self.values_hbox = QHBoxLayout()
        self.buttons_hbox = QHBoxLayout()

        self.values_hbox.addWidget(self.select_sample_value_group_parkinsons)
        self.values_hbox.addWidget(self.select_sample_value_group_primitives)

        self.save_button_primitive = QPushButton("Save primitive data")
        self.save_button_primitive.clicked.connect(self.save_primitive)

        self.save_button = QPushButton("Save features data")
        self.save_button.clicked.connect(self.save_train_data)

        self.buttons_hbox.addWidget(self.save_button)
        self.buttons_hbox.addWidget(self.save_button_primitive)

        self.layout.addLayout(self.values_hbox)
        self.layout.addLayout(self.buttons_hbox)

        self.setLayout(self.layout)
        self.setWindowTitle("Save sample")
    def change_value_parkinsons(self):
        if self.positive_btn.isChecked():
            self.select_value_parkinsons = PARK_POSITIVE
        elif self.negative_btn.isChecked():
            self.select_value_parkinsons = PARK_NEGATIVE
        #print(self.select_value_parkinsons)
    def change_value_primitive(self):
        for primitive_id in self.radio_buttons_primitives:
            if self.radio_buttons_primitives[primitive_id].isChecked():
                self.select_value_primitive = primitive_id
                #print(primitive_id)
    def save_primitive(self):
        if self.select_value_primitive is not None:
            converted_stroke = convert_stroke(self.current_stroke)
            save_stroke(converted_stroke, str(self.select_value_primitive))
            self.save_button_primitive.setDisabled(True)
    def save_train_data(self):
        if self.select_value_parkinsons is not None and self.select_value_primitive is not None:
            features = self.current_stroke.get_features_as_plain_dict()
            save_features(features, f"{self.select_value_parkinsons}-{self.select_value_primitive}")
            self.save_button.setDisabled(True)