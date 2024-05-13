from typing import Dict, List
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QRadioButton, QHBoxLayout, QPushButton

from core import ModelDefinition, ModelsObject, ModelsObjectHolder, load_model_definitions

class QSelectModel(QWidget):
    def __init__(self, models_object_holder: ModelsObjectHolder, segment_widget):
        super().__init__(parent=None)
        self.setWindowTitle("Select model")
        self.segment_widget = segment_widget
        self.models_object_holder = models_object_holder
        self.layout = QVBoxLayout()
        self.model_definitions : List[ModelDefinition] = load_model_definitions()
        #print(self.model_definitions)
        self.radio_btns : Dict[int, QRadioButton] = {}
        self.model_definitions_dict : Dict[int, ModelDefinition] = {}
        self.select_model_group_box = QGroupBox("Select model")
        self.vbox = QVBoxLayout()
        self.default_model_radio_btn = QRadioButton("General model")
        self.default_model_radio_btn.toggled.connect(self.select_default_model)
        self.vbox.addWidget(self.default_model_radio_btn)
        if models_object_holder.model_object is not None and models_object_holder.model_object.model_definition.default:
            self.default_model_radio_btn.setChecked(True)
        for model_definition in self.model_definitions:
            primitives_names = tuple(map(lambda x: x.name, model_definition.primitives))
            name = f"id: {model_definition.model_id} primitives: {primitives_names}"
            radio_btn = QRadioButton(name)
            radio_btn.toggled.connect(self.change_model)
            if models_object_holder.model_object is not None:
                if model_definition.model_id == models_object_holder.model_object.model_definition.model_id:
                    radio_btn.setChecked(True)
            self.vbox.addWidget(radio_btn)
            self.radio_btns[model_definition.model_id] = radio_btn
            self.model_definitions_dict[model_definition.model_id] = model_definition
        self.select_model_group_box.setLayout(self.vbox)
        self.layout.addWidget(self.select_model_group_box)
        self.setLayout(self.layout)
    def change_model(self):
        for model_id in self.radio_btns:
            radio_btn = self.radio_btns[model_id]
            if radio_btn.isChecked():
                self.models_object_holder.model_object = ModelsObject(self.model_definitions_dict[model_id])
                self.segment_widget.destroy_select_type_window()
                self.segment_widget.update_type()
                self.segment_widget.update_prediction()
                break
    def select_default_model(self):
        if self.default_model_radio_btn.isChecked():
            self.models_object_holder.model_object = ModelsObject(ModelDefinition(default=True))
            self.segment_widget.update_type()
            self.segment_widget.update_prediction()

class QSelectType(QWidget):
    def __init__(self, models_object_holder: ModelsObjectHolder, segment_widget):
        super().__init__(parent=None)
        self.segment_widget = segment_widget
        self.layout = QVBoxLayout()
        self.radio_btns : Dict[int, QRadioButton] = {}
        self.select_model_group_box = QGroupBox("Select type")
        self.vbox = QVBoxLayout()
        self.automatic_radio_btn = QRadioButton("Automatic")
        if self.segment_widget.current_segment.forced_type is None:
            self.automatic_radio_btn.setChecked(True)
            self.automatic_radio_btn.toggled.connect(self.set_to_automatic)
        self.vbox.addWidget(self.automatic_radio_btn)
        if models_object_holder.model_object is None:
            raise RuntimeError("Model object must initialized in order to open this window!!!")
        for primitive in models_object_holder.model_object.model_definition.primitives:
            radio_btn = QRadioButton(primitive.name)
            radio_btn.toggled.connect(self.change_type)
            self.vbox.addWidget(radio_btn)
            self.radio_btns[primitive] = radio_btn
            if self.segment_widget.current_segment.forced_type is not None and self.segment_widget.current_segment.forced_type.id == primitive.id:
                radio_btn.setChecked(True)
        self.select_model_group_box.setLayout(self.vbox)
        self.layout.addWidget(self.select_model_group_box)
        self.setLayout(self.layout)
    def set_to_automatic(self):
        if self.automatic_radio_btn.isChecked():
            #print("Back to automatic!")
            self.segment_widget.current_segment.forced_type = None
            self.segment_widget.update_type()
            self.segment_widget.update_prediction()
    def change_type(self):
        for primitive in self.radio_btns:
            radio_btn = self.radio_btns[primitive]
            if radio_btn.isChecked():
                self.segment_widget.current_segment.forced_type = primitive
                self.segment_widget.update_type()
                self.segment_widget.update_prediction()
                break
