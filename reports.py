from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QMessageBox, QCheckBox

from typing import Dict, List
from core import Stroke, get_last_index_in_folder
from customization import Customization, FilterSettings

from matplotlib.backends.backend_pdf import PdfPages

from plot import plot_2d

import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

class QGenerateReports(QWidget):
    def __init__(self, customization_2d : Customization, segments : Dict[str, Stroke], original_data_frame):
        super().__init__(parent=None)
        self.layout = QVBoxLayout()
        self.setWindowTitle("Generate report")

        self.segments = segments
        self.customization = customization_2d

        self.original_data_frame = original_data_frame

        self.include_segments_label = QLabel("Select segments to include in report: ")
        self.layout.addWidget(self.include_segments_label)
        self.checkbox_list : List[QCheckBox] = []
        for segment in segments.values():
            checkbox = QCheckBox(segment.name)
            checkbox.setChecked(True)
            self.checkbox_list.append(checkbox)
            self.layout.addWidget(checkbox)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_report)
        self.layout.addWidget(self.generate_button)

        self.setLayout(self.layout)
    def generate_report(self):
        segments_to_include : List[Stroke] = []
        self.generate_button.setText("Generating....")
        self.generate_button.repaint()
        for checkbox in self.checkbox_list:
            if checkbox.isChecked():
                name = checkbox.text()
                segment = self.segments[name]
                segments_to_include.append(segment)
        if segments_to_include:
            last_index = get_last_index_in_folder('data/reports', '.pdf')
            file = f"data/reports/report-{last_index + 1}.pdf"
            with PdfPages(file) as pdf:
                for segment in segments_to_include:
                    if segment.original:
                        segment.restore(self.original_data_frame)
                    ax_backup = self.customization.ax
                    canvas_backup = self.customization.canvas
                    self.customization.ax = None
                    self.customization.canvas = None
                    customization_copy = copy.deepcopy(self.customization)
                    self.customization.ax = ax_backup
                    self.customization.canvas = canvas_backup

                    class CustomCanvas:
                        figure = None

                    fig = plt.figure(figsize=(8,12))
                    canvas = CustomCanvas()
                    canvas.figure = fig 

                    gs = gridspec.GridSpec(3, 1, height_ratios=[6, 1, 4])


                    #ax1 = fig.add_subplot(2, 1, 1)
                    #ax3 = fig.add_subplot(2, 1, 2)

                    ax1 = fig.add_subplot(gs[0])
                    ax2 = fig.add_subplot(gs[1])
                    ax3 = fig.add_subplot(gs[2])

                    customization_copy.ax = ax1
                    customization_copy.canvas = canvas

                    plot_2d(segment, customization_copy)
                    customization_copy.ax.set_title(segment.name)

                    features_table = segment.get_features_as_table()
                    #kyrka = [['a', 'b'], ['c', 'd']]
                    #ax2.axis('off')
                    #ax2.axis('tight')

                    ax2.axis('off')
                    #ax2.axis('tight')

                    ax3.axis('off')
                    #ax3.axis('tight')

                    text = ""
                    #print(segment.patient)
                    if segment.patient is not None:
                        text = text + "Patiend ID: " + segment.patient + '\n'
                    if segment.time is not None:
                        text = text + "Time: " + segment.time + '\n'
                    if segment.hand is not None:
                        text = text + "Hand: " + segment.hand + '\n'
                    if segment.prediction is not None:
                        text = text + f"Neural network output: {segment.prediction}" + '\n'
                    if segment.note is not None:
                        text = text + "Note: " + segment.note
                    
                    t = ax2.text(.01, .99, text, ha='left', va='top', transform=ax2.transAxes)
                    #t = ax2.table([["fglkglkergelkg\ngdfgsjgrjgrtjgkrtjgltrkjgtrkljgkrtlgjrtlkg\njgjkterklgjrkgjerlkgjkltrjglkrtjglktrjglkg\ngnjkgnrtjkgrtjkghrtjkgbrtjkghtrjkghrtjkghrthjg\ngrg"]])

                    # table1 = ax3.table(cellText=[["kyrka kyrka kyrka"]],
                    #                                     cellLoc='left',
                    #                                     loc='upper left')
                    # if segment.note:
                    #     print(segment.note)
                    #     table1 = ax3.table(cellText=[[segment.note]],
                    #                 cellLoc='left',
                    #                 loc='upper left')
                    # if segment.prediction:
                    #     table2 = ax3.table(cellText=[[f"Neural network prediction: {segment.prediction}"], [f"Selected model: kyrka"]],
                    #                 cellLoc='left',
                    #                 loc='upper left')
                    table = ax3.table(cellText=features_table,
                                  cellLoc='left',
                                  loc='upper left')
                    table.auto_set_font_size(False)
                    table.set_fontsize(6)

                    ax3.set_title("Features")
                    
                    pdf.savefig(fig)

                    # self.customization.ax.cla()
                    # plot_2d(segment, self.customization)
                    # self.customization.ax.set_title(segment.name)
                    # features_table = segment.get_features_as_table()
                    # self.customization.ax.table(cellText=features_table,
                    #                             loc='bottom',
                    #                             edges='open')
                    # self.customization.canvas.figure.subplots_adjust(left=0.2, bottom=0.2)
                    # pdf.savefig(self.customization.canvas.figure)
        self.close()
                


    
        