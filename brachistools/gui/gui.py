
import typing
import os
from PyQt6 import QtCore, QtGui
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QMainWindow,
    QApplication,
    QGridLayout,
    QCheckBox,
    QPushButton,
    QPlainTextEdit,
    QLabel,
    QProgressBar,
    QProgressDialog,
    QFileDialog,
    QMessageBox,
    QListView,
    QWidget,
    QAction)

from .io import load_folder, imread

from ..version import brachistools_version
from ..segmentation import segmentation_pipeline
from ..classification import classification_pipeline

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"brachistools v{brachistools_version}")

        # Set central widget for layout control
        self.cwidget = QWidget(self)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(6)

        # Add UI components
        self.SelectInputButton = QPushButton('Select input')
        self.BatchCheckBox = QCheckBox('Select all')
        self.SegmentationButton = QPushButton('Segmentation')
        self.DiagnosisButton = QPushButton('Suggest diagnosis')
        self.SaveButton = QPushButton('Save')
        self.InputFolderLabel = QLabel()
        self.PrevImgButton = QPushButton('Previous')
        self.ImgFileLabel = QLabel()
        self.NextImgButton = QPushButton('Next')
        self.InputListView = QListView()
        self.ImgDisplayLabel = QLabel()
        self.DiagnosisLabel = QLabel('Suggested diagnosis:')
        self.DiagnosisTextEdit = QPlainTextEdit()

        # Initialize UI components
        self._init_ui_components()

        # Set data slots
        self._input_filenames = []
        self._input_folder_path = None
        self._curr_index = None
        self._curr_img = None

    def help_window(self):
        ...

    def segmentation_window(self):
        ...

    def segment_current(self):
        ...

    def classify_current(self):
        ...

    def segment_all(self):
        QMessageBox.critical(self, "Not supported", "Sorry, this option is under development")

    def classify_all(self):
        QMessageBox.critical(self, "Not supported", "Sorry, this option is under development")

    def do_segment(self):
        if self.BatchCheckBox.isChecked():
            self.segment_all()
        else:
            self.segment_current()

    def do_classify(self):
        if self.BatchCheckBox.isChecked():
            self.classify_all()
        else:
            self.classify_current()

    def select_image(self, index):
        if not self._input_filenames:
            QMessageBox.critical(self,
                                "Invalid operation",
                                "Input file list is empty")
            return

        selected_img_fn = self._input_filenames[index]
        try:
            self._curr_img = imread(os.path.join(self._input_folder_path, selected_img_fn))
        except:
            QMessageBox.critical(self, "Invalid operation", "Failed to load image")
            return

        self._curr_index = index
        self.ImgFileLabel.setText(selected_img_fn)
        self.load_image(selected_img_fn)

    def prev_image(self):
        self.select_image((self._curr_index - 1) % len(self._input_filenames))

    def next_image(self):
        self.select_image((self._curr_index + 1) % len(self._input_filenames))

    def load_image(self, im_fn):
        pixmap = QtGui.QPixmap(fileName=im_fn)
        self.ImgDisplayLabel.setPixmap(pixmap)
        self.ImgDisplayLabel.setScaledContents(True)
        self.ImgDisplayLabel.show()

    def select_input(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select input folder")
        input_filenames = load_folder(folder_path, ['PNG', 'JPG', 'JPEG'])
        if not input_filenames:
            QMessageBox.critical(self, "Invalid operation", "Input folder does not contain any PNG/JPG files")
        else:
            self._input_folder_path = folder_path
            self.InputFolderLabel.setText(folder_path)
            self._input_filenames = input_filenames

    def keyPressEvent(self, event):
        if self._loaded:
            modifiers = (event.modifiers() & (
                    QtCore.Qt.ControlModifier |
                    QtCore.Qt.ShiftModifier |
                    QtCore.Qt.AltModifier
                ))
            if not modifiers:
                if event.key() == QtCore.Qt.Key_Left:
                    self.prev_image()
                if event.key() == QtCore.Qt.Key_Right:
                    self.next_image()

    def _init_ui_components(self):
        self._init_slots()
        self._init_layout()

    def _init_layout(self):
        self.l0.addWidget(self.SelectInputButton, 0, 0, 1, 3)
        woff = 3
        self.l0.addWidget(self.BatchCheckBox, 0, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.SegmentationButton, 0, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.DiagnosisButton, 0, woff, 1, 2)
        woff += 2

        self.l0.addWidget(self.InputFolderLabel, 1, 0, 1, 3)
        woff = 3
        self.l0.addWidget(self.PrevImgButton, 1, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.ImgFileLabel, 1, woff, 1, 2)
        woff += 4
        self.l0.addWidget(self.NextImgButton, 1, woff, 1, 2)

        self.InputListViewModel = QtCore.QStringListModel()
        # self.InputListViewModel.resetInternalData()
        self.InputListView.setModel(self.InputListViewModel)
        self.l0.addWidget(self.InputListView, 2, 0, 7, 3)
        woff = 3
        self.l0.addWidget(self.ImgDisplayLabel, 2, woff, 6, 8)
        self.l0.addWidget(self.DiagnosisLabel, 8, woff, 1, 4,
                          QtCore.Qt.AlignmentFlag.AlignRight)
        woff += 4
        self.l0.addWidget(self.DiagnosisTextEdit, 8, woff, 1, 2)

    def _init_slots(self):
        self.SelectInputButton.clicked.connect(self.select_input)

        self.SegmentationButton.clicked.connect(self.do_segment)
        self.DiagnosisButton.clicked.connect(self.do_classify)

        self.InputListView.clicked.connect(self.select_image)
        self.PrevImgButton.clicked.connect(self.prev_image)
        self.NextImgButton.clicked.connect(self.next_image)
