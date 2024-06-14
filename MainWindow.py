from PyQt6.QtWidgets import QWidget, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QScrollArea, QLabel, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox
from NeuralNetworkModel import NeuralNetworkModel
from PaintWidget import *
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QThread, pyqtSignal
from time import sleep



class MainWindow(QMainWindow):
    def __init__(self):#, model: NeuralNetworkModel=None):
        super().__init__()
        self.model = None
        self.initUI()

    def initUI(self):
        self.setGeometry(50, 50, 1200, 500)
        self.setWindowTitle("Digit Recognizer")
        # self.move(QApplication.primaryScreen().geometry().center() - self.frameGeometry().center())

        # Right widget
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        # train button
        self.train_button = QPushButton("Train model and save it as (.pkl)")
        self.train_button.clicked.connect(self.handleTrain)
        # iterations
        iterations_label = QLabel("Iterations (Epochs)")
        self.iterations_input = QLineEdit(f"{NeuralNetworkModel.EPOCHS}") # QDoubleSpinBox()
        # accurancy
        accurancy_label = QLabel("Target accurancy")
        self.accurancy_input = QLineEdit(f"{NeuralNetworkModel.TARGET_ACCURANCY}") # QDoubleSpinBox()
        # learning rate
        learning_rate_label = QLabel("learning Rate")
        self.learning_rate_input = QLineEdit(f"{NeuralNetworkModel.LEARNING_RATE}") # QDoubleSpinBox()
        # load button
        self.load_button = QPushButton("Load model (.pkl)")
        self.load_button.clicked.connect(self.handleLoad)
        # training label
        self.training_label = QLabel("")
        # Create a QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.training_label)
        self.scroll_area.setWidgetResizable(True)

        right_layout.addWidget(iterations_label)
        right_layout.addWidget(self.iterations_input)
        right_layout.addWidget(accurancy_label)
        right_layout.addWidget(self.accurancy_input)
        right_layout.addWidget(learning_rate_label)
        right_layout.addWidget(self.learning_rate_input)
        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.load_button)
        right_layout.addWidget(self.scroll_area)

        # Center widget
        center_widget = QWidget()
        center_layout = QVBoxLayout()
        center_widget.setLayout(center_layout)
        # results text
        results_label = QLabel("")
        results_label.setFont(QFont('Helvetica', 32))
        predict_label = QLabel("")
        center_layout.addWidget(results_label)
        center_layout.addWidget(predict_label)

        # Left widget
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        info_label = QLabel("draw a number, and click on 'Recognize'")
        # put the canvas in the left widget
        self.canvas = PaintWidget(self.model, results_label, predict_label)
        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.canvas.clear)
        # recognize button
        recognize_button = QPushButton('Recognize')
        recognize_button.clicked.connect(self.canvas.classify_handwriting)
        left_layout.addWidget(info_label)
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(recognize_button)
        left_layout.addWidget(clear_button)
        left_widget.setLayout(left_layout)

        
        # main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(center_widget)
        main_layout.addWidget(right_widget)

    def handleTrain(self):
        try:
            iterations = int(self.iterations_input.text()) 
        except Exception:
            iterations = None
        try:
            target_accurancy = float(self.accurancy_input.text())
        except Exception:
            target_accurancy = None
        try:
            learning_rate = float(self.learning_rate_input.text())
        except Exception:
            learning_rate = None
        
        if learning_rate == None:
            dlg = CustomDialog("a error occurred", "please type a valid learning rate parameter (must be float)!")
            dlg.exec()
            return
                
        self.train_button.setText("Training model, please wait ...")
        self.training_label.setText("")
        self.model = NeuralNetworkModel()
        self.worker = ParallelWorker(self.scroll_area, self.training_label, iterations, target_accurancy, learning_rate, self.model)
        self.worker.result_signal.connect(self.trainFinished)
        self.worker.start()
        self.canvas.set_model(self.model)
        self.train_button.setText("Train model and save it (.pkl)")

    def trainFinished(self):
        self.model.show_evaluation()

    def handleLoad(self):
        (fname, _) = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${trained_params.pkl}",
            "trained_params (*.pkl);;",
        )
        self.train_button.setText(f"Loading  model '{_}'")
        self.model = NeuralNetworkModel()
        self.model.load_model(file_path=fname)
        self.canvas.set_model(self.model)
        self.train_button.setText("Load model (.pkl)")
        

class ParallelWorker(QThread):
    result_signal = pyqtSignal()

    def __init__(self, scroll_area: QScrollArea, label: QLabel, iterations: int, target_accurancy: float, learning_rate: float, model: NeuralNetworkModel):
        super().__init__()
        self.scroll_area = scroll_area
        self.label = label
        self.iterations = iterations
        self.target_accurancy = target_accurancy
        self.learning_rate = learning_rate
        self.model = model


    def run(self):
        training = self.model.train(target_accurancy=self.target_accurancy, epochs=self.iterations, learning_rate=self.learning_rate)
        try:
            while True:
                text, epoch, train_accurancy, train_loss, val_accurancy, val_loss = next(training)
                self.label.setText(f"{self.label.text()}\n{text}")
                # Scroll to the bottom of the QScrollArea
                sleep(0.1)
                self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        except StopIteration:
            pass
        self.result_signal.emit()
