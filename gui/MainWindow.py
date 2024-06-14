from PyQt6.QtWidgets import QWidget, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QScrollArea, QLabel, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox
from algorithm.DigitRecognizer import DigitRecognizer
from gui.PaintWidget import PaintWidget
from PyQt6.QtCore import QThread, pyqtSignal
from time import sleep



class MainWindow(QMainWindow):
    def __init__(self, digit_recognizer: DigitRecognizer=None):
        super().__init__()
        self.digit_recognizer = digit_recognizer
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
        # input iterations
        self.iterations_input = QLineEdit("2000") # QDoubleSpinBox()
        # self.iterations_input.valueChanged.connect(self.value_changed)
        # load button
        self.load_button = QPushButton("Load model (.pkl)")
        self.load_button.clicked.connect(self.handleLoad)
        # training label
        self.training_label = QLabel("")
        # Create a QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.training_label)
        self.scroll_area.setWidgetResizable(True)

        right_layout.addWidget(self.iterations_input)
        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.load_button)
        right_layout.addWidget(self.scroll_area)

        # Center widget
        center_widget = QWidget()
        center_layout = QVBoxLayout()
        center_widget.setLayout(center_layout)
        # results text
        results_label = QLabel("draw a number, and click on 'Recognize'") # font=("Helvetica", 48)
        predict_label = QLabel("")
        center_layout.addWidget(results_label)
        center_layout.addWidget(predict_label)

        # Left widget
        left_widget = QWidget()
        left_widget.setFixedSize(300,300)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        # put the canvas in the left widget
        self.canvas = PaintWidget(self.digit_recognizer, results_label, predict_label)
        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.canvas.clear)
        # recognize button
        recognize_button = QPushButton('Recognize')
        recognize_button.clicked.connect(self.canvas.classify_handwriting)
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
        iterations = int(self.iterations_input.text())
        self.train_button.setText("Training model, please wait ...")
        self.digit_recognizer = DigitRecognizer()
        self.training_label.setText("")

        self.worker = ParallelWorker(self.scroll_area, self.training_label, iterations, self.digit_recognizer)
        self.worker.result_signal.connect(self.trainFinished)
        self.worker.start()
        self.canvas.set_digit_recognizer(self.digit_recognizer)
        self.train_button.setText("Train model and save it (.pkl)")

    def trainFinished(self, history):
        self.digit_recognizer.show_evaluation(history)

    def handleLoad(self):
        (fname, _) = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${trained_params.pkl}",
            "trained_params (*.pkl);;",
        )
        self.train_button.setText(f"Loading  model '{_}'")
        self.digit_recognizer = DigitRecognizer()
        self.digit_recognizer.load_model(file_path=fname)
        self.canvas.set_digit_recognizer(self.digit_recognizer)
        self.train_button.setText("Load model (.pkl)")
        

class ParallelWorker(QThread):
    result_signal = pyqtSignal(object)

    def __init__(self, scroll_area: QScrollArea, label: QLabel, iterations: int, digit_recognizer: DigitRecognizer):
        super().__init__()
        self.scroll_area = scroll_area
        self.label = label
        self.iterations = iterations
        self.digit_recognizer = digit_recognizer


    def run(self):
        (X_train, Y_train), (X_test, Y_test) = self.digit_recognizer.load_data()
        training = self.digit_recognizer.train(X_train, Y_train, X_test, Y_test, self.iterations)
        try:
            while True:
                history, W1, b1, W2, b2 = next(training)
                train_accuracy = history["train"]["accuracy"][-1]
                train_loss = history["train"]["loss"][-1]
                val_accuracy = history["validation"]["accuracy"][-1]
                val_loss = history["validation"]["loss"][-1]
                i = history["iterations"][-1]
                text = f"""Iteration: {i} / {self.iterations}\nTraining Accuracy: {train_accuracy:.3%} | Training Loss: {train_loss:.4f}\nValidation Accuracy: {val_accuracy:.3%} | Validation Loss: {val_loss:.4f}"""
                print(text)
                self.label.setText(f"{self.label.text()}\n{text}")
                # Scroll to the bottom of the QScrollArea
                
                sleep(0.1)
                self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        except StopIteration:
            pass
        self.result_signal.emit(history)
