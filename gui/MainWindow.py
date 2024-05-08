from PyQt6.QtWidgets import QWidget, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox
from algorithm.DigitRecognizer import DigitRecognizer
from gui.PaintWidget import PaintWidget

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
        self.train_button = QPushButton("Train model (.pkl)")
        self.train_button.clicked.connect(self.handleTrain)
        # input iterations
        self.iterations_input = QLineEdit("2000") # QDoubleSpinBox()
        # self.iterations_input.valueChanged.connect(self.value_changed)
        # load button
        self.load_button = QPushButton("Load model (.pkl)")
        self.load_button.clicked.connect(self.handleLoad)
        # training label
        training_label = QLabel("")
        right_layout.addWidget(self.iterations_input)
        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.load_button)
        right_layout.addWidget(training_label)

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
        print(iterations)
        self.train_button.setText("Training model, please wait ...")
        self.digit_recognizer = DigitRecognizer()
        (X_train, Y_train), (X_test, Y_test) = self.digit_recognizer.load_data()
        self.digit_recognizer.train(X_train, Y_train, X_test, Y_test, iterations)
        self.canvas.set_digit_recognizer(self.digit_recognizer)
        self.train_button.setText("Train model (.pkl)")

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
        