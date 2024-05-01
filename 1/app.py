import sys, io
from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QFont, QCursor
from PyQt6.QtCore import Qt, QPoint, QSize
from DigitRecognizer import *
from PIL import ImageQt


class MainWindow(QMainWindow):
    def __init__(self, digit_recognizer=None):
        super().__init__()
        self.digit_recognizer = digit_recognizer
        self.setGeometry(100, 100, 1000, 500)
        self.setWindowTitle("Digit Recognizer")
        # self.move(QApplication.primaryScreen().geometry().center() - self.frameGeometry().center())
        

        # results text
        results_label = QLabel("Thinking..") # font=("Helvetica", 48)
        predict_label = QLabel("")
        # put the canvas in the left widget
        canvas = PaintWidget(self.digit_recognizer, results_label, predict_label)
        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(canvas.clear)
        # recognize button
        recognize_button = QPushButton('Recognize')
        recognize_button.clicked.connect(canvas.classify_handwriting)

        # Left widget
        left_widget = QWidget()
        left_widget.setFixedSize(400,400)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(canvas)
        left_layout.addWidget(recognize_button)
        left_layout.addWidget(clear_button)
        left_widget.setLayout(left_layout)

        # Right widget
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        right_layout.addWidget(results_label)
        right_layout.addWidget(predict_label)

        
        # main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)


class PaintWidget(QWidget):
    def __init__(self, digit_recognizer, results_label, predict_label, width=400, height=400):
        super().__init__()
        self.digit_recognizer = digit_recognizer
        self.results_label = results_label
        self.predict_label = predict_label
        self.initUI()

    def initUI(self):
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.points = []
        # self.results_label.setFont(QFont('Arial', 10)) 
        # model_json, model_summary = self.digit_recognizer.log()
        # self.results_label.setText(model_summary)
        
    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Set white background
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(self.rect())
        # Set black border
        painter.setPen(QPen(Qt.GlobalColor.white, 0, Qt.PenStyle.SolidLine))
        painter.drawRect(self.rect())
        # draw points
        painter.setPen(QPen(Qt.GlobalColor.black, 10, Qt.PenStyle.SolidLine))
        painter.drawPoints(self.points)
        # for i in range(1, len(self.points)):
        #     x, y = self.points[i].x(), self.points[i].y()
        #     painter.drawPoint(x, y)
        #     # painter.drawLine(self.points[i - 1], self.points[i])
        #     # r = 8
        #     # painter.drawArc(x-r, y-r, r, r, 0, 360)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # self.points = [event.pos()]
            self.points.append(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.MouseButton.LeftButton:
            self.points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # self.points = []
            self.update()

    def to_image(self):
        # Create a QImage with the same size as the widget
        qimage = QImage(self.size(), QImage.Format.Format_RGB32)
        qimage.fill(255)  # Fill the image with white background
        # Render the widget onto the image
        painter = QPainter(qimage)
        self.render(painter)
        painter.end()
        # Save the image
        # image.save('drawn_image.png')  # Save the image to a file
        return ImageQt.fromqimage(qimage) # image

    def clear(self):
        self.points = []
        self.update()

    def classify_handwriting(self):
        img = self.to_image()
        img = DigitRecognizer.process_image(img)
        digit, accuracy, predictions = self.digit_recognizer.make_predictions(img)
        self.results_label.setFont(QFont('Helvetica', 40))
        self.results_label.setText(' digit : {} \n accuracy: {:.2f}%'.format(digit, accuracy))
        predictions_text = ["{} => {:.2f}%".format(digit, accuracy) for (digit, accuracy) in predictions]
        self.predict_label.setText("\n".join(predictions_text))
        print(predictions)
        # digit, accuracy, prediction = self.digit_recognizer.predict_digit(img)
        # self.results_label.setFont(QFont('Helvetica', 40))
        # self.results_label.setText(' digit : {} \n accuracy: {}%'.format(digit, int(accuracy*100)))
        # predictions = ["{} => {:.2f}%".format(d, a*100) for d, a  in enumerate(prediction)]
        # self.predict_label.setText("\n".join(predictions))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    digit_recognizer = DigitRecognizer()
    digit_recognizer.load_model()
    window = MainWindow(digit_recognizer)
    window.show()
    sys.exit(app.exec())
