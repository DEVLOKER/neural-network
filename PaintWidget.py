
from PyQt6.QtWidgets import QWidget, QDialog, QDialogButtonBox, QLabel, QVBoxLayout
from PyQt6.QtGui import QPainter, QPen, QImage, QCursor
from PyQt6.QtCore import Qt
from PIL import ImageQt
from NeuralNetworkModel import NeuralNetworkModel



class PaintWidget(QWidget):
    def __init__(self, model: NeuralNetworkModel, results_label, predict_label, width=400, height=400):
        super().__init__()
        self.model = model
        self.results_label = results_label
        self.predict_label = predict_label
        self.predict_label.setWordWrap(True)
        self.setFixedSize(300,300)
        self.initUI()

    def initUI(self):
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.points = []
        # self.results_label.setFont(QFont('Arial', 10)) 
        # model_json, model_summary = self.model.log()
        # self.results_label.setText(model_summary)
        
    def set_model(self, model):
        self.model = model

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
        if self.model is None:
            dlg = CustomDialog("a error occurred", "please load or train a model first!")
            dlg.exec()
            return
        img = self.to_image()
        img = NeuralNetworkModel.process_image(img)
        digit, accuracy, predictions = self.model.make_predictions(img)
        self.results_label.setText(' digit : {} \n accuracy: {:.2f}%'.format(digit, accuracy))
        predictions_text = ["{} => {:.2f}%".format(digit, accuracy) for (digit, accuracy) in predictions]
        self.predict_label.setText("\n".join(predictions_text))


class CustomDialog(QDialog):
    def __init__(self, title, body):
        super().__init__()

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.StandardButton.Ok #| QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(body)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)