import sys
from PyQt6.QtWidgets import QApplication
from algorithm.DigitRecognizer import DigitRecognizer
from gui.MainWindow import MainWindow

sys.path.insert(0, '/algorithm')
sys.path.insert(0, '/gui')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # digit_recognizer = DigitRecognizer()
    # digit_recognizer.load_model()
    window = MainWindow()#digit_recognizer)
    window.show()
    sys.exit(app.exec())
