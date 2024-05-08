import sys
from PyQt6.QtWidgets import QApplication
from gui.MainWindow import MainWindow

sys.path.insert(0, '/algorithm')
sys.path.insert(0, '/gui')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
