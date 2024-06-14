import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
# sys.path.insert(0, '/algorithm')

from PyQt6.QtWidgets import QApplication
from MainWindow import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
