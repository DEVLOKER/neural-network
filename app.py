import sys
from PyQt6.QtWidgets import QApplication
from gui.MainWindow import MainWindow

sys.path.insert(0, '/algorithm')
sys.path.insert(0, '/gui')


# def fancy_generator():
#     my_list = [1, 2, 3]
#     for i in my_list:
#             yield i * 2

# mygen = fancy_generator()
# print(mygen)
# try:
#     while True:
#         val = next(mygen)
#         print(val)
# except StopIteration:
#     pass
# while True:
#     val = next(mygen)
#     if val is None:
#         break
#     print(val)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
