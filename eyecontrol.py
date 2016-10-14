from __future__ import division
import sys
import queue
import socket
import threading

import matplotlib
matplotlib.use("Qt4Agg")

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt

from PyQt4.QtCore import (Qt, QTimer,)
from PyQt4.QtGui import (QApplication, QMainWindow,
                         QWidget, QSlider, QPushButton, QLabel,
                         QVBoxLayout, QHBoxLayout)


class EyeControlApp(QMainWindow):

    pass


class EyeControlSocketThread(threading.Thread):

    pass


def main():

    app = QApplication(sys.argv)
    form = EyeControlApp()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
