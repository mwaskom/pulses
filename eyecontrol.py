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

    ADDRESS_INFO = ("localhost", 50001)

    def __init__(self):

        super(EyeControlSocketThread, self).__init__()
        self.alive = threading.Event()
        self.alive.set()

    def join(self, timeout=None):

        self.alive.clear()
        threading.Thread.join(self, timeout)


class EyeControlClientThread(EyeControlSocketThread):

    def __init__(self, gaze_q, param_q):

        super(EyeControlClientThread, self).__init__()


class EyeControlServerThread(EyeControlSocketThread):

    def __init__(self, gaze_q, param_q):

        super(EyeControlServerThread, self).__init__()


def main():

    app = QApplication(sys.argv)
    form = EyeControlApp()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
