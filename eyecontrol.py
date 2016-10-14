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

    def __init__(self, parent=None):

        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Eye Control")

        self.poll_dur = 100
        self.gaze_data = np.zeros((10, 2))
        self.axes_background = None

        self.gaze_q = queue.Queue()
        self.param_q = queue.Queue()

        self.create_main_frame()
        self.create_plot_objects()
        self.create_client()
        self.create_timers()

    def update_plot(self):

        pass

    def update_labels(self):

        fix_value = self.fix_slider.value() / 10
        self.fix_label.setText("Fix window: {:.1f}".format(fix_value))

        x_value = self.x_slider.value() / 10
        self.x_label.setText("x offset: {:.1f}".format(x_value))

        y_value = self.y_slider.value() / 10
        self.y_label.setText("y window: {:.1f}".format(y_value))

    def send_params(self):

        pass

    def create_main_frame(self):

        self.main_frame = QWidget()

        # --- Matplotlib figure

        self.dpi = 100
        self.fig = Figure((5, 5), dpi=self.dpi, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self.main_frame)

        self.ax = self.fig.add_subplot(111)
        self.ax.set(xlim=(-10, 10),
                    ylim=(-10, 10),
                    aspect="equal")
        ticks = np.linspace(-10, 10, 21)
        self.ax.set_xticks(ticks, minor=True)
        self.ax.set_yticks(ticks, minor=True)

        grid_kws = dict(which="minor", lw=.5, ls="-", c=".8")
        self.ax.xaxis.grid(True, **grid_kws)
        self.ax.yaxis.grid(True, **grid_kws)

        # --- Control GUI Elements

        self.fix_label = QLabel("Fix window: 2.0")
        self.fix_slider = QSlider(Qt.Horizontal)
        self.fix_slider.setRange(0, 50)
        self.fix_slider.setValue(25)
        self.fix_slider.setTickPosition(QSlider.TicksBelow)
        self.fix_slider.setTracking(True)
        self.fix_slider.valueChanged.connect(self.update_labels)

        self.x_label = QLabel("x offset: 0.0")
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(-40, 40)
        self.x_slider.setValue(0)
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.setTracking(True)
        self.x_slider.valueChanged.connect(self.update_labels)

        self.y_label = QLabel("y offset: 0.0")
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(-40, 40)
        self.y_slider.setValue(0)
        self.y_slider.setTickPosition(QSlider.TicksBelow)
        self.y_slider.setTracking(True)
        self.y_slider.valueChanged.connect(self.update_labels)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.send_params)

        # --- Control GUI Layout

        controls = QHBoxLayout()

        for (l, w) in [(self.fix_label, self.fix_slider),
                       (self.x_label, self.x_slider),
                       (self.y_label, self.y_slider)]:

            vbox = QVBoxLayout()
            vbox.addWidget(l)
            vbox.addWidget(w)
            vbox.setAlignment(w, Qt.AlignVCenter)
            controls.addLayout(vbox)

        controls.addWidget(self.update_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(controls)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_plot_objects(self):

        pass

    def create_client(self):

        pass

    def create_timers(self):

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(self.poll_dur)


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
