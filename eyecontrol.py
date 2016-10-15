from __future__ import division
import sys
import Queue
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

        self.poll_dur = 50
        self.gaze_data = np.zeros((10, 2)) * np.nan
        self.axes_background = None

        self.gaze_q = Queue.Queue()
        self.param_q = Queue.Queue()

        # TODO read this from a params file
        self.current_params = dict(fix_radius=2, x_offset=0, y_offset=0)

        self.create_main_frame()
        self.create_plot_objects()
        self.create_client()
        self.create_timers()

    def update_plot(self):

        # Try to open the socket, if it isn't open
        if self.client is None:
            self.create_client()

        # Read a new datapoint from the queue
        new_data = []
        while True:
            try:
                data = self.gaze_q.get(block=False)
                new_point = np.fromstring(data)
                new_data.append(new_point)
            except Queue.Empty:
                if not new_data:
                    new_data.append(np.array([np.nan, np.nan]))
                break

        # Update the data array
        new_data = np.vstack(new_data)
        if len(new_data) > 10:
            self.gaze_data = new_data[-10:]
        else:
            n = len(new_data)
            self.gaze_data[:-n] = self.gaze_data[n:]
            self.gaze_data[-n:] = new_data

        # Capture the static background
        # This is done here because the figure gets resized at some point
        # during app/frame setup and it messes up the background capture
        if self.axes_background is None:
            self.fig.canvas.draw()
            self.axes_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Update the dynamic plot objects
        self.gaze.set_offsets(self.gaze_data)
        self.fix_window.set_radius(self.fix_slider.value() / 10)

        # Re-draw the plot
        self.fig.canvas.restore_region(self.axes_background)
        self.ax.draw_artist(self.gaze)
        self.ax.draw_artist(self.fix_window)
        self.canvas.blit(self.ax.bbox)

    def update_fix_label(self):

        fix_value = self.fix_slider.value() / 10
        self.fix_label.setText("Fix window: {:.1f}".format(fix_value))
        self.fix_label.setStyleSheet("color: red")

    def update_x_label(self):

        x_value = self.x_slider.value() / 10
        self.x_label.setText("x offset: {:.1f}".format(x_value))
        self.x_label.setStyleSheet("color: red")

    def update_y_label(self):

        y_value = self.y_slider.value() / 10
        self.y_label.setText("y window: {:.1f}".format(y_value))
        self.y_label.setStyleSheet("color: red")

    def update_params(self):

        self.current_params["fix_label"] = self.fix_slider.value() / 10
        self.current_params["x_offset"] = self.x_slider.value() / 10
        self.current_params["y_offset"] = self.y_slider.value() / 10

        param_data = np.array([self.current_params["fix_label"],
                               self.current_params["x_offset"],
                               self.current_params["y_offset"]])
        self.param_q.put(param_data.tostring())

        self.fix_label.setStyleSheet("color: black")
        self.x_label.setStyleSheet("color: black")
        self.y_label.setStyleSheet("color: black")

    def reset_params(self):

        self.fix_slider.setValue(int(self.current_params["fix_radius"] * 10))
        self.x_slider.setValue(int(self.current_params["x_offset"] * 10))
        self.y_slider.setValue(int(self.current_params["y_offset"] * 10))

        self.fix_label.setStyleSheet("color: black")
        self.x_label.setStyleSheet("color: black")
        self.y_label.setStyleSheet("color: black")

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
        self.fix_slider.setValue(int(self.current_params["fix_radius"] * 10))
        self.fix_slider.setTickPosition(QSlider.TicksBelow)
        self.fix_slider.setTracking(True)
        self.fix_slider.valueChanged.connect(self.update_fix_label)

        self.x_label = QLabel("x offset: 0.0")
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(-40, 40)
        self.x_slider.setValue(int(self.current_params["x_offset"] * 10))
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.setTracking(True)
        self.x_slider.valueChanged.connect(self.update_x_label)

        self.y_label = QLabel("y offset: 0.0")
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(-40, 40)
        self.y_slider.setValue(int(self.current_params["y_offset"] * 10))
        self.y_slider.setTickPosition(QSlider.TicksBelow)
        self.y_slider.setTracking(True)
        self.y_slider.valueChanged.connect(self.update_y_label)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_params)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_params)

        # --- Layout

        controls = QHBoxLayout()

        for (l, w) in [(self.fix_label, self.fix_slider),
                       (self.x_label, self.x_slider),
                       (self.y_label, self.y_slider)]:

            vbox = QVBoxLayout()
            vbox.addWidget(l)
            vbox.addWidget(w)
            vbox.setAlignment(w, Qt.AlignVCenter)
            controls.addLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self.update_button)
        vbox.addWidget(self.reset_button)
        controls.addLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(controls)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_plot_objects(self):

        gaze_color = (.3, .45, .7)
        gaze_rgba = np.zeros((10, 4))
        gaze_rgba[:, :3] = gaze_color
        gaze_rgba[:, -1] = np.linspace(.1, 1, 10)

        # Gaze data points
        x, y = self.gaze_data.T
        self.gaze = self.ax.scatter(x, y,
                                    c=gaze_rgba, s=20,
                                    zorder=3,
                                    linewidth=0,
                                    animated=True)

        # Fixation window
        radius = self.current_params["fix_radius"]
        self.fix_window = plt.Circle((0, 0), radius,
                                     facecolor="none",
                                     linewidth=.8,
                                     edgecolor=".3",
                                     animated=True)
        self.ax.add_artist(self.fix_window)

        # Fixation point and saccade targets
        # TODO get from params file
        point_locs = np.array([(0, 0), (-8, 3), (8, 3)])
        point_rgba = [(.9, .8, .1, 1)]
        x, y = point_locs.T
        self.points = self.ax.scatter(x, y,
                                      c=point_rgba, s=50,
                                      linewidth=0)

    def create_client(self):

        try:
            self.client = EyeControlClientThread(self.gaze_q, self.param_q)
        except socket.error:
            self.client = None

        if self.client is not None:
            self.client.start()

    def create_timers(self):

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
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

        self.gaze_q = gaze_q
        self.param_q = param_q

        self.socket = socket.socket()
        self.socket.connect(self.ADDRESS_INFO)
        self.socket.settimeout(.005)

    def run(self):

        while self.alive.isSet():

            try:
                data = self.socket.recv(16)
            except socket.timeout:
                continue

            if data == "updateparameters":
                try:
                    new_params = self.param_q.get(block=False)
                    self.socket.sendall(new_params)

                except Queue.Empty:
                    pass

            else:
                self.gaze_q.put(data)


class EyeControlServerThread(EyeControlSocketThread):

    def __init__(self, gaze_q, param_q, cmd_q):

        super(EyeControlServerThread, self).__init__()

        self.gaze_q = gaze_q
        self.param_q = param_q
        self.cmd_q = cmd_q

        self.server = socket.socket()
        self.server.bind(("localhost", 50001))
        self.server.listen(2)

    def run(self):

        clientsocket, _ = self.server.accept()
        clientsocket.settimeout(.2)

        try:
            while self.alive.isSet():

                try:
                    _ = self.cmd_q.get(block=False)
                    clientsocket.sendall("updateparameters")
                    try:
                        new_params = clientsocket.recv(24)
                        new_params = np.fromstring(new_params)
                        self.param_q.put(new_params)
                    except socket.timeout:
                        pass
                except Queue.Empty:
                    pass

                try:
                    data = self.gaze_q.get(block=False)
                    data = np.array(data).tostring()
                    clientsocket.sendall(data)
                except Queue.Empty:
                    pass

        finally:
            self.server.close()

def main():

    app = QApplication(sys.argv)
    form = EyeControlApp()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
