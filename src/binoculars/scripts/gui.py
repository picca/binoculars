import sys
import os
import json
import queue
import signal
import subprocess
import socket
import socketserver
import threading

import numpy
import matplotlib.figure
import matplotlib.image

from matplotlib.backends.backend_qt5agg import \
    (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.pyplot import rcParams

from PyQt5.Qt import (Qt)  # noqa
from PyQt5.QtCore import (QThread, pyqtSignal)
from PyQt5.QtGui import (QPainter)
from PyQt5.QtWidgets import \
    (QAction, QApplication, QStyle, QSlider, QMenuBar, QTabWidget,
     QFileDialog, QStatusBar, QMessageBox, QRadioButton,
     QButtonGroup, QCheckBox, QPushButton, QHBoxLayout,
     QVBoxLayout, QSplitter, QTableWidgetItem, QTableWidget,
     QLabel, QLineEdit, QStyleOptionSlider, QMainWindow, QWidget)

rcParams['image.cmap'] = 'jet'

import binoculars.main
import binoculars.space
import binoculars.plot
import binoculars.util

# RangeSlider is taken from
#    https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html


class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged
    """
    def __init__(self, *args):
        super().__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on
        # http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp

        painter = QPainter(self)
        style = QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't
            # get drawn on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle
                # QStyle.SC_SliderGroove
                # |
                # QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()

        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts

        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider,
                                                  opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = \
                    self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseReleaseEvent(self, _event):
        self.sliderReleased.emit()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos

        self.update()

        self.sliderMoved.emit(new_pos)

    def __pick(self, pt):
        if self.orientation() == Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt,
                                  style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt,
                                  style.SC_SliderHandle, self)

        if self.orientation() == Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(),
                                             self.maximum(),
                                             pos-slider_min,
                                             slider_max-slider_min,
                                             opt.upsideDown)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        menu_bar = QMenuBar()

        # Menu FILE

        f = menu_bar.addMenu("&File")

        newproject = QAction("New project", self)
        newproject.triggered.connect(self.newproject)
        f.addAction(newproject)

        loadproject = QAction("Open project", self)
        loadproject.triggered.connect(self.loadproject)
        f.addAction(loadproject)

        saveproject = QAction("Save project", self)
        saveproject.triggered.connect(self.saveproject)
        f.addAction(saveproject)

        addspace = QAction("Import space", self)
        addspace.triggered.connect(self.add_to_project)
        f.addAction(addspace)

        savespace = QAction("Export space", self)
        savespace.triggered.connect(self.exportspace)
        f.addAction(savespace)

        # Menu EDIT

        edit = menu_bar.addMenu("&Edit")

        merge = QAction("Merge Spaces", self)
        merge.triggered.connect(self.merge)
        edit.addAction(merge)

        subtract = QAction("Subtract a Space", self)
        subtract.triggered.connect(self.subtract)
        edit.addAction(subtract)

        # Menu Server

        serve = menu_bar.addMenu("&Serve")

        start_server = QAction("Start server queue", self)
        start_server.triggered.connect(lambda: self.open_server(startq=True))
        serve.addAction(start_server)

        stop_server = QAction("Stop server queue", self)
        stop_server.triggered.connect(self.kill_server)
        serve.addAction(stop_server)

        recieve = QAction("Open for spaces", self)
        recieve.triggered.connect(lambda: self.open_server(startq=False))
        serve.addAction(recieve)



        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.tab_widget.removeTab)

        self.statusbar = QStatusBar()

        self.setCentralWidget(self.tab_widget)
        self.setMenuBar(menu_bar)
        self.setStatusBar(self.statusbar)

        self.threads = []
        self.pro = None

    def closeEvent(self, event):
        self.kill_subprocess()
        super().closeEvent(event)

    def newproject(self):
        widget = ProjectWidget([], parent=self)
        self.tab_widget.addTab(widget, 'New Project')
        self.tab_widget.setCurrentWidget(widget)

    def loadproject(self, filename=None):
        if not filename:
            dialog = QFileDialog(self, "Load project")
            dialog.setNameFilters(['binoculars project file (*.proj)'])
            dialog.setFileMode(QFileDialog.ExistingFiles)
            dialog.setAcceptMode(QFileDialog.AcceptOpen)
            if not dialog.exec_():
                return
            fname = dialog.selectedFiles()
            if not fname:
                return
            for name in fname:
                try:
                    widget = ProjectWidget.fromfile(str(name), parent=self)
                    self.tab_widget.addTab(widget, short_filename(str(name)))
                    self.tab_widget.setCurrentWidget(widget)
                except Exception as e:
                    QMessageBox.critical(
                        self, 'Load project',
                        f"Unable to load project from {fname}: {e}")
        else:
            widget = ProjectWidget.fromfile(filename, parent=self)
            self.tab_widget.addTab(widget, short_filename(filename))

    def saveproject(self):
        widget = self.tab_widget.currentWidget()
        dialog = QFileDialog(self, "Save project")
        dialog.setNameFilters(['binoculars project file (*.proj)'])
        dialog.setDefaultSuffix('proj')
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            index = self.tab_widget.currentIndex()
            self.tab_widget.setTabText(index, short_filename(fname))
            widget.tofile(fname)
        except Exception as e:
            QMessageBox.critical(self, 'Save project',
                                 f"Unable to save project to {fname}: {e}")

    def add_to_project(self):
        if self.tab_widget.count() == 0:
            self.newproject()

        dialog = QFileDialog(self, "Import spaces")
        dialog.setNameFilters(['binoculars space file (*.hdf5)'])
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()
        if not fname:
            return
        for name in fname:
            widget = self.tab_widget.currentWidget()
            widget.addspace(str(name), True)

    def exportspace(self):
        widget = self.tab_widget.currentWidget()
        dialog = QFileDialog(self, "save mesh")
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            # FP useless ? _index = self.tab_widget.currentIndex()
            widget.space_to_file(str(fname))
        except Exception as e:
            QMessageBox.critical(self, 'export fitdata',
                                 f"Unable to save mesh to {fname}: {e}")

    def merge(self):
        widget = self.tab_widget.currentWidget()
        dialog = QFileDialog(self, "save mesh")
        dialog.setNameFilters(['binoculars space file (*.hdf5)'])
        dialog.setDefaultSuffix('hdf5')
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            # FP useless ? _index = self.tab_widget.currentIndex()
            widget.merge(str(fname))
        except Exception as e:
            QMessageBox.critical(self, 'merge',
                                 f"Unable to save mesh to {fname}: {e}")

    def subtract(self):
        dialog = QFileDialog(self, "subtract space")
        dialog.setNameFilters(['binoculars space file (*.hdf5)'])
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()
        if not fname:
            return
        for name in fname:
            try:
                widget = self.tab_widget.currentWidget()
                widget.subtractspace(str(name))
            except Exception as e:
                QMessageBox.critical(self, 'Import spaces',
                                     f"Unable to import space {fname}: {e}")

    def open_server(self, startq=True):
        if len(self.threads) != 0:
            print('Server already running')
        else:
            HOST, PORT = socket.gethostbyname(socket.gethostname()), 0

            self.q = queue.Queue()
            server = ThreadedTCPServer((HOST, PORT), SpaceTCPHandler)
            server.q = self.q

            self.ip, self.port = server.server_address

            if startq:
                cmd = ['python',
                       os.path.join(os.path.dirname(__file__),
                                    'binoculars-server.py'),
                       str(self.ip), str(self.port)]
                self.pro = subprocess.Popen(cmd, stdin=None,
                                            stdout=None, stderr=None,
                                            preexec_fn=os.setsid)

            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            updater = UpdateThread()
            updater.data_found.connect(self.update)
            updater.q = self.q
            self.threads.append(updater)
            updater.start()

            if not startq:
                print("GUI server started running at ip "
                      f"{self.ip} and port {self.port}.")

    def kill_server(self):
        if len(self.threads) == 0:
            print('No server running.')
        else:
            self.threads = []
            self.kill_subprocess()
            self.pro = None

    def kill_subprocess(self):
        if self.pro is not None:
            os.killpg(self.pro.pid, signal.SIGTERM)

    def update(self):
        names = []
        for tab in range(self.tab_widget.count()):
            names.append(self.tab_widget.tabText(tab))

        if 'server' not in names:
            widget = ProjectWidget([], parent=self)
            self.tab_widget.addTab(widget, 'server')
            names.append('server')

        index = names.index('server')
        serverwidget = self.tab_widget.widget(index)

        while not self.threads[0].fq.empty():
            command, space = self.threads[0].fq.get()
            serverwidget.table.addfromserver(command, space)
            serverwidget.table.select()
            if serverwidget.auto_update.isChecked():
                serverwidget.limitwidget.refresh()


class UpdateThread(QThread):
    fq = queue.Queue()
    data_found = pyqtSignal(object)

    def run(self):
        delay = binoculars.util.loop_delayer(1)
        jobs = []
        labels = []
        while 1:
            if not self.q.empty():
                command, space = self.q.get()
                if command in labels:
                    jobs[labels.index(command)].append(space)
                else:
                    jobs.append([space])
                    labels.append(command)
            elif self.q.empty() and len(jobs) > 0:
                self.fq.put((labels.pop(), binoculars.space.sum(jobs.pop())))
                self.data_found.emit('data found')
            else:
                next(delay)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class SpaceTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        command, config, metadata, axes, photons, contributions = \
            binoculars.util.socket_recieve(self)
        space = binoculars.space.Space(binoculars.space.Axes.fromarray(axes))
        space.config = binoculars.util.ConfigFile.fromserial(config)
        space.config.command = command
        space.config.origin = 'server'
        space.metadata = binoculars.util.MetaData.fromserial(metadata)
        space.photons = photons
        space.contributions = contributions
        self.server.q.put((command, space))


class HiddenToolbar(NavigationToolbar2QT):
    def __init__(self, show_coords, update_sliders, canvas):
        NavigationToolbar2QT.__init__(self, canvas, None)
        self.show_coords = show_coords
        self.update_sliders = update_sliders
        self.zoom()

        self.threed = False

    def mouse_move(self, event):
        if not self.threed:
            self.show_coords(event)

    def press_zoom(self, event):
        super().press_zoom(event)
        if not self.threed:
            self.inaxes = event.inaxes

    def release_zoom(self, event):
        super().release_zoom(event)
        if not self.threed:
            self.update_sliders(self.inaxes)


class ProjectWidget(QWidget):
    def __init__(self, filelist, key=None, projection=None, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar(self.show_coords,
                                     self.update_sliders, self.canvas)

        self.lin = QRadioButton('lin', self)
        self.lin.setChecked(False)
        self.lin.toggled.connect(self.plot)

        self.log = QRadioButton('log', self)
        self.log.setChecked(True)
        self.log.toggled.connect(self.plot)

        self.loglog = QRadioButton('loglog', self)
        self.loglog.setChecked(False)
        self.loglog.toggled.connect(self.plot)

        self.loggroup = QButtonGroup(self)
        self.loggroup.addButton(self.lin)
        self.loggroup.addButton(self.log)
        self.loggroup.addButton(self.loglog)

        self.swap_axes = QCheckBox('ax', self)
        self.swap_axes.setChecked(False)
        self.swap_axes.stateChanged.connect(self.plot)

        self.samerange = QCheckBox('same', self)
        self.samerange.setChecked(False)
        self.samerange.stateChanged.connect(self.update_colorbar)

        self.legend = QCheckBox('legend', self)
        self.legend.setChecked(True)
        self.legend.stateChanged.connect(self.plot)

        self.threed = QCheckBox('3d', self)
        self.threed.setChecked(False)
        self.threed.stateChanged.connect(self.plot)

        self.auto_update = QCheckBox('auto', self)
        self.auto_update.setChecked(True)

        self.datarange = RangeSlider(Qt.Horizontal)
        self.datarange.setMinimum(0)
        self.datarange.setMaximum(250)
        self.datarange.setLow(0)
        self.datarange.setHigh(self.datarange.maximum())
        self.datarange.setTickPosition(QSlider.TicksBelow)
        self.datarange.sliderMoved.connect(self.update_colorbar)

        self.table = TableWidget(filelist)
        self.table.selectionError.connect(self.selectionerror)
        self.table.plotaxesChanged.connect(self.plotaxes_changed)

        self.key = key
        self.projection = projection

        self.button_save = QPushButton('save image')
        self.button_save.clicked.connect(self.save)

        self.button_refresh = QPushButton('refresh')
        self.button_refresh.clicked.connect(self.table.select)

        self.limitwidget = LimitWidget(self.table.plotaxes)
        self.limitwidget.keydict.connect(self.update_key)
        self.limitwidget.rangechange.connect(self.update_figure_range)

        self.initUI()

        self.table.select()

    def initUI(self):
        self.control_widget = QWidget(self)
        hbox = QHBoxLayout()
        left = QVBoxLayout()

        pushbox = QHBoxLayout()
        pushbox.addWidget(self.button_save)
        pushbox.addWidget(self.button_refresh)
        left.addLayout(pushbox)

        radiobox = QHBoxLayout()
        self.group = QButtonGroup(self)
        for label in ['stack', 'grid']:
            rb = QRadioButton(label, self.control_widget)
            rb.setChecked(True)
            self.group.addButton(rb)
            radiobox.addWidget(rb)

        radiobox.addWidget(self.lin)
        radiobox.addWidget(self.log)
        radiobox.addWidget(self.loglog)

        datarangebox = QHBoxLayout()
        datarangebox.addWidget(self.samerange)
        datarangebox.addWidget(self.legend)
        datarangebox.addWidget(self.threed)
        datarangebox.addWidget(self.swap_axes)
        datarangebox.addWidget(self.auto_update)

        left.addLayout(radiobox)
        left.addLayout(datarangebox)
        left.addWidget(self.datarange)

        left.addWidget(self.table)
        left.addWidget(self.limitwidget)
        self.control_widget.setLayout(left)

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.control_widget)
        splitter.addWidget(self.canvas)

        hbox.addWidget(splitter)
        self.setLayout(hbox)

    def show_coords(self, event):
        plotaxes = event.inaxes
        if hasattr(plotaxes, 'space'):
            if plotaxes.space.dimension == 2:
                labels = numpy.array([plotaxes.get_xlabel(),
                                      plotaxes.get_ylabel()])
                order = [plotaxes.space.axes.index(label) for label in labels]
                labels = labels[order]
                coords = numpy.array([event.xdata, event.ydata])[order]
                try:
                    rounded_coords = \
                        [ax[ax.get_index(coord)] for ax,
                         coord in zip(plotaxes.space.axes, coords)]
                    intensity = f'{plotaxes.space[list(coords)]:.2e}'
                    self.parent.statusbar.showMessage(
                        f"{labels[0]} = {rounded_coords[0]}, "
                        f"{labels[1]} = {rounded_coords[1]}, "
                        f"Intensity = {intensity}")
                except ValueError:
                    self.parent.statusbar.showMessage('out of range')
            elif plotaxes.space.dimension == 1:
                xlabel = plotaxes.get_xlabel()
                xaxis = plotaxes.space.axes[plotaxes.space.axes.index(xlabel)]
                if event.xdata in xaxis:
                    xcoord = xaxis[xaxis.get_index(event.xdata)]
                    intensity = f'{event.ydata:.2e}'
                    self.parent.statusbar.showMessage(
                        f"{xaxis.label} = {xcoord}, "
                        f"Intensity = {intensity}")

    def update_sliders(self, plotaxes):
        if plotaxes is not None:
            if hasattr(plotaxes, 'space'):
                space = plotaxes.space
                if space.dimension == 2:
                    labels = numpy.array([plotaxes.get_xlabel(),
                                          plotaxes.get_ylabel()])
                    limits = list(lim for lim in [plotaxes.get_xlim(),
                                                  plotaxes.get_ylim()])
                elif space.dimension == 1:
                    labels = [plotaxes.get_xlabel()]
                    limits = [plotaxes.get_xlim()]
                keydict = dict()
                for key, value in zip(labels, limits):
                    keydict[key] = value
                self.limitwidget.update_from_zoom(keydict)

    def selectionerror(self, message):
        self.limitwidget.setDisabled(True)
        self.errormessage(message)

    def plotaxes_changed(self, plotaxes):
        self.limitwidget.setEnabled(True)
        self.limitwidget.axes_update(plotaxes)

    def update_key(self, input):
        self.key = input['key']
        self.projection = input['project']

        if len(self.limitwidget.sliders) - len(self.projection) == 1:
            self.datarange.setDisabled(True)
            self.samerange.setDisabled(True)
            self.swap_axes.setDisabled(True)
            self.loglog.setEnabled(True)
        elif len(self.limitwidget.sliders) - len(self.projection) == 2:
            self.loglog.setDisabled(True)
            self.datarange.setEnabled(True)
            self.samerange.setEnabled(True)
            self.swap_axes.setEnabled(True)
        self.plot()

    def get_norm(self, mi, ma):
        log = self.log.isChecked()

        rangemin = self.datarange.low() * 1.0 / self.datarange.maximum()
        rangemax = self.datarange.high() * 1.0 / self.datarange.maximum()

        if log:
            power = 3
            vmin = mi + (ma - mi) * rangemin ** power
            vmax = mi + (ma - mi) * rangemax ** power
        else:
            vmin = mi + (ma - mi) * rangemin
            vmax = mi + (ma - mi) * rangemax

        if log:
            return matplotlib.colors.LogNorm(vmin, vmax)
        else:
            return matplotlib.colors.Normalize(vmin, vmax)

    def get_normlist(self):
        # FP useless ? _log = self.log.isChecked()
        same = self.samerange.checkState()

        if same:
            return [self.get_norm(min(self.datamin),
                                  max(self.datamax))] * len(self.datamin)
        else:
            norm = []
            for i in range(len(self.datamin)):
                norm.append(self.get_norm(self.datamin[i], self.datamax[i]))
            return norm

    def plot(self):
        if len(self.table.plotaxes) == 0:
            return
        self.figure.clear()
        self.parent.statusbar.clearMessage()

        self.figure_images = []
        log = self.log.isChecked()
        loglog = self.loglog.isChecked()

        plotcount = len(self.table.selection)
        plotcolumns = int(numpy.ceil(numpy.sqrt(plotcount)))
        plotrows = int(numpy.ceil(float(plotcount) / plotcolumns))
        plotoption = None
        if self.group.checkedButton():
            plotoption = self.group.checkedButton().text()

        spaces = []

        for i, filename in enumerate(self.table.selection):
            axes = self.table.getax(filename)
            rkey = axes.restricted_key(self.key)
            if rkey is None:
                space = self.table.getspace(filename)
            else:
                try:
                    space = self.table.getspace(filename, rkey)
                except KeyError:
                    return
            projection = [ax for ax in self.projection if ax in space.axes]
            if projection:
                space = space.project(*projection)
            dimension = space.dimension
            if dimension == 0:
                self.errormessage('Choose suitable number of projections')
            if dimension == 3 and not self.threed.isChecked():
                self.errormessage(
                    'Switch on 3D plotting, only works with small spaces')
            spaces.append(space)

        self.datamin = []
        self.datamax = []
        for space in spaces:
            data = space.get_masked().compressed()
            if log or loglog:
                data = data[data > 0]
            if len(data) > 0:
                self.datamin.append(data.min())
                self.datamax.append(data.max())
            else:  # min, max when there is no data to plot
                self.datamin.append(numpy.pi)
                self.datamax.append(numpy.pi)

        norm = self.get_normlist()

        if dimension == 1 or dimension == 2:
            self.toolbar.threed = False
        else:
            self.toolbar.threed = True

        for i, space in enumerate(spaces):
            filename = self.table.selection[i]
            basename = os.path.splitext(os.path.basename(filename))[0]
            if plotcount > 1:
                if dimension == 1 and (plotoption == 'stack' or
                                       plotoption is None):
                    self.ax = self.figure.add_subplot(111)
                if dimension == 2 and plotoption != 'grid':
                    sys.stderr.write(
                        'warning: stack display not supported'
                        ' for multi-file-plotting, falling back to grid\n')
                    plotoption = 'grid'
                elif dimension > 3:
                    sys.stderr.write(
                        'error: cannot display 4 or higher dimensional data,'
                        ' use --project or --slice '
                        'to decrease dimensionality\n')
                    sys.exit(1)

            if plotoption == 'grid':
                if dimension == 1 or dimension == 2:
                    self.ax = self.figure.add_subplot(plotrows,
                                                      plotcolumns, i+1)
                elif self.threed.isChecked():
                    self.ax = self.figure.gca(projection='3d')
                self.ax.set_title(basename)
            else:
                self.ax = self.figure.add_subplot(111)

            if dimension == 2 and self.swap_axes.checkState():
                space = space.reorder(list(ax.label for ax in
                                           space.axes)[::-1])

            self.ax.space = space
            im = binoculars.plot.plot(space, self.figure, self.ax,
                                      log=log, loglog=loglog,
                                      label=basename, norm=norm[i])

            self.figure_images.append(im)

        if dimension == 1 and self.legend.checkState():
            self.ax.legend()

        self.update_figure_range(self.key_to_str(self.key))
        self.canvas.draw()

    def merge(self, filename):
        try:
            spaces = tuple(self.table.getspace(selected_filename)
                           for selected_filename in self.table.selection)
            newspace = \
                binoculars.space.sum(binoculars.space.make_compatible(spaces))
            newspace.tofile(filename)
            list(map(self.table.remove, self.table.selection))
            self.table.add_space(filename, True)
        except Exception as e:
            QMessageBox.critical(self, 'Merge',
                                 f"Unable to merge the meshes. {e}")

    def subtractspace(self, filename):
        try:
            subtractspace = binoculars.space.Space.fromfile(filename)
            spaces = tuple(self.table.getspace(selected_filename)
                           for selected_filename in self.table.selection)
            newspaces = tuple(space - subtractspace for space in spaces)
            for space, selected_filename in zip(newspaces,
                                                self.table.selection):
                newfilename = \
                    binoculars.util.find_unused_filename(selected_filename)
                space.tofile(newfilename)
                self.table.remove(selected_filename)
                self.table.add_space(newfilename, True)
        except Exception as e:
            QMessageBox.critical(self, 'Subtract',
                                 f"Unable to subtract the meshes. {e}")

    def errormessage(self, message):
        self.figure.clear()
        self.canvas.draw()
        self.parent.statusbar.showMessage(message)

    def update_figure_range(self, key):
        if len(key) == 0:
            return
        for ax in self.figure.axes:
            plotaxes = self.table.plotaxes
            xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
            if xlabel in plotaxes:
                xindex = plotaxes.index(xlabel)
                ax.set_xlim(key[xindex][0], key[xindex][1])
            if ylabel in plotaxes:
                yindex = plotaxes.index(ylabel)
                ax.set_ylim(key[yindex][0], key[yindex][1])
        self.canvas.draw()

    def update_colorbar(self, value):
        normlist = self.get_normlist()
        for im, norm in zip(self.figure_images, normlist):
            im.set_norm(norm)
        self.canvas.draw()

    @staticmethod
    def key_to_str(key):
        return list([s.start, s.stop] for s in key)

    @staticmethod
    def str_to_key(s):
        return tuple(slice(float(key[0]), float(key[1])) for key in s)

    def tofile(self, filename=None):
        dict = {}
        dict['filelist'] = self.table.filelist
        dict['key'] = self.key_to_str(self.key)
        dict['projection'] = self.projection

        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(self,
                                                      'Save Project', '.')

        with open(filename, 'w') as fp:
            json.dump(dict, fp)

    @classmethod
    def fromfile(cls, filename=None, parent=None):
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(cls,
                                                      'Open Project',
                                                      '.',
                                                      '*.proj')
        try:
            with open(filename) as fp:
                dict = json.load(fp)
        except OSError as e:
            raise cls.error.showMessage(
                f"unable to open '{filename}' as project file "
                f"(original error: {e:1!r})")

        newlist = []
        for fn in dict['filelist']:
            if not os.path.exists(fn):
                warningbox = \
                    QMessageBox(2, 'Warning',
                                f'Cannot find space at path {fn};'
                                ' locate proper space',
                                buttons=QMessageBox.Open)
                warningbox.exec_()
                newname, _ = QFileDialog.getOpenFileName(caption='Open space {fn}',
                                                         directory='.',
                                                         filter='*.hdf5')
                newlist.append(newname)
            else:
                newlist.append(fn)

        widget = cls(newlist, cls.str_to_key(dict['key']),
                     dict['projection'], parent=parent)

        return widget

    def addspace(self, filename=None, add=False):
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      'Open Project',
                                                      '.',
                                                      '*.hdf5')
            print(filename)
        self.table.add_space(filename, add)

    def save(self):
        dialog = QFileDialog(self, "Save image")
        dialog.setNameFilters(['Portable Network Graphics (*.png)',
                               'Portable Document Format (*.pdf)'])
        dialog.setDefaultSuffix('png')
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            self.figure.savefig(str(fname))
        except Exception as e:
            QMessageBox.critical(self, 'Save image',
                                 f"Unable to save image to {fname}: {e}")

    def space_to_file(self, fname):
        ext = os.path.splitext(fname)[-1]

        for i, filename in enumerate(self.table.selection):
            axes = self.table.getax(filename)
            space = self.table.getspace(filename,
                                        key=axes.restricted_key(self.key))
            projection = [ax for ax in self.projection if ax in space.axes]
            if projection:
                space = space.project(*projection)

            space.trim()
            outfile = binoculars.util.find_unused_filename(fname)

            if ext == '.edf':
                binoculars.util.space_to_edf(space, outfile)
                self.parent.statusbar.showMessage(f"saved at {outfile}")
            elif ext == '.hdf5':
                space.tofile(outfile)
                self.parent.statusbar.showMessage(f"saved at {outfile}")
            elif ext == '.npy':
                binoculars.util.space_to_npy(space, outfile)
                self.parent.statusbar.showMessage(f"saved at {outfile}")
            elif ext == '.txt':
                binoculars.util.space_to_txt(space, outfile)
                self.parent.statusbar.showMessage(f"saved at {outfile}")
            else:
                self.parent.statusbar.showMessage(
                    f"unknown extension {ext}, unable to save!\n")


def short_filename(filename):
    return filename.split('/')[-1].split('.')[0]


class SpaceContainer(QTableWidgetItem):
    def __init__(self, label, space=None):
        super().__init__(short_filename(label))
        self.label = label
        self.space = space

    def get_space(self, key=None):
        if self.space is None:
            return binoculars.space.Space.fromfile(self.label, key=key)
        else:
            if key is None:
                key = Ellipsis
            return self.space[key]

    def get_ax(self):
        if self.space is None:
            return binoculars.space.Axes.fromfile(self.label)
        else:
            return self.space.axes

    def add_to_space(self, space):
        if self.space is None:
            newspace = binoculars.space.Space.fromfile(self.label) + space
            newspace.tofile(self.label)
        else:
            self.space += space


class TableWidget(QWidget):
    selectionError = pyqtSignal(str, name='Selection Error')
    plotaxesChanged = pyqtSignal(binoculars.space.Axes,
                                 name='plot axes changed')

    def __init__(self, filelist=[], parent=None):
        super().__init__(parent)

        hbox = QHBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', 'filename',
                                              'labels', 'remove'])

        for index, width in enumerate([25, 150, 50, 70]):
            self.table.setColumnWidth(index, width)

        for filename in filelist:
            self.add_space(filename)

        hbox.addWidget(self.table)
        self.setLayout(hbox)

    def add_space(self, filename, add=True, space=None):
        index = self.table.rowCount()
        self.table.insertRow(index)

        checkboxwidget = QCheckBox()
        checkboxwidget.setChecked(add)
        checkboxwidget.clicked.connect(self.select)
        self.table.setCellWidget(index, 0, checkboxwidget)

        container = SpaceContainer(filename, space)
        self.table.setItem(index, 1, container)

        item = QTableWidgetItem(','.join(list(ax.label.lower()
                                              for ax in container.get_ax())))
        self.table.setItem(index, 2, item)

        buttonwidget = QPushButton('remove')
        buttonwidget.clicked.connect(lambda: self.remove(filename))
        self.table.setCellWidget(index, 3, buttonwidget)
        if add:
            self.select()

    def addfromserver(self, command, space):
        if command not in self.filelist:
            self.add_space(command, add=False, space=space)
        else:
            container = self.table.item(self.filelist.index(command), 1)
            container.add_to_space(space)

    def remove(self, filename):
        self.table.removeRow(self.filelist.index(filename))
        self.select()
        print(f'removed: {filename}')

    def select(self):
        axes = self.plotaxes
        if len(axes) > 0:
            self.plotaxesChanged.emit(axes)
        else:
            self.selectionError.emit(
                'no spaces selected '
                'or spaces with non identical labels selected')

    @property
    def selection(self):
        return list(container.label
                    for checkbox, container
                    in zip(self.itercheckbox(), self.itercontainer())
                    if checkbox.checkState())

    @property
    def plotaxes(self):
        axes = tuple(container.get_ax()
                     for checkbox, container
                     in zip(self.itercheckbox(), self.itercontainer())
                     if checkbox.checkState())
        if len(axes) > 0:
            try:
                return binoculars.space.Axes(
                    binoculars.space.union_unequal_axes(ax)
                    for ax in zip(*axes))
            except ValueError:
                return ()
        else:
            return ()

    @property
    def filelist(self):
        return list(container.label for container in self.itercontainer())

    def getax(self, filename):
        index = self.filelist.index(filename)
        return self.table.item(index, 1).get_ax()

    def getspace(self, filename, key=None):
        index = self.filelist.index(filename)
        return self.table.item(index, 1).get_space(key)

    def itercheckbox(self):
        return iter(self.table.cellWidget(index, 0)
                    for index in range(self.table.rowCount()))

    def itercontainer(self):
        return iter(self.table.item(index, 1)
                    for index in range(self.table.rowCount()))


class LimitWidget(QWidget):
    keydict = pyqtSignal(dict, name="keydict")
    rangechange = pyqtSignal(list, name="rangechange")

    def __init__(self, axes, parent=None) -> None:
        super().__init__(parent)
        self.initUI(axes)

    def initUI(self, axes) -> None:
        self.axes = axes

        self.sliders = list()
        self.qlabels = list()
        self.leftindicator = list()
        self.rightindicator = list()

        labels = list(ax.label for ax in axes)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        self.projectionlabel = QLabel(self)
        self.projectionlabel.setText('projection along axis')
        self.refreshbutton = QPushButton('all')
        self.refreshbutton.clicked.connect(self.refresh)

        vbox.addWidget(self.projectionlabel)

        self.checkbox = list()
        self.state = list()

        for label in labels:
            self.checkbox.append(QCheckBox(label, self))
        for box in self.checkbox:
            self.state.append(box.checkState())
            hbox.addWidget(box)
            box.stateChanged.connect(self.update_checkbox)

        self.state = numpy.array(self.state, dtype=numpy.bool_)
        self.init_checkbox()

        vbox.addLayout(hbox)

        for label in labels:
            self.qlabels.append(QLabel(self))
            self.leftindicator.append(QLineEdit(self))
            self.rightindicator.append(QLineEdit(self))
            self.sliders.append(RangeSlider(Qt.Horizontal))

        for index, label in enumerate(labels):
            box = QHBoxLayout()
            box.addWidget(self.qlabels[index])
            box.addWidget(self.leftindicator[index])
            box.addWidget(self.sliders[index])
            box.addWidget(self.rightindicator[index])
            vbox.addLayout(box)

        for left in self.leftindicator:
            left.setMaximumWidth(50)
        for right in self.rightindicator:
            right.setMaximumWidth(50)

        for index, label in enumerate(labels):
            self.qlabels[index].setText(label)

        for index, ax in enumerate(axes):
            self.sliders[index].setMinimum(0)
            self.sliders[index].setMaximum(len(ax) - 1)
            self.sliders[index].setLow(0)
            self.sliders[index].setHigh(len(ax) - 1)
            self.sliders[index].setTickPosition(QSlider.TicksBelow)

        self.update_lines()

        for slider in self.sliders:
            slider.sliderMoved.connect(self.update_lines)
        for slider in self.sliders:
            slider.sliderReleased.connect(self.send_signal)

        for line in self.leftindicator:
            line.editingFinished.connect(self.update_sliders_left)
            line.editingFinished.connect(self.send_signal)
        for line in self.rightindicator:
            line.editingFinished.connect(self.update_sliders_right)
            line.editingFinished.connect(self.send_signal)

        vbox.addWidget(self.refreshbutton)

        if self.layout() is None:
            self.setLayout(vbox)

    def refresh(self) -> None:
        for slider in self.sliders:
            slider.setLow(slider.minimum())
            slider.setHigh(slider.maximum())

        self.update_lines()
        self.send_signal()

    def update_lines(self, value=0) -> None:
        for index, slider in enumerate(self.sliders):
            self.leftindicator[index].setText(
                str(self.axes[index][slider.low()]))
            self.rightindicator[index].setText(
                str(self.axes[index][slider.high()]))
        key = list((float(str(left.text())), float(str(right.text())))
                   for left, right
                   in zip(self.leftindicator, self.rightindicator))
        self.rangechange.emit(key)

    def send_signal(self) -> None:
        signal = {}
        key = ((float(str(left.text())), float(str(right.text())))
               for left, right
               in zip(self.leftindicator, self.rightindicator))
        key = [left if left == right else slice(left, right, None)
               for left, right in key]
        project = []
        for ax, state in zip(self.axes, self.state):
            if state:
                project.append(ax.label)
        signal['project'] = project
        signal['key'] = key
        self.keydict.emit(signal)

    def update_sliders_left(self) -> None:
        for ax, left, right, slider in zip(self.axes,
                                           self.leftindicator,
                                           self.rightindicator,
                                           self.sliders):
            try:
                leftvalue = ax.get_index(float(str(left.text())))
                rightvalue = ax.get_index(float(str(right.text())))
                if leftvalue >= slider.minimum() and leftvalue < rightvalue:
                    slider.setLow(leftvalue)
                else:
                    slider.setLow(rightvalue - 1)
            except ValueError:
                slider.setLow(0)
            left.setText(str(ax[slider.low()]))

    def update_sliders_right(self) -> None:
        for ax, left, right, slider in zip(self.axes,
                                           self.leftindicator,
                                           self.rightindicator,
                                           self.sliders):
            leftvalue = ax.get_index(float(str(left.text())))
            try:
                rightvalue = ax.get_index(float(str(right.text())))
                if rightvalue <= slider.maximum() and rightvalue > leftvalue:
                    slider.setHigh(rightvalue)
                else:
                    slider.setHigh(leftvalue + 1)
            except ValueError:
                slider.setHigh(len(ax) - 1)
            right.setText(str(ax[slider.high()]))

    def update_checkbox(self) -> None:
        self.state = list()
        for box in self.checkbox:
            self.state.append(box.checkState())
        self.send_signal()

    def init_checkbox(self) -> None:
        while len(self.state) - self.state.sum() > 2:
            # FP useless ? _index = numpy.where(self.state is False)[-1]
            self.state[-1] = True
        for box, state in zip(self.checkbox, self.state):
            box.setChecked(bool(state))

    def axes_update(self, axes) -> None:
        if ({ax.label for ax in self.axes} != {ax.label for ax in axes}):
            QWidget().setLayout(self.layout())
            self.initUI(axes)
            self.send_signal()
        else:
            low = tuple(self.axes[index][slider.low()]
                        for index, slider
                        in enumerate(self.sliders))
            high = tuple(self.axes[index][slider.high()]
                         for index, slider
                         in enumerate(self.sliders))

            for index, ax in enumerate(axes):
                self.sliders[index].setMinimum(0)
                self.sliders[index].setMaximum(len(ax) - 1)

            self.axes = axes

            for index, slider in enumerate(self.sliders):
                self.leftindicator[index].setText(str(low[index]))
                self.rightindicator[index].setText(str(high[index]))

            self.update_sliders_left()
            self.update_sliders_right()

            self.send_signal()

    def update_from_zoom(self, keydict) -> None:
        for key in keydict:
            index = self.axes.index(key)
            self.leftindicator[index].setText(str(keydict[key][0]))
            self.rightindicator[index].setText(str(keydict[key][1]))
        self.update_sliders_left()
        self.update_sliders_right()
        self.send_signal()


def is_empty(keys) -> bool:
    for k in keys:
        if isinstance(k, slice):
            if k.start == k.stop:
                return True
    return False


def main():
    app = QApplication(sys.argv)

    binoculars.space.silence_numpy_errors()

    main = Window()
    main.resize(1000, 600)
    main.newproject()
    main.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
