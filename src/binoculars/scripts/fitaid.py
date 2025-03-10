#!/usr/bin/env python3

# TODO: export des fit en ascii should be versionned.

from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Type, Union

import sys
import os.path
import itertools

import h5py
import matplotlib.figure
import matplotlib.image
import numpy

import binoculars.main
import binoculars.plot
import binoculars.util

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.pyplot import Rectangle
from numpy import ndarray
from numpy.ma import MaskedArray
from PyQt5.Qt import (Qt)
from PyQt5.QtCore import (pyqtSignal)
from PyQt5.QtWidgets import (
    QAction, QApplication, QSlider, QMenuBar, QTabWidget,
    QFileDialog, QStatusBar, QMessageBox, QRadioButton,
    QButtonGroup, QCheckBox, QPushButton, QHBoxLayout,
    QVBoxLayout, QSplitter, QTableWidgetItem, QTableWidget,
    QLabel, QLineEdit, QMainWindow, QWidget, QComboBox,
    QProgressDialog, QDoubleSpinBox)
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError

from binoculars.fit import PeakFitBase
from binoculars.space import Axes, Axis, Space, get_axis_values, get_bins

class FitData:

    def __init__(self, filename: str, parent: QWidget) -> None:
        self.filename = filename
        self.axdict = {}  # type: Dict[str, Axes]

        with h5py.File(self.filename, 'a') as db:
            for rodkey in self.rods():
                spacename = db[rodkey].attrs['filename']
                if not os.path.exists(spacename):
                    result = QMessageBox.question(
                        parent,
                        'Warning',
                        f'Cannot find space {rodkey} at file {spacename}; locate proper space',  # noqa
                        QMessageBox.Open,
                        QMessageBox.Ignore
                    )
                    if result == QMessageBox.Open:
                        spacename, _ = QFileDialog.getOpenFileName(
                            caption=f'Open space {rodkey}',
                            directory='.',
                            filter='*.hdf5')
                        db[rodkey].attrs['filename'] = spacename
                    else:
                        raise OSError('Select proper input')
                self.axdict[rodkey] = Axes.fromfile(spacename)

    def create_rod(self, rodkey: str, spacename: str) -> None:
        with h5py.File(self.filename, 'a') as db:
            if rodkey not in list(db.keys()):
                db.create_group(rodkey)
                db[rodkey].attrs['filename'] = spacename
                self.axdict[rodkey] = Axes.fromfile(spacename)

    def delete_rod(self, rodkey: str) -> None:
        with h5py.File(self.filename, 'a') as db:
            del db[rodkey]

    def rods(self) -> List[str]:
        with h5py.File(self.filename, 'a') as db:
            rods = list(db.keys())
        return rods

    def copy(self, oldkey: str, newkey: str) -> None:
        with h5py.File(self.filename, 'a') as db:
            if oldkey in list(db.keys()):
                db.copy(db[oldkey], db, name=newkey)

    @property
    def filelist(self) -> List[str]:
        filelist = []
        with h5py.File(self.filename, 'a') as db:
            for key in db.keys():
                filelist.append(db[key].attrs['filename'])
        return filelist

    def save_axis(self, rodkey: str, axis: str) -> None:
        return self.save(rodkey, 'axis', axis)

    def save_resolution(self, rodkey: str, resolution: float) -> None:
        return self.save(rodkey, 'resolution', resolution)

    def save(self, rodkey: str, key: str, value: ndarray) -> None:
        with h5py.File(self.filename, 'a') as db:
            db[rodkey].attrs[str(key)] = value

    def load(self, rodkey: str, key: str) -> Optional[ndarray]:
        with h5py.File(self.filename, 'a') as db:
            if rodkey in db:
                if key in db[rodkey].attrs:
                    return db[rodkey].attrs[str(key)]
        return None

class RodData(FitData):

    def __init__(self, filename: str, rodkey: Optional[str], axis: str,
                 resolution: float,
                 parent: QWidget) -> None:
        super().__init__(filename, parent)
        if rodkey is not None:
            self.rodkey = rodkey
            self.slicekey = f'{axis}_{resolution}'
            self.axis = axis
            self.resolution = resolution

            with h5py.File(self.filename, 'a') as db:
                if rodkey in db:
                    if self.slicekey not in db[rodkey]:
                        db[rodkey].create_group(self.slicekey)
                        db[rodkey][self.slicekey].create_group('attrs')

    def paxes(self) -> List[Axis]:
        axes = self.axdict[self.rodkey]
        projected = list(axes)
        axindex = axes.index(self.axis)
        projected.pop(axindex)
        return projected

    def get_bins(self) -> Tuple[ndarray, Axis, int]:
        axes = self.axdict[self.rodkey]
        axindex = axes.index(self.axis)
        ax = axes[axindex]

        bins = get_bins(ax, self.resolution)
        return bins, ax, axindex

    def rodlength(self) -> int:
        bins, ax, axindex = self.get_bins()
        return len(bins) - 1

    def get_index_value(self, index: int) -> ndarray:
        values = get_axis_values(self.axdict[self.rodkey],
                                 self.axis,
                                 self.resolution)
        return values[index]

    def get_key(self, index: int) -> List[slice]:
        axes = self.axdict[self.rodkey]
        bins, ax, axindex = self.get_bins()
        assert len(bins) > index, (bins, index)
        start, stop = bins[index], bins[index + 1]
        k = [slice(None) for i in axes]
        k[axindex] = slice(start, stop)
        return k

    def space_from_index(self, index: int) -> Space:
        with h5py.File(self.filename, 'a') as db:
            filename = db[self.rodkey].attrs['filename']
        space = Space.fromfile(filename, self.get_key(index))
        return space.project(self.axis)

    def replace_dataset(self, grp: h5py.Group, dataset_name: str, arr: ndarray) -> None:
        if dataset_name in grp:
            del grp[dataset_name]

        dataset = grp.require_dataset(dataset_name,
                                      arr.shape, dtype=arr.dtype,
                                      exact=False, compression='gzip')
        dataset.write_direct(arr)

    def save_data(self, index: int, key: str, data: MaskedArray) -> None:
        with h5py.File(self.filename, 'a') as db:
            grp = db[self.rodkey][self.slicekey]

            self.replace_dataset(grp, f"{index}_{key}", data)
            self.replace_dataset(grp, f"{index}_{key}_mask", data.mask)

    def load_data(self, index: int, key: str) -> Optional[MaskedArray]:
        with h5py.File(self.filename, 'a') as db:
            try:
                grp = db[self.rodkey][self.slicekey]

                data = grp[f"{index}_{key}"]
                mask = grp[f"{index}_{key}_mask"]

                return numpy.ma.array(data, mask=mask)
            except KeyError:
                return None

    def save_sliceattr(self, index: int, key: str, value: ndarray) -> None:
        mkey = f'mask{key}'
        with h5py.File(self.filename, 'a') as db:
            try:
                group = db[self.rodkey][self.slicekey]['attrs']  # else it breaks with the old fitaid
            except KeyError:
                db[self.rodkey][self.slicekey].create_group('attrs')
                group = db[self.rodkey][self.slicekey]['attrs']
            if key not in group:
                dataset = group.create_dataset(key, (self.rodlength(),))
                dataset = group.create_dataset(mkey, (self.rodlength(),),
                                               dtype=numpy.bool_)
                dataset.write_direct(
                    numpy.ones(self.rodlength(), dtype=numpy.bool_)
                )
            group[key][index] = value
            group[mkey][index] = 0

    def load_sliceattr(self, index: int, key: str) -> Optional[MaskedArray]:
        mkey = f'mask{key}'
        with h5py.File(self.filename, 'a') as db:
            try:
                group = db[self.rodkey][self.slicekey]['attrs']
            except KeyError:
                db[self.rodkey][self.slicekey].create_group('attrs')
                group = db[self.rodkey][self.slicekey]['attrs']
            if key in list(group.keys()):
                g = group[key]
                if g.shape == (0,):
                    return None
                else:
                    mg = group[mkey]
                    if mg.shape == (0,):
                        return None
                    else:
                        return numpy.ma.array(group[key][index],
                                              mask=group[mkey][index])
            else:
                return None

    def all_attrkeys(self) -> List[str]:
        with h5py.File(self.filename, 'a') as db:
            group = db[self.rodkey][self.slicekey]['attrs']
            return list(group.keys())

    def all_from_key(self, key: str) -> Optional[Tuple[ndarray, MaskedArray]]:
        with h5py.File(self.filename, 'a') as db:
            mkey = f'mask{key}'
            axes = self.axdict[self.rodkey]
            group = db[self.rodkey][self.slicekey]['attrs']
            if key in list(group.keys()):
                return (get_axis_values(axes, self.axis,
                                        self.resolution),
                        numpy.ma.array(group[key],
                                       mask=numpy.array(group[mkey])))
        return None

    def load_loc(self, index: Optional[int]) -> Optional[List[Optional[MaskedArray]]]:
        if index is not None:
            loc = list()
            count = itertools.count()
            key = f'guessloc{next(count)}'
            while self.load_sliceattr(index, key) is not None:
                loc.append(self.load_sliceattr(index, key))
                key = f'guessloc{next(count)}'
            if len(loc) > 0:
                return loc
            else:
                count = itertools.count()
                key = f'loc{next(count)}'
                while self.load_sliceattr(index, key) is not None:
                    loc.append(self.load_sliceattr(index, key))
                    key = f'loc{next(count)}'
                if len(loc) > 0:
                    return loc
        return None

    def save_loc(self,
                 index: Optional[int],
                 loc: Sequence[Optional[MaskedArray]]) -> None:
        if index is not None:
            for i, value in enumerate(loc):
                self.save_sliceattr(index, f'guessloc{i}', value)

    def save_segments(self, segments: ndarray) -> None:
        with h5py.File(self.filename, 'a') as db:
            grp = db[self.rodkey][self.slicekey]

            self.replace_dataset(grp, 'segment', segments)

    def load_segments(self) -> Optional[ndarray]:
        with h5py.File(self.filename, 'a') as db:
            try:
                grp = db[self.rodkey][self.slicekey]

                return grp['segment'][()]
            except KeyError:
                return None

    def load_int(self, key: str) -> Optional[int]:
        v = self.load('', key)
        if v is not None:
            return int(v)
        return None

    def save_aroundroi(self, aroundroi: bool) -> None:
        return super().save(self.rodkey, 'aroundroi', aroundroi)

    def save_fromfit(self, fromfit: bool) -> None:
        return super().save(self.rodkey, 'fromfit', fromfit)

    def save_index(self, index: int) -> None:
        return super().save(self.rodkey, 'index', index)

    def save_roi(self, roi: List[float]) -> None:
        return super().save(self.rodkey, 'roi', roi)

    def currentindex(self) -> Optional[int]:
        index = self.load_int('index')
        if index is not None:
            if index < 0 or index >= self.rodlength():  # deal with no selection
                index = None
        return index

    def load(self, _rodkey: str, key: str) -> Optional[ndarray]:
        return super().load(self.rodkey, key)

    def __iter__(self) -> Generator[Space, None, None]:
        for index in range(self.rodlength()):
            yield self.space_from_index(index)

class Window(QMainWindow):

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        newproject = QAction("New project", self)
        newproject.triggered.connect(self.newproject)

        loadproject = QAction("Open project", self)
        loadproject.triggered.connect(self.loadproject)

        addspace = QAction("Import space", self)
        addspace.triggered.connect(self.add_to_project)

        menu_bar = QMenuBar()
        file = menu_bar.addMenu("&File")
        file.addAction(newproject)
        file.addAction(loadproject)
        file.addAction(addspace)

        self.setMenuBar(menu_bar)
        self.statusbar = QStatusBar()

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.tab_widget.removeTab)

        self.setCentralWidget(self.tab_widget)
        self.setMenuBar(menu_bar)
        self.setStatusBar(self.statusbar)

    def newproject(self) -> None:
        dialog = QFileDialog(self, "project filename")
        dialog.setNameFilters(['binoculars fit file (*.fit)'])
        dialog.setDefaultSuffix('fit')
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            widget = TopWidget(str(fname), parent=self)
            self.tab_widget.addTab(widget, short_filename(str(fname)))
            self.tab_widget.setCurrentWidget(widget)
        except Exception as e:
            QMessageBox.critical(
                self,
                'New project',
                f'Unable to save project to {fname}: {e}'
            )

    def loadproject(self, filename: Optional[str]=None) -> None:
        if not filename:
            dialog = QFileDialog(self, "Load project")
            dialog.setNameFilters(['binoculars fit file (*.fit)'])
            dialog.setFileMode(QFileDialog.ExistingFiles)
            dialog.setAcceptMode(QFileDialog.AcceptOpen)
            if not dialog.exec_():
                return
            fname = dialog.selectedFiles()[0]
            if not fname:
                return
            try:
                widget = TopWidget(str(fname), parent=self)
                self.tab_widget.addTab(widget, short_filename(str(fname)))
                self.tab_widget.setCurrentWidget(widget)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    'Load project',
                    f'Unable to load project from {fname}: {e}'
                )
        else:
            widget = TopWidget(str(fname), parent=self)
            self.tab_widget.addTab(widget, 'fname')
            self.tab_widget.setCurrentWidget(widget)

    def add_to_project(self, filename: Optional[str]=None) -> None:
        if self.tab_widget.count() == 0:
            QMessageBox.warning(
                self, 'Warning', 'First select a file to store data')
            self.newproject()

        if not filename:
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
                try:
                    widget = self.tab_widget.currentWidget()
                    widget.addspace(str(name))
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        'Import spaces',
                        f'Unable to import space {fname}: {e}'
                    )
        else:
            widget = self.tab_widget.currentWidget()
            widget.addspace(filename)


class TopWidget(QWidget):

    def __init__(self, filename: str, parent: Optional[QWidget]=None):
        super().__init__(parent)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        minihbox = QHBoxLayout()
        minihbox2 = QHBoxLayout()

        self.database = FitData(filename, self)
        self.table = TableWidget(self.database)
        self.nav = ButtonedSlider()
        self.nav.slice_index.connect(self.index_change)
        self.table.trigger.connect(self.active_change)
        self.table.check_changed.connect(self.refresh_plot)
        self.tab_widget = QTabWidget()

        self.fitwidget = FitWidget(None, self)
        self.integratewidget = IntegrateWidget(None, self, self)
        self.plotwidget = OverviewWidget(None, self)
        self.peakwidget = PeakWidget(None, self)

        self.tab_widget.addTab(self.fitwidget, 'Fit')
        self.tab_widget.addTab(self.integratewidget, 'Integrate')
        self.tab_widget.addTab(self.plotwidget, 'plot')
        self.tab_widget.addTab(self.peakwidget, 'Peaktracker')

        self.emptywidget = QWidget()
        self.emptywidget.setLayout(vbox)

        vbox.addWidget(self.table)
        vbox.addWidget(self.nav)

        self.functions = list() # type: List[Type[PeakFitBase]]
        self.function_box = QComboBox()
        for function in dir(binoculars.fit):
            cls = getattr(binoculars.fit, function)
            if isinstance(cls, type)\
               and issubclass(cls, PeakFitBase):
                self.functions.append(cls)
                self.function_box.addItem(function)
        self.function_box.setCurrentIndex(
            self.function_box.findText('PolarLorentzian2D')
        )

        vbox.addWidget(self.function_box)
        vbox.addLayout(minihbox)
        vbox.addLayout(minihbox2)

        self.all_button = QPushButton('fit all')
        self.rod_button = QPushButton('fit rod')
        self.slice_button = QPushButton('fit slice')

        self.all_button.clicked.connect(self.fit_all)
        self.rod_button.clicked.connect(self.fit_rod)
        self.slice_button.clicked.connect(self.fit_slice)

        minihbox.addWidget(self.all_button)
        minihbox.addWidget(self.rod_button)
        minihbox.addWidget(self.slice_button)

        self.allint_button = QPushButton('int all')
        self.rodint_button = QPushButton('int rod')
        self.sliceint_button = QPushButton('int slice')

        self.allint_button.clicked.connect(self.int_all)
        self.rodint_button.clicked.connect(self.int_rod)
        self.sliceint_button.clicked.connect(self.int_slice)

        minihbox2.addWidget(self.allint_button)
        minihbox2.addWidget(self.rodint_button)
        minihbox2.addWidget(self.sliceint_button)

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.emptywidget)
        splitter.addWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.tab_change)

        hbox.addWidget(splitter)
        self.setLayout(hbox)

    def tab_change(self, index: int) -> None:
        if index == 2:
            self.refresh_plot()

    def addspace(self, filename: Optional[str]=None) -> None:
        self.table.addspace(filename or QFileDialog.getOpenFileName(self, 'Open Project', '.', '*.hdf5')[0])  # noqa

    def active_change(self) -> None:
        rodkey, axis, resolution = self.table.currentkey()
        newdatabase = RodData(self.database.filename, rodkey, axis, resolution, self)
        self.integratewidget.database = newdatabase
        self.peakwidget.database = newdatabase
        self.integratewidget.set_axis()
        self.peakwidget.set_axis()
        self.fitwidget.database = newdatabase
        self.nav.set_length(newdatabase.rodlength())
        index = newdatabase.currentindex()
        if index is not None:
            self.nav.set_index(index)
            self.index_change(index)

    def index_change(self, index: int) -> None:
        # deal with no index
        if index != -1:
            if self.fitwidget.database is not None:
                self.fitwidget.database.save_index(index)
                self.fitwidget.plot(index)
                self.integratewidget.plot(index)

    def refresh_plot(self) -> None:
        self.plotwidget.refresh(
            [RodData(self.database.filename, rodkey, axis, resolution, self)
             for rodkey, axis, resolution in self.table.checked()]
        )

    @property
    def fitclass(self) -> Type[PeakFitBase]:
        return self.functions[self.function_box.currentIndex()]

    def fit_slice(self) -> None:
        index = self.nav.index()
        if self.fitwidget.database is not None:
            space = self.fitwidget.database.space_from_index(index)
            self.fitwidget.fit(index, space, self.fitclass)
            self.fit_loc(self.fitwidget.database)
            self.fitwidget.plot(index)

    def fit_rod(self) -> None:
        def function(index: int, space: Space) -> None:
            self.fitwidget.fit(index, space, self.fitclass)

        if self.fitwidget.database is not None:
            self.progressbox(self.fitwidget.database.rodkey, function, enumerate(
                self.fitwidget.database), self.fitwidget.database.rodlength())
            self.fit_loc(self.fitwidget.database)
            self.fitwidget.plot()

    def fit_all(self) -> None:
        def function(index: int, space: Space) -> None:
            self.fitwidget.fit(index, space, self.fitclass)

        for rodkey, axis, resolution in self.table.checked():
            self.fitwidget.database = RodData(
                self.database.filename, rodkey, axis, resolution, self)
            self.progressbox(
                self.fitwidget.database.rodkey,
                function,
                enumerate(self.fitwidget.database),
                self.fitwidget.database.rodlength()
            )
            self.fit_loc(self.fitwidget.database)

        self.fitwidget.plot()

    def int_slice(self) -> None:
        index = self.nav.index()
        if self.fitwidget.database is not None:
            space = self.fitwidget.database.space_from_index(index)
            self.integratewidget.integrate(index, space)
            self.integratewidget.plot(index)

    def int_rod(self) -> None:
        if self.integratewidget.database is not None:
            self.progressbox(
                self.integratewidget.database.rodkey,
                self.integratewidget.integrate,
                enumerate(self.integratewidget.database),
                self.integratewidget.database.rodlength()
            )
            self.integratewidget.plot()

    def int_all(self) -> None:
        for rodkey, axis, resolution in self.table.checked():
            self.integratewidget.database = RodData(
                self.database.filename, rodkey, axis, resolution, self)
            self.progressbox(
                self.integratewidget.database.rodkey,
                self.integratewidget.integrate,
                enumerate(self.integratewidget.database),
                self.integratewidget.database.rodlength()
            )
        self.integratewidget.plot()

    def fit_loc(self, database: RodData) -> None:
        deg = 2
        for param in database.all_attrkeys():
            if param.startswith('loc'):
                res = database.all_from_key(param)
                if res is not None:
                    x, y = res
                else:
                    return

                res = database.all_from_key(f'var_{param}')
                if res is not None:
                    x, yvar = res
                else:
                    return

                cx = x[numpy.invert(y.mask)]
                y = y.compressed()
                yvar = yvar.compressed()

                w = numpy.log(1 / yvar)
                w[w == numpy.inf] = 0
                w = numpy.nan_to_num(w)
                w[w < 0] = 0
                w[w < numpy.median(w)] = 0
                if len(x) > 0:
                    c = numpy.polynomial.polynomial.polyfit(cx, y, deg, w=w)
                    newy = numpy.polynomial.polynomial.polyval(x, c)
                    for index, newval in enumerate(newy):
                        database.save_sliceattr(
                            index,
                            'guessloc{}'.format(param.lstrip('loc')),
                            newval
                        )

    def progressbox(self, rodkey: str, function: Any, iterator: Any, length: int) -> None:
        pd = QProgressDialog(
            f'Processing {rodkey}', 'Cancel', 0, length)
        pd.setWindowModality(Qt.WindowModal)
        pd.show()

        def progress(index: int, item: Any) -> None:
            pd.setValue(index)
            if pd.wasCanceled():
                raise KeyboardInterrupt
            QApplication.processEvents()
            function(*item)

        for index, item in enumerate(iterator):
            progress(index, item)
        pd.close()


class TableWidget(QWidget):
    trigger = pyqtSignal()
    check_changed = pyqtSignal()

    def __init__(self, database: FitData, parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)

        hbox = QHBoxLayout()
        self.database = database

        self.activeindex = 0

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['', 'rod', 'axis', 'res', 'remove'])

        self.table.cellClicked.connect(self.setlength)

        for index, width in enumerate([25, 150, 40, 50, 70]):
            self.table.setColumnWidth(index, width)

        for filename, rodkey in zip(database.filelist, database.rods()):
            self.addspace(filename, rodkey)

        hbox.addWidget(self.table)
        self.setLayout(hbox)

    def addspace(self, filename: str, rodkey: Optional[str]=None) -> None:
        def remove_callback(rodkey: str) -> Callable[[], None]:
            return lambda: self.remove(rodkey)

        def activechange_callback(index: int) -> Callable[[], None]:
            return lambda: self.setlength(index, 1)

        if rodkey is None:
            rodkey = short_filename(filename)
            if rodkey in self.database.rods():
                newkey = find_unused_rodkey(rodkey, self.database.rods())
                self.database.copy(rodkey, newkey)
                rodkey = newkey

        old_axis, old_resolution = (self.database.load(rodkey, 'axis'),
                                    self.database.load(rodkey, 'resolution'))
        self.database.create_rod(rodkey, filename)
        index = self.table.rowCount()
        self.table.insertRow(index)

        axes = Axes.fromfile(filename)

        checkboxwidget = QCheckBox()
        checkboxwidget.rodkey = rodkey
        checkboxwidget.setChecked(False)
        self.table.setCellWidget(index, 0, checkboxwidget)
        checkboxwidget.clicked.connect(self.check_changed)

        item = QTableWidgetItem(rodkey)
        self.table.setItem(index, 1, item)

        axis = QComboBox()
        for ax in axes:
            axis.addItem(ax.label)
        self.table.setCellWidget(index, 2, axis)
        if old_axis is not None:
            self.table.cellWidget(
                index, 2).setCurrentIndex(axes.index(old_axis))
        elif index > 0:
            self.table.cellWidget(0, 2).setCurrentIndex(
                self.table.cellWidget(0, 2).currentIndex())
        axis.currentIndexChanged.connect(self.trigger)

        resolution = QLineEdit()
        if old_resolution is not None:
            resolution.setText(str(old_resolution))
        elif index > 0:
            resolution.setText(self.table.cellWidget(0, 3).text())
        else:
            resolution.setText(
                str(axes[axes.index(str(axis.currentText()))].res))

        resolution.editingFinished.connect(activechange_callback(index))
        self.table.setCellWidget(index, 3, resolution)

        buttonwidget = QPushButton('remove')
        buttonwidget.clicked.connect(remove_callback(rodkey))
        self.table.setCellWidget(index, 4, buttonwidget)

    def remove(self, rodkey: str) -> None:
        table_rodkeys = [self.table.cellWidget(index, 0).rodkey
                         for index in range(self.table.rowCount())]
        for index, label in enumerate(table_rodkeys):
            if rodkey == label:
                self.table.removeRow(index)
        self.database.delete_rod(rodkey)
        print(f'removed: {rodkey}')

    def setlength(self, y: int, x: int=1) -> None:
        if self.database is not None:
            if x == 1:
                self.activeindex = y
                rodkey, axis, resolution = self.currentkey()
                self.database.save_axis(rodkey, axis)
                self.database.save_resolution(rodkey, resolution)
                self.trigger.emit()

    def currentkey(self) -> Tuple[Any, str, float]:
        rodkey = self.table.cellWidget(self.activeindex, 0).rodkey
        axis = str(self.table.cellWidget(self.activeindex, 2).currentText())
        resolution = float(self.table.cellWidget(self.activeindex, 3).text())
        return rodkey, axis, resolution

    def checked(self) -> List[Tuple[Any, str, float]]:
        selection = []
        for index in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(index, 0)
            if checkbox.isChecked():
                rodkey = self.table.cellWidget(index, 0).rodkey
                axis = str(self.table.cellWidget(index, 2).currentText())
                resolution = float(self.table.cellWidget(index, 3).text())
                selection.append((rodkey, axis, resolution))
        return selection


def short_filename(filename: str) -> str:
    return filename.split('/')[-1].split('.')[0]


class HiddenToolbar(NavigationToolbar2QT):

    def __init__(self, corner_callback: Any, canvas: Any) -> None:
        super().__init__(canvas, None)
        self._corner_callback = corner_callback
        self.zoom()

    def _generate_key(self) -> List[List[float]]:
        limits = []
        for a in self.canvas.figure.get_axes():
            limits.append([a.get_xlim(), a.get_ylim()])
        return limits

    def press_zoom(self, event: Any) -> None:
        super().press_zoom(event)
        self._corner_preclick = self._generate_key()

    def release_zoom(self, event: Any) -> None:
        super().release_zoom(event)
        if self._corner_preclick == self._generate_key():
            self._corner_callback(event.xdata, event.ydata)
        self._corner_preclick = []


class FitWidget(QWidget):

    def __init__(self,
                 database: Optional[RodData]=None,
                 parent: Optional[QWidget]=None):
        super().__init__(parent)

        self.database = database
        vbox = QHBoxLayout()

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar(self.loc_callback, self.canvas)

        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

    def loc_callback(self, x: int, y: int) -> None:
        if self.database is not None and self.ax:
            self.database.save_loc(self.database.currentindex(),
                                   numpy.array([x, y]))

    def plot(self, index: Optional[int]=None) -> None:
        if self.database is not None:
            if index is None:
                index = self.database.currentindex()
            if index is not None:
                space = self.database.space_from_index(index)
                fitdata = self.database.load_data(index, 'fit')
                self.figure.clear()
                self.figure.space_axes = space.axes
                info = self.database.get_index_value(index)
                label = self.database.axis

                if fitdata is not None:
                    if space.dimension == 1:
                        self.ax = self.figure.add_subplot(111)
                        binoculars.plot.plot(
                            space, self.figure, self.ax, fit=fitdata)
                    elif space.dimension == 2:
                        self.ax = self.figure.add_subplot(121)
                        binoculars.plot.plot(space, self.figure, self.ax, fit=None)
                        self.ax = self.figure.add_subplot(122)
                        binoculars.plot.plot(
                            space, self.figure, self.ax, fit=fitdata)
                else:
                    self.ax = self.figure.add_subplot(111)
                    binoculars.plot.plot(space, self.figure, self.ax)
                self.figure.suptitle('{}, res = {}, {} = {}'.format(
                    self.database.rodkey, self.database.resolution, label, info))
                self.canvas.draw()

    def fit(self, index: int, space: Space, function: Any) -> None:
        if self.database is not None:
            if not len(space.get_masked().compressed()) == 0:
                loc = self.get_loc()
                fit = function(space, loc=loc)
                fit.fitdata.mask = space.get_masked().mask
                self.database.save_data(index, 'fit',  fit.fitdata)
                params = list(line.split(':')[0]
                              for line in fit.summary.split('\n'))
                print(fit.result, fit.variance)
                for key, value in zip(params, fit.result):
                    self.database.save_sliceattr(index, key, value)
                for key, value in zip(params, fit.variance):
                    self.database.save_sliceattr(
                        index, f'var_{key}', value)

    def get_loc(self) -> Optional[MaskedArray]:
        if self.database is not None:
            return self.database.load_loc(self.database.currentindex())
        return None

class IntegrateWidget(QWidget):

    def __init__(self,
                 database: Optional[RodData],
                 topwidget: TopWidget,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.database = database
        self.topwidget = topwidget

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar(self.loc_callback, self.canvas)

        hbox = QHBoxLayout()

        splitter = QSplitter(Qt.Vertical)
        self.make_controlwidget()

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.control_widget)

        hbox.addWidget(splitter)
        self.setLayout(hbox)

    def make_controlwidget(self) -> None:
        self.control_widget = QWidget()

        integratebox = QVBoxLayout()
        intensitybox = QHBoxLayout()
        backgroundbox = QHBoxLayout()

        self.aroundroi = QCheckBox('background around roi')
        self.aroundroi.setChecked(True)
        self.aroundroi.clicked.connect(self.refresh_aroundroi)

        self.hsize = QDoubleSpinBox()
        self.vsize = QDoubleSpinBox()

        intensitybox.addWidget(QLabel('roi size:'))
        intensitybox.addWidget(self.hsize)
        intensitybox.addWidget(self.vsize)

        self.left = QDoubleSpinBox()
        self.right = QDoubleSpinBox()
        self.top = QDoubleSpinBox()
        self.bottom = QDoubleSpinBox()

        self.hsize.valueChanged.connect(self.send)
        self.vsize.valueChanged.connect(self.send)
        self.left.valueChanged.connect(self.send)
        self.right.valueChanged.connect(self.send)
        self.top.valueChanged.connect(self.send)
        self.bottom.valueChanged.connect(self.send)

        backgroundbox.addWidget(self.aroundroi)
        backgroundbox.addWidget(self.left)
        backgroundbox.addWidget(self.right)
        backgroundbox.addWidget(self.top)
        backgroundbox.addWidget(self.bottom)

        integratebox.addLayout(intensitybox)
        integratebox.addLayout(backgroundbox)

        self.fromfit = QRadioButton('peak from fit', self)
        self.fromfit.setChecked(True)
        self.fromfit.toggled.connect(self.plot_box)
        self.fromfit.toggled.connect(self.refresh_tracker)

        self.fromsegment = QRadioButton('peak from segment', self)
        self.fromsegment.setChecked(False)
        self.fromsegment.toggled.connect(self.plot_box)
        self.fromsegment.toggled.connect(self.refresh_tracker)

        self.trackergroup = QButtonGroup(self)
        self.trackergroup.addButton(self.fromfit)
        self.trackergroup.addButton(self.fromsegment)

        radiobox = QHBoxLayout()
        radiobox.addWidget(self.fromfit)
        radiobox.addWidget(self.fromsegment)

        integratebox.addLayout(radiobox)

        self.control_widget.setLayout(integratebox)

        # set default values
        # self.hsize.setValue(0.1)
        # self.vsize.setValue(0.1)

    def refresh_aroundroi(self) -> None:
        if self.database is not None:
            self.database.save_aroundroi(self.aroundroi.isChecked())
            axes = self.database.paxes()
            if not self.aroundroi.isChecked():
                self.left.setMinimum(axes[0].min)
                self.left.setMaximum(axes[0].max)
                self.right.setMinimum(axes[0].min)
                self.right.setMaximum(axes[0].max)
                self.top.setMinimum(axes[1].min)
                self.top.setMaximum(axes[1].max)
                self.bottom.setMinimum(axes[1].min)
                self.bottom.setMaximum(axes[1].max)
            else:
                self.left.setMinimum(0)
                self.left.setMaximum(axes[0].max - axes[0].min)
                self.right.setMinimum(0)
                self.right.setMaximum(axes[0].max - axes[0].min)
                self.top.setMinimum(0)
                self.top.setMaximum(axes[1].max - axes[1].min)
                self.bottom.setMinimum(0)
                self.bottom.setMaximum(axes[1].max - axes[1].min)

    def refresh_tracker(self) -> None:
        if self.database is not None:
            index = self.database.currentindex()
            if index is not None:
                self.database.save_fromfit(self.fromfit.isChecked())
                self.plot_box()

    def set_axis(self) -> None:
        if self.database is not None:
            roi = self.database.load('', 'roi')

            aroundroi = self.database.load('', 'aroundroi')
            if aroundroi is not None:
                self.aroundroi.setChecked(bool(aroundroi))
            else:
                self.aroundroi.setChecked(True)
            self.refresh_aroundroi()

            axes = self.database.paxes()

            self.hsize.setSingleStep(axes[1].res)
            self.hsize.setDecimals(len(str(axes[1].res)) - 2)
            self.vsize.setSingleStep(axes[0].res)
            self.vsize.setDecimals(len(str(axes[0].res)) - 2)
            self.left.setSingleStep(axes[1].res)
            self.left.setDecimals(len(str(axes[1].res)) - 2)
            self.right.setSingleStep(axes[1].res)
            self.right.setDecimals(len(str(axes[1].res)) - 2)
            self.top.setSingleStep(axes[0].res)
            self.top.setDecimals(len(str(axes[0].res)) - 2)
            self.bottom.setSingleStep(axes[0].res)
            self.bottom.setDecimals(len(str(axes[0].res)) - 2)

            tracker = self.database.load('', 'fromfit')
            if tracker is not None:
                if tracker:
                    self.fromfit.setChecked(True)
                else:
                    self.fromsegment.setChecked(True)

            if roi is not None:
                boxes = [self.hsize, self.vsize, self.left, self.right, self.top, self.bottom]  # noqa
                for box, value in zip(boxes, roi):
                    box.setValue(value)

    def send(self) -> None:
        if self.database is not None:
            index = self.database.currentindex()
            if index is not None:
                roi = [self.hsize.value(), self.vsize.value(), self.left.value(),
                       self.right.value(), self.top.value(), self.bottom.value()]
                self.database.save_roi(roi)
                self.plot_box()

    def integrate(self, index: int, space: Space) -> None:
        loc = self.get_loc()

        if loc is not None:
            axes = space.axes

            key = space.get_key(self.intkey(loc, axes))

            if self.database is not None:
                fitdata = self.database.load_data(index, 'fit')
                if fitdata is not None:
                    fitintensity = fitdata[key].data.flatten()
                    fitbkg = numpy.hstack([fitdata[space.get_key(bkgkey)].data.flatten()  # noqa
                                           for bkgkey in self.bkgkeys(loc, axes)])
                    if len(fitbkg) == 0:
                        fitstructurefactor = fitintensity.sum()
                    elif len(fitintensity) == 0:
                        fitstructurefactor = numpy.nan
                    else:
                        fitstructurefactor = numpy.sqrt(fitintensity.sum() - len(fitintensity) * 1.0 / len(fitbkg) * fitbkg.sum())  # noqa
                    self.database.save_sliceattr(
                        index, 'fitsf', fitstructurefactor)

                niintensity = space[
                    self.intkey(loc, axes)].get_masked().compressed()

                try:
                    intensity = interpolate(
                        space[self.intkey(loc, axes)]).flatten()
                    bkg = numpy.hstack([space[bkgkey].get_masked().compressed()
                                        for bkgkey in self.bkgkeys(loc, axes)])
                    interdata = space.get_masked()
                    interdata[key] = intensity.reshape(interdata[key].shape)
                    interdata[key].mask = numpy.zeros_like(interdata[key])
                    self.database.save_data(index, 'inter',  interdata)
                except ValueError as e:
                    print(f'Warning error interpolating silce {index}: {e}')  # noqa
                    intensity = numpy.array([])
                    bkg = numpy.array([])
                except QhullError as e:
                    print(f'Warning error interpolating silce {index}: {e}')  # noqa
                    intensity = numpy.array([])
                    bkg = numpy.array([])

                if len(intensity) == 0:
                    structurefactor = numpy.nan
                    nistructurefactor = numpy.nan
                elif len(bkg) == 0:
                    structurefactor = numpy.sqrt(intensity.sum())
                    nistructurefactor = numpy.sqrt(niintensity.sum())
                else:
                    structurefactor = numpy.sqrt(intensity.sum() - len(intensity) * 1.0 / len(bkg) * bkg.sum())  # noqa
                    nistructurefactor = numpy.sqrt(niintensity.sum() - len(niintensity) * 1.0 / len(bkg) * bkg.sum())  # noqa

                self.database.save_sliceattr(index, 'sf', structurefactor)
                self.database.save_sliceattr(index, 'nisf', nistructurefactor)

                print(f'Structurefactor {index}: {structurefactor}')

    def intkey(self,
               coords: Optional[Union[List[Optional[MaskedArray]],
                                      ndarray]],
               axes: Axes) -> Tuple[slice, ...]:
        if coords is not None:
            vsize = self.vsize.value() / 2
            hsize = self.hsize.value() / 2
            return tuple(ax.restrict_slice(slice(coord - size, coord + size))
                         for ax, coord, size in zip(axes, coords, [vsize, hsize])
                         if coord is not None)
        return ()

    def bkgkeys(self,
                coords: Optional[Union[List[Optional[MaskedArray]],
                                       ndarray]],
                axes: Axes) -> Union[Tuple[Tuple[slice, slice],
                                           Tuple[slice, slice],
                                           Tuple[slice, slice],
                                           Tuple[slice, slice]],
                                     List[Tuple[slice, slice]]]:
        if self.database is not None:
            aroundroi = self.database.load('', 'aroundroi')
            if aroundroi and coords is not None and coords[0] is not None and coords[1] is not None:
                key = self.intkey(coords, axes)

                vsize = self.vsize.value() / 2
                hsize = self.hsize.value() / 2

                leftkey_slice = slice(coords[1] - hsize - self.left.value(),
                                      coords[1] - hsize)
                leftkey = (key[0], axes[1].restrict_slice(leftkey_slice))

                rightkey_slice = slice(coords[1] + hsize,
                                       coords[1] + hsize + self.right.value())
                rightkey = (key[0], axes[1].restrict_slice(rightkey_slice))

                topkey_slice = slice(coords[0] - vsize - self.top.value(),
                                     coords[0] - vsize)
                topkey = (axes[0].restrict_slice(topkey_slice), key[1])

                bottomkey_slice = slice(coords[0] + vsize,
                                        coords[0] + vsize + self.bottom.value())
                bottomkey = (axes[0].restrict_slice(bottomkey_slice), key[1])

                return leftkey, rightkey, topkey, bottomkey
            else:
                slice0 = slice(self.left.value(), self.right.value())
                slice1 = slice(self.top.value(), self.bottom.value())
                return [(axes[0].restrict_slice(slice0),
                         axes[1].restrict_slice(slice1))]
        return []

    def get_loc(self) -> Optional[Union[List[Optional[MaskedArray]],
                                        ndarray]]:
        if self.database is not None:
            index = self.database.currentindex()
            if index is not None:
                if self.fromfit.isChecked():
                    return self.database.load_loc(index)
                else:
                    indexvalue = self.database.get_index_value(index)
                    return self.topwidget.peakwidget.get_coords(indexvalue)
        return None

    def loc_callback(self, x: int, y: int) -> None:
        if self.ax and self.database is not None:
            index = self.database.currentindex()
            if index is not None:
                if self.fromfit.isChecked():
                    self.database.save_loc(index,
                                           numpy.array([x, y]))
                else:
                    indexvalue = self.database.get_index_value(index)
                    self.topwidget.peakwidget.add_row(numpy.array([indexvalue, x, y]))
                self.plot_box()

    def plot(self, index: Optional[int]=None) -> None:
        if self.database is not None:
            if index is None:
                index = self.database.currentindex()
            if index is not None:
                space = self.database.space_from_index(index)
                interdata = self.database.load_data(index, 'inter')
                info = self.database.get_index_value(index)
                label = self.database.axis

                self.figure.clear()
                self.figure.space_axes = space.axes

                if interdata is not None:
                    if space.dimension == 1:
                        self.ax = self.figure.add_subplot(111)
                        binoculars.plot.plot(
                            space, self.figure, self.ax, fit=interdata)
                    elif space.dimension == 2:
                        self.ax = self.figure.add_subplot(121)
                        binoculars.plot.plot(space, self.figure, self.ax, fit=None)
                        self.ax = self.figure.add_subplot(122)
                        binoculars.plot.plot(
                            space, self.figure, self.ax, fit=interdata)
                else:
                    self.ax = self.figure.add_subplot(111)
                    binoculars.plot.plot(space, self.figure, self.ax)

                self.figure.suptitle('{}, res = {}, {} = {}'.format(
                    self.database.rodkey, self.database.resolution, label, info))

                self.plot_box()
                self.canvas.draw()

    def plot_box(self) -> None:
        if self.database is not None:
            index = self.database.currentindex()
            if index is not None:
                loc = self.get_loc()
                if len(self.figure.get_axes()) != 0 and loc is not None:
                    ax = self.figure.get_axes()[0]
                    axes = self.figure.space_axes
                    key = self.intkey(loc, axes)
                    bkgkey = self.bkgkeys(loc, axes)
                    ax.patches.clear()
                    rect = Rectangle((key[0].start, key[1].start),
                                     key[0].stop - key[0].start,
                                     key[1].stop - key[1].start,
                                     alpha=0.2, color='k')
                    ax.add_patch(rect)
                    for k in bkgkey:
                        bkg = Rectangle((k[0].start, k[1].start),
                                        k[0].stop - k[0].start,
                                        k[1].stop - k[1].start,
                                        alpha=0.2, color='r')
                        ax.add_patch(bkg)
                    self.canvas.draw()


class ButtonedSlider(QWidget):
    slice_index = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget]=None):
        super().__init__(parent)

        self.navigation_button_left_end = QPushButton('|<')
        self.navigation_button_left_one = QPushButton('<')
        self.navigation_slider = QSlider(Qt.Horizontal)
        self.navigation_slider.sliderReleased.connect(self.send)

        self.navigation_button_right_one = QPushButton('>')
        self.navigation_button_right_end = QPushButton('>|')

        self.navigation_button_left_end.setMaximumWidth(20)
        self.navigation_button_left_one.setMaximumWidth(20)
        self.navigation_button_right_end.setMaximumWidth(20)
        self.navigation_button_right_one.setMaximumWidth(20)

        self.navigation_button_left_end.clicked.connect(
            self.slider_change_left_end)
        self.navigation_button_left_one.clicked.connect(
            self.slider_change_left_one)
        self.navigation_button_right_end.clicked.connect(
            self.slider_change_right_end)
        self.navigation_button_right_one.clicked.connect(
            self.slider_change_right_one)

        box = QHBoxLayout()
        box.addWidget(self.navigation_button_left_end)
        box.addWidget(self.navigation_button_left_one)
        box.addWidget(self.navigation_slider)
        box.addWidget(self.navigation_button_right_one)
        box.addWidget(self.navigation_button_right_end)

        self.setDisabled(True)
        self.setLayout(box)

    def set_length(self, length: int) -> None:
        self.navigation_slider.setMinimum(0)
        self.navigation_slider.setMaximum(length - 1)
        self.navigation_slider.setTickPosition(QSlider.TicksBelow)
        self.navigation_slider.setValue(0)
        self.setEnabled(True)

    def send(self) -> None:
        self.slice_index.emit(self.navigation_slider.value())

    def slider_change_left_one(self) -> None:
        self.navigation_slider.setValue(
            max(self.navigation_slider.value() - 1, 0))
        self.send()

    def slider_change_left_end(self) -> None:
        self.navigation_slider.setValue(0)
        self.send()

    def slider_change_right_one(self) -> None:
        self.navigation_slider.setValue(
            min(self.navigation_slider.value() + 1,
                self.navigation_slider.maximum()))
        self.send()

    def slider_change_right_end(self) -> None:
        self.navigation_slider.setValue(self.navigation_slider.maximum())
        self.send()

    def index(self) -> int:
        return self.navigation_slider.value()

    def set_index(self, index: int) -> None:
        self.navigation_slider.setValue(index)


class HiddenToolbar2(NavigationToolbar2QT):

    def __init__(self, canvas: Any):
        super().__init__(canvas, None)
        self.zoom()


class OverviewWidget(QWidget):

    def __init__(self,
                 database: Optional[RodData],
                 parent: Optional[QWidget]=None):
        super().__init__(parent)

        self.databaselist = []  # type: List[RodData]

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar2(self.canvas)

        self.table = QTableWidget(0, 2)
        self.make_table()

        self.table.cellClicked.connect(self.plot)

        hbox = QHBoxLayout()

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.control_widget)

        hbox.addWidget(splitter)
        self.setLayout(hbox)

    def select(self) -> List[str]:
        selection = []
        for index in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(index, 0)
            if checkbox.isChecked():
                selection.append(str(self.table.cellWidget(index, 1).text()))
        return selection

    def make_table(self) -> None:
        self.control_widget = QWidget()
        vbox = QVBoxLayout()
        minibox = QHBoxLayout()

        vbox.addWidget(self.table)
        self.table.setHorizontalHeaderLabels(['', 'param'])
        for index, width in enumerate([25, 50]):
            self.table.setColumnWidth(index, width)
        self.log = QCheckBox('log')
        self.log.clicked.connect(self.plot)
        self.export_button = QPushButton('export curves')

        self.export_button.clicked.connect(self.export)

        minibox.addWidget(self.log)
        minibox.addWidget(self.export_button)
        vbox.addLayout(minibox)
        self.control_widget.setLayout(vbox)

    def export(self) -> None:
        folder = str(QFileDialog.getExistingDirectory(
            self, "Select directory to save curves"))
        params = self.select()
        for param in params:
            for database in self.databaselist:
                res = database.all_from_key(param)
                if res is not None:
                    x, y = res
                    args = numpy.argsort(x)
                    filename = f'{param}_{database.rodkey}.txt'
                    numpy.savetxt(os.path.join(folder, filename),
                                  numpy.vstack(arr[args] for arr in [x, y]).T)

    def refresh(self, databaselist: List[RodData]) -> None:
        self.databaselist = databaselist
        params = self.select()
        while self.table.rowCount() > 0:
            self.table.removeRow(0)

        allparams = [[param
                      for param in database.all_attrkeys()
                      if not param.startswith('mask')]
                     for database in databaselist]

        allparams.extend([['locx_s', 'locy_s']
                          for database in databaselist
                          if database.load_segments() is not None])

        if len(allparams) > 0:
            uniqueparams = numpy.unique(
                numpy.hstack([params for params in allparams]))
        else:
            uniqueparams = []

        for param in uniqueparams:
            index = self.table.rowCount()
            self.table.insertRow(index)

            checkboxwidget = QCheckBox()
            if param in params:
                checkboxwidget.setChecked(True)
            else:
                checkboxwidget.setChecked(False)
            self.table.setCellWidget(index, 0, checkboxwidget)
            checkboxwidget.clicked.connect(self.plot)

            item = QLabel(param)
            self.table.setCellWidget(index, 1, item)

        self.plot()

    def plot(self) -> None:
        params = self.select()
        self.figure.clear()

        self.ax = self.figure.add_subplot(111)
        for param in params:
            for database in self.databaselist:
                if param == 'locx_s':
                    segments = database.load_segments()
                    if segments is not None:
                        x = numpy.hstack(
                            [database.get_index_value(index)
                             for index in range(database.rodlength())]
                        )
                        y = numpy.vstack(
                            [get_coords(xvalue, segments) for xvalue in x]
                        )
                        self.ax.plot(
                            x, y[:, 0], '+',
                            label='{} - {}'.format('locx_s', database.rodkey)
                        )
                elif param == 'locy_s':
                    segments = database.load_segments()
                    if segments is not None:
                        x = numpy.hstack(
                            [database.get_index_value(index)
                             for index in range(database.rodlength())]
                        )
                        y = numpy.vstack(
                            [get_coords(xvalue, segments) for xvalue in x]
                        )
                        self.ax.plot(
                            x, y[:, 1], '+',
                            label='{} - {}'.format('locy_s', database.rodkey)
                        )
                else:
                    res = database.all_from_key(param)
                    if res is not None:
                        x, y = res
                        self.ax.plot(
                            x, y, '+',
                            label=f'{param} - {database.rodkey}'
                        )

        self.ax.legend()
        if self.log.isChecked():
            self.ax.semilogy()
        self.canvas.draw()


class PeakWidget(QWidget):

    def __init__(self,
                 database: Optional[RodData],
                 parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.database = database

        # create a QTableWidget
        self.table = QTableWidget(0, 3, self)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self.save)

        self.btn_add_row = QPushButton('+', self)
        self.btn_add_row.clicked.connect(self.add_row)  # TODO this cause an issu by passing a boool instead of a nunpy array.

        self.buttonRemove = QPushButton('-', self)
        self.buttonRemove.clicked.connect(self.remove)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        hbox.addWidget(self.btn_add_row)
        hbox.addWidget(self.buttonRemove)

        vbox.addLayout(hbox)
        vbox.addWidget(self.table)
        self.setLayout(vbox)

    def set_axis(self) -> None:
        if self.database is not None:
            self.axes = self.database.paxes()
            while self.table.rowCount() > 0:
                self.table.removeRow(0)
            segments = self.database.load_segments()
            if segments is not None:
                for index in range(segments.shape[0]):
                    self.add_row(segments[index, :])
            self.table.setHorizontalHeaderLabels(
                [f'{self.database.axis}',
                 f'{self.axes[0].label}',
                 f'{self.axes[1].label}']
            )

    def add_row(self, row: Optional[ndarray]=None) -> None:
        rowindex = self.table.rowCount()
        self.table.insertRow(rowindex)
        if row is not None:
            for index in range(3):
                newitem = QTableWidgetItem(str(row[index]))
                self.table.setItem(rowindex, index, newitem)

    def remove(self) -> None:
        self.table.removeRow(self.table.currentRow())
        self.save()

    def axis_coords(self) -> ndarray:
        a = numpy.zeros((self.table.rowCount(), self.table.columnCount()))
        for rowindex in range(a.shape[0]):
            for columnindex in range(a.shape[1]):
                item = self.table.item(rowindex, columnindex)
                if item is not None:
                    a[rowindex, columnindex] = float(item.text())
        return a

    def save(self) -> None:
        if self.database is not None:
            self.database.save_segments(self.axis_coords())

    def get_coords(self, x: int) -> Optional[ndarray]:
        return get_coords(x, self.axis_coords())


def get_coords(x: int, coords: ndarray) -> Optional[ndarray]:
    if coords.shape[0] == 0:
        return None

    if coords.shape[0] == 1:
        return coords[0, 1:]

    args = numpy.argsort(coords[:, 0])

    x0 = coords[args, 0]
    x1 = coords[args, 1]
    x2 = coords[args, 2]

    if x < x0.min():
        first = 0
        last = 1
    elif x > x0.max():
        first = -2
        last = -1
    else:
        first = numpy.searchsorted(x0, x) - 1
        last = numpy.searchsorted(x0, x)

    a1 = (x1[last] - x1[first]) / (x0[last] - x0[first])
    b1 = x1[first] - a1 * x0[first]
    a2 = (x2[last] - x2[first]) / (x0[last] - x0[first])
    b2 = x2[first] - a2 * x0[first]

    return numpy.array([a1 * x + b1, a2 * x + b2])


def interpolate(space: Space) -> MaskedArray:
    data = space.get_masked()
    mask = data.mask
    grid = numpy.vstack([numpy.ma.array(g, mask=mask).compressed()
                         for g in space.get_grid()]).T
    open = numpy.vstack(
        [numpy.ma.array(g, mask=numpy.invert(mask)).compressed()
         for g in space.get_grid()]
    ).T
    if open.shape[0] == 0:
        return data.compressed()
    elif grid.shape[0] == 0:
        return data.compressed()
    else:
        interpolated = griddata(grid, data.compressed(), open)
        values = data.data.copy()
        values[mask] = interpolated
        mask = numpy.isnan(values)
        if mask.sum() > 0:
            data = numpy.ma.array(values, mask=mask)
            grid = numpy.vstack([numpy.ma.array(g, mask=mask).compressed()
                                 for g in space.get_grid()]).T
            open = numpy.vstack(
                [numpy.ma.array(g, mask=numpy.invert(mask)).compressed()
                 for g in space.get_grid()]
            ).T
            interpolated = griddata(
                grid, data.compressed(), open, method='nearest')
            values[mask] = interpolated
        return values


def find_unused_rodkey(rodkey: str, rods: List[str]) -> str:
    if rodkey not in rods:
        newkey = rodkey
    else:
        for index in itertools.count(0):
            newkey = f'{rodkey}_{index}'
            if newkey not in rods:
                break
    return newkey


def main():
    app = QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    main.show()

    sys.exit(app.exec_())
