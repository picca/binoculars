#!/usr/bin/env python3
"""
Created on Wed Dec 07 11:10:28 2016

@author: Prevot
"""
# program for visualisation of the binoculars file the values
# corresponds to the pixel center. ie, there is 11 points in the [0,1]
# range with 0.1 resolution the data are loaded in the form of a numpy
# array of shape (nx,ny,nz)

# -*- coding: utf-8 -*-
#
# Copyright © 2009-2011, 2020, 2021, 2023 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Oblique averaged cross section test"""

SHOW = True  # Show test in GUI-based test launcher

import os.path as osp
import numpy as np
import tables

import guiqwt.cross_section

# debug mode shows the ROI in the top-left corner of the image plot:
guiqwt.cross_section.DEBUG = True

from guidata.configtools import get_icon
from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import IntItem, FloatItem, BoolItem

from guidata.qt.QtCore import SIGNAL, QSize, QObject, Qt, QPoint, QPointF
from guidata.qt.QtGui import (
    qApp,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QKeyEvent,
    QWidget,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QColor,
    QComboBox,
    QAction,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPolygonF,
    QPushButton,
    QButtonGroup,
    QDialog,
    QMessageBox,
    QToolBar,
    QGroupBox,
)
from guiqwt.baseplot import canvas_to_axes
from guiqwt.builder import PlotItemBuilder as GuiPlotItemBuilder
from guiqwt.config import _, make_title
from guiqwt.curve import CurvePlot, CurveItem
from guiqwt.events import DragHandler, setup_standard_tool_filter
from guiqwt.histogram import lut_range_threshold
from guiqwt.image import MaskedImageItem
from guiqwt.interfaces import ICurveItemType, IPanel
from guiqwt.plot import ImageDialog, PlotManager, CurveDialog
from guiqwt.signals import (
    SIG_ITEMS_CHANGED,
    SIG_STOP_NOT_MOVING,
    SIG_MOVE,
    SIG_STOP_MOVING,
    SIG_START_TRACKING,
    SIG_MARKER_CHANGED,
    SIG_RANGE_CHANGED,
    SIG_ITEM_SELECTION_CHANGED,
    SIG_VALIDATE_TOOL,
    SIG_ACTIVE_ITEM_CHANGED,
)
from guiqwt.styles import ImageParam, MaskedImageParam, CurveParam
from guiqwt.tools import (
    OpenFileTool,
    CommandTool,
    DefaultToolbarID,
    InteractiveTool,
    RectangleTool,
    CircleTool,
    FreeFormTool,
    BasePlotMenuTool,
    AntiAliasingTool,
    SelectTool,
)
from guiqwt.panels import PanelWidget
from guiqwt.shapes import (
    XRangeSelection,
    Marker,
    PointShape,
    RectangleShape,
    EllipseShape,
    PolygonShape,
)
from guiqwt._scaler import _histogram
from scipy.interpolate import griddata
from scipy.ndimage import laplace, uniform_filter
from scipy import signal

from grafit import Fit2DWindow

CURVE_COUNT = 0
SIG_STOP_TRACKING = SIGNAL("stop_tracking")
SIG_DOUBLE_VALUE_CHANGED = SIGNAL("valueChanged(double)")
SIG_INT_VALUE_CHANGED = SIGNAL("valueChanged(int)")
SIG_SLIDER_PRESSED = SIGNAL("sliderPressed()")
SIG_SLIDER_RELEASED = SIGNAL("sliderReleased()")
SIG_STATE_CHANGED = SIGNAL("stateChanged(int)")
SIG_CLICKED = SIGNAL("clicked()")
ID_IMAGEMASKING = "image masking"


class XRangeSelection2(XRangeSelection):
    def move_point_to(self, hnd, pos, ctrl=None):
        val, _ = pos
        if hnd == 0:
            if val <= self._max:
                self._min = val
        elif hnd == 1:
            if val >= self._min:
                self._max = val
        elif hnd == 2:
            move = val - (self._max + self._min) / 2
            self._min += move
            self._max += move

        self.plot().emit(SIG_RANGE_CHANGED, self, self._min, self._max)


def _nanmin(data):
    if data.dtype.name in ("float32", "float64", "float128"):
        return np.nanmin(data)
    else:
        return data.min()


def _nanmax(data):
    if data.dtype.name in ("float32", "float64", "float128"):
        return np.nanmax(data)
    else:
        return data.max()


def remove_masked_slices(line, length):
    # remove slice if length<param.length
    if np.any(line.mask):
        slilist = np.ma.clump_masked(line)
        for sli in slilist:
            if sli.stop - sli.start < length:
                line.mask[sli] = False
    return line


def mask_linear_defects(data, threshold=3, length=1):
    diff1 = np.diff(data, n=1, axis=0)  # difference between line j+1 and j
    rms = np.mean(diff1 * diff1)
    diff2 = np.roll(diff1, 1, axis=0)  # first line shifted
    dd = -diff1[1:] * diff2[1:]
    mask = dd > rms * threshold  # values that are above threshold
    msign = diff1[1:] > 0
    # lines below
    ldown = np.logical_and(mask, np.logical_not(msign))
    # lines above
    lup = np.logical_and(mask, msign)

    temp = np.ma.zeros(mask.shape)
    temp.mask = ldown

    np.ma.apply_along_axis(remove_masked_slices, 1, temp, *(length,))
    ldown = np.copy(temp.mask)

    temp.mask = lup
    np.ma.apply_along_axis(remove_masked_slices, 1, temp, *(length,))

    mask = np.zeros_like(data, dtype=np.bool_)
    mask[1:-1] = np.logical_or(ldown, temp.mask)
    return mask


class Preferences(DataSet):
    eraser_size = IntItem("Eraser size", default=1, min=1)


class LaplaceParam(DataSet):
    cutoff = IntItem("remove contributions smaller than", default=1, min=1)
    steps = IntItem("integration steps", default=10, min=0)
    increase = FloatItem("speed up coefficient", default=1.1, min=1.0)
    decrease = FloatItem(
        "speed down coefficient", default=0.5, nonzero=True, max=1.0
    ).set_pos(col=1)
    fill = FloatItem("starting values", default=1.0, min=0.0)
    fillc = FloatItem("contributions", default=10, min=1.0).set_pos(col=1)
    doslice = BoolItem("apply along slice", default=False)


class FilterParam(DataSet):
    cutoff = IntItem("remove contributions smaller than", default=1, min=1)


class BgRangeSelection(XRangeSelection2):
    def __init__(self, _min, _max, shapeparam=None):
        super(BgRangeSelection, self).__init__(_min, _max, shapeparam=shapeparam)
        self.shapeparam.fill = "#000000"
        self.shapeparam.update_range(self)  # creates all the above QObjects


class BinoPlot(CurvePlot):
    def keyPressEvent(self, event):
        "on redefinit l'action liee a un evenement clavier quand la fenetre a le focus"
        # the keys have been inverted for the vertical displacement because the y axis is oriented in the increasing direction

        if type(event) == QKeyEvent:
            # here accept the event and do something
            touche = event.key()
            if touche == 16777236:
                self.emit(SIGNAL("Move(int)"), 1)
            elif touche == 16777234:
                self.emit(SIGNAL("Move(int)"), -1)


class BinoCurve(CurveItem):
    def get_closest_x(self, xc, yc):  # need to correct an error in guiqwt!
        # We assume X is sorted, otherwise we'd need :
        # argmin(abs(x-xc))
        i = self._x.searchsorted(xc)
        n = len(self._x)
        if 0 < i < n:
            if np.fabs(self._x[i - 1] - xc) < np.fabs(self._x[i] - xc):
                return self._x[i - 1], self._y[i - 1]
        elif i == n:
            i = n - 1
        return self._x[i], self._y[i]


class AlphaMaskedArea(object):
    """Defines masked/alpha areas for a masked/alpha image item"""

    """geometry can be rectangular, elliptical or polygonal"""
    """mask can be applied inside or outside the shape"""
    """the shape can be used to mask or unmask"""
    """a gradient can be applied along a direction """

    def __init__(self, geometry=None, pts=None, inside=None, mask=None, gradient=None):
        self.geometry = geometry
        self.pts = pts
        self.inside = inside
        self.mask = mask
        self.gradient = gradient

    def __eq__(self, other):
        return (
            self.geometry == other.geometry
            and np.array_equal(self.pts, other.pts)
            and self.inside == other.inside
            and self.mask == self.mask
            and self.gradient == self.gradient
        )

    def serialize(self, writer):
        """Serialize object to HDF5 writer"""
        for name in ("geometry", "inside", "mask", "gradient", "pts"):
            writer.write(getattr(self, name), name)

    def deserialize(self, reader):
        """Deserialize object from HDF5 reader"""
        self.geometry = reader.read("geometry")
        self.inside = reader.read("inside")
        self.mask = reader.read("mask")
        self.gradient = reader.read("gradient")
        self.pts = reader.read(group_name="pts", func=reader.read_array)


class MaskedImageNan(MaskedImageItem):
    def get_histogram(self, nbins):
        """interface de IHistDataSource"""
        if self.data is None:
            return [0,], [0, 1]
        if self.histogram_cache is None or nbins != self.histogram_cache[0].shape[0]:
            # from guidata.utils import tic, toc
            if False:
                # tic("histo1")
                res = np.histogram(self.data[~np.isnan(self.data)], nbins)
                # toc("histo1")
            else:
                # TODO: _histogram is faster, but caching is buggy
                # in this version
                # tic("histo2")
                _min = _nanmin(self.data)
                _max = _nanmax(self.data)
                if self.data.dtype in (np.float64, np.float32):
                    bins = np.unique(
                        np.array(
                            np.linspace(_min, _max, nbins + 1), dtype=self.data.dtype
                        )
                    )
                else:
                    bins = np.arange(_min, _max + 2, dtype=self.data.dtype)
                res2 = np.zeros((bins.size + 1,), np.uint32)
                _histogram(self.data.flatten(), bins, res2)
                # toc("histo2")
                res = res2[1:-1], bins
            self.histogram_cache = res
        else:
            res = self.histogram_cache
        return res

    def set_masked_areas(self, areas):
        """Set masked areas (see set_mask_filename)"""
        self._masked_areas = areas

    def get_masked_areas(self):
        return self._masked_areas

    def add_masked_area(self, geometry, pts, inside, mask):
        area = AlphaMaskedArea(
            geometry=geometry, pts=pts, inside=inside, mask=mask, gradient=None
        )
        for _area in self._masked_areas:
            if area == _area:
                return
        self._masked_areas.append(area)

    def mask_rectangular_area(
        self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True
    ):
        """
        Mask rectangular area
        If inside is True (default), mask the inside of the area
        Otherwise, mask the outside
        """
        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
        if inside:
            self.data[iy0:iy1, ix0:ix1] = np.ma.masked
        else:
            indexes = np.ones(self.data.shape, dtype=np.bool_)
            indexes[iy0:iy1, ix0:ix1] = False
            self.data[indexes] = np.ma.masked
        if trace:
            self.add_masked_area(
                "rectangular",
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                inside,
                mask=True,
            )
        if do_signal:
            self._mask_changed()

    def mask_square_area(self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True):
        """
        Mask a square area defined by the sponge
        If inside is True (default), mask the inside of the area
        Otherwise, mask the outside
        """
        ix0, iy0 = self.get_nearest_indexes(x0, y0)
        ix1, iy1 = self.get_nearest_indexes(x1, y1)

        if inside:
            self.data[iy0:iy1, ix0:ix1] = np.ma.masked
        else:
            indexes = np.ones(self.data.shape, dtype=np.bool_)
            indexes[iy0:iy1, ix0:ix1] = False
            self.data[indexes] = np.ma.masked
        if trace:
            self.add_masked_area(
                "square",
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                inside,
                mask=True,
            )
        if do_signal:
            self._mask_changed()

    def mask_circular_area(
        self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True
    ):
        """
        Mask circular area, inside the rectangle (x0, y0, x1, y1), i.e.
        circle with a radius of .5*(x1-x0)
        If inside is True (default), mask the inside of the area
        Otherwise, mask the outside
        """
        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        radius = 0.5 * (x1 - x0)
        xdata, ydata = self.get_x_values(ix0, ix1), self.get_y_values(iy0, iy1)
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                distance = np.sqrt(
                    (xdata[ix - ix0] - xc) ** 2 + (ydata[iy - iy0] - yc) ** 2
                )
                if inside:
                    if distance <= radius:
                        self.data[iy, ix] = np.ma.masked
                elif distance > radius:
                    self.data[iy, ix] = np.ma.masked
        if not inside:
            self.mask_rectangular_area(x0, y0, x1, y1, inside, trace=False)
        if trace:
            xc = (x0 + x1) / 2.0
            yc = (y0 + y1) / 2.0
            dx = abs(x1 - x0) / 2.0
            dy = abs(y1 - y0) / 2.0
            self.add_masked_area(
                "circular",
                np.array([[xc, yc + dy], [xc, yc - dy], [xc + dx, yc], [xc - dx, yc]]),
                inside,
                mask=True,
            )
        if do_signal:
            self._mask_changed()

    def mask_polygonal_area(self, pts, inside=True, trace=True, do_signal=True):
        """
        Mask polygonal area, inside the rectangle (x0, y0, x1, y1)
        #points is a np array of the polygon points
        """
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)

        if not self.plot():
            return

        # we construct a QpolygonF to use the containsPoint function of PyQt
        poly = QPolygonF()

        for i in range(pts.shape[0]):
            poly.append(QPointF(pts[i, 0], pts[i, 1]))

        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)

        xdata, ydata = (
            self.get_x_values(ix0, ix1),
            self.get_y_values(iy0, iy1),
        )  # values in the axis referential
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                inside_poly = poly.containsPoint(
                    QPointF(xdata[ix - ix0], ydata[iy - iy0]), Qt.OddEvenFill
                )

                if inside:
                    if inside_poly:
                        self.data[iy, ix] = np.ma.masked
                elif not inside_poly:
                    self.data[iy, ix] = np.ma.masked
        if not inside:
            self.mask_rectangular_area(x0, y0, x1, y1, inside, trace=False)
        if trace:
            self.add_masked_area("polygonal", pts, inside, mask=True)
        if do_signal:
            self._mask_changed()

    def unmask_rectangular_area(
        self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True
    ):
        """
        Unmask rectangular area
        If inside is True (default), unmask the inside of the area
        Otherwise, unmask the outside
        """
        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
        if inside:
            self.data.mask[iy0:iy1, ix0:ix1] = False
        else:
            indexes = np.ones(self.data.shape, dtype=np.bool_)
            indexes[iy0:iy1, ix0:ix1] = False
            self.data.mask[indexes] = False
        if trace:
            self.add_masked_area(
                "rectangular",
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                inside,
                mask=False,
            )
        if do_signal:
            self._mask_changed()

    def unmask_square_area(
        self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True
    ):
        """
        Unmask square area
        If inside is True (default), unmask the inside of the area
        Otherwise, unmask the outside
        """
        ix0, iy0 = self.get_nearest_indexes(x0, y0)
        ix1, iy1 = self.get_nearest_indexes(x1, y1)

        if inside:
            self.data.mask[iy0:iy1, ix0:ix1] = False
        else:
            indexes = np.ones(self.data.shape, dtype=np.bool_)
            indexes[iy0:iy1, ix0:ix1] = False
            self.data.mask[indexes] = False
        if trace:
            self.add_masked_area(
                "square",
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                inside,
                mask=False,
            )
        if do_signal:
            self._mask_changed()

    def unmask_circular_area(
        self, x0, y0, x1, y1, inside=True, trace=True, do_signal=True
    ):
        """
        Unmask circular area, inside the rectangle (x0, y0, x1, y1), i.e.
        circle with a radius of .5*(x1-x0)
        If inside is True (default), unmask the inside of the area
        Otherwise, unmask the outside
        """
        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        radius = 0.5 * (x1 - x0)
        xdata, ydata = self.get_x_values(ix0, ix1), self.get_y_values(iy0, iy1)
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                distance = np.sqrt(
                    (xdata[ix - ix0] - xc) ** 2 + (ydata[iy - iy0] - yc) ** 2
                )
                if inside:
                    if distance <= radius:
                        self.data.mask[iy, ix] = False
                elif distance > radius:
                    self.data.mask[iy, ix] = False

        if not inside:
            self.unmask_rectangular_area(x0, y0, x1, y1, inside, trace=False)
        if trace:
            xc = (x0 + x1) / 2.0
            yc = (y0 + y1) / 2.0
            dx = abs(x1 - x0) / 2.0
            dy = abs(y1 - y0) / 2.0
            self.add_masked_area(
                "circular",
                np.array([[xc, yc + dy], [xc, yc - dy], [xc + dx, yc], [xc - dx, yc]]),
                inside,
                mask=False,
            )
        if do_signal:
            self._mask_changed()

    def unmask_polygonal_area(self, pts, inside=True, trace=True, do_signal=True):
        """
        Unmask polygonal area, inside the polygon
        points is a np array of the polygon points
        """
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)

        if not self.plot():
            return

        # we construct a QpolygonF to use the containsPoint function of PyQt
        poly = QPolygonF()

        for i in range(pts.shape[0]):
            poly.append(QPointF(pts[i, 0], pts[i, 1]))

        ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)

        xdata, ydata = (
            self.get_x_values(ix0, ix1),
            self.get_y_values(iy0, iy1),
        )  # values in the axis referential
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                inside_poly = poly.containsPoint(
                    QPointF(xdata[ix - ix0], ydata[iy - iy0]), Qt.OddEvenFill
                )

                if inside:
                    if inside_poly:
                        self.data.mask[iy, ix] = False
                elif not inside_poly:
                    self.data.mask[iy, ix] = False
        if not inside:
            self.unmask_rectangular_area(x0, y0, x1, y1, inside, trace=False)
        if trace:
            self.add_masked_area("polygonal", pts, inside, mask=False)
        if do_signal:
            self._mask_changed()

    def update_mask(self):
        if isinstance(self.data, np.ma.MaskedArray):
            self.data.set_fill_value(self.imageparam.filling_value)
        # self.mask_qcolor=self.imageparam.mask_color


class PlotItemBuilder(GuiPlotItemBuilder):
    def __init__(self):
        super(PlotItemBuilder, self).__init__()

    def binocurve(
        self,
        x,
        y,
        title=u"",
        color=None,
        linestyle=None,
        linewidth=None,
        marker=None,
        markersize=None,
        markerfacecolor=None,
        markeredgecolor=None,
        shade=None,
        fitted=None,
        curvestyle=None,
        curvetype=None,
        baseline=None,
        xaxis="bottom",
        yaxis="left",
    ):
        """
        Make a curve `plot item` from x, y, data
        (:py:class:`guiqwt.curve.CurveItem` object)
            * x: 1D NumPy array
            * y: 1D NumPy array
            * color: curve color name
            * linestyle: curve line style (MATLAB-like string or attribute name
              from the :py:class:`PyQt4.QtCore.Qt.PenStyle` enum
              (i.e. "SolidLine" "DashLine", "DotLine", "DashDotLine",
              "DashDotDotLine" or "NoPen")
            * linewidth: line width (pixels)
            * marker: marker shape (MATLAB-like string or attribute name from
              the :py:class:`PyQt4.Qwt5.QwtSymbol.Style` enum (i.e. "Cross",
              "Ellipse", "Star1", "XCross", "Rect", "Diamond", "UTriangle",
              "DTriangle", "RTriangle", "LTriangle", "Star2" or "NoSymbol")
            * markersize: marker size (pixels)
            * markerfacecolor: marker face color name
            * markeredgecolor: marker edge color name
            * shade: 0 <= float <= 1 (curve shade)
            * fitted: boolean (fit curve to data)
            * curvestyle: attribute name from the
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveStyle` enum
              (i.e. "Lines", "Sticks", "Steps", "Dots" or "NoCurve")
            * curvetype: attribute name from the
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveType` enum
              (i.e. "Yfx" or "Xfy")
            * baseline (float: default=0.0): the baseline is needed for filling
              the curve with a brush or the Sticks drawing style.
              The interpretation of the baseline depends on the curve type
              (horizontal line for "Yfx", vertical line for "Xfy")
            * xaxis, yaxis: X/Y axes bound to curve

        Examples:
        curve(x, y, marker='Ellipse', markerfacecolor='#ffffff')
        which is equivalent to (MATLAB-style support):
        curve(x, y, marker='o', markerfacecolor='w')
        """
        basename = _("Curve")
        param = CurveParam(title=basename, icon="curve.png")
        if not title:
            global CURVE_COUNT
            CURVE_COUNT += 1
            title = make_title(basename, CURVE_COUNT)
        self.__set_param(
            param,
            title,
            color,
            linestyle,
            linewidth,
            marker,
            markersize,
            markerfacecolor,
            markeredgecolor,
            shade,
            fitted,
            curvestyle,
            curvetype,
            baseline,
        )
        curve = BinoCurve(param)
        curve.set_data(x, y)
        curve.update_params()
        self.__set_curve_axes(curve, xaxis, yaxis)
        return curve

    def imagenan(
        self,
        data=None,
        filename=None,
        title=None,
        alpha_mask=None,
        alpha=None,
        background_color=None,
        colormap=None,
        xdata=[None, None],
        ydata=[None, None],
        pixel_size=None,
        center_on=None,
        interpolation="linear",
        eliminate_outliers=None,
        xformat="%.1f",
        yformat="%.1f",
        zformat="%.1f",
    ):
        """
        Make an image `plot item` from data
        (:py:class:`guiqwt.image.ImageItem` object or
        :py:class:`guiqwt.image.RGBImageItem` object if data has 3 dimensions)
        """
        assert isinstance(xdata, (tuple, list)) and len(xdata) == 2
        assert isinstance(ydata, (tuple, list)) and len(ydata) == 2
        param = ImageParam(title="Image", icon="image.png")
        data, filename, title = self._get_image_data(
            data, filename, title, to_grayscale=True
        )
        assert data.ndim == 2, "Data must have 2 dimensions"
        if pixel_size is None:
            assert center_on is None, (
                "Ambiguous parameters: both `center_on`"
                " and `xdata`/`ydata` were specified"
            )
            xmin, xmax = xdata
            ymin, ymax = ydata
        else:
            xmin, xmax, ymin, ymax = self.compute_bounds(data, pixel_size, center_on)
        self.__set_image_param(
            param,
            title,
            alpha_mask,
            alpha,
            interpolation,
            background=background_color,
            colormap=colormap,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            xformat=xformat,
            yformat=yformat,
            zformat=zformat,
        )

        image = MaskedImageNan(data, None, param)
        image.set_filename(filename)
        if eliminate_outliers is not None:
            image.set_lut_range(lut_range_threshold(image, 256, eliminate_outliers))
        return image

    def maskedimagenan(
        self,
        data=None,
        mask=None,
        filename=None,
        title=None,
        alpha_mask=False,
        alpha=1.0,
        xdata=[None, None],
        ydata=[None, None],
        pixel_size=None,
        center_on=None,
        background_color=None,
        colormap=None,
        show_mask=False,
        fill_value=None,
        interpolation="linear",
        eliminate_outliers=None,
        xformat="%.1f",
        yformat="%.1f",
        zformat="%.1f",
    ):
        """
        Make a masked image `plot item` from data
        (:py:class:`guiqwt.image.MaskedImageItem` object)
        """
        assert isinstance(xdata, (tuple, list)) and len(xdata) == 2
        assert isinstance(ydata, (tuple, list)) and len(ydata) == 2
        param = MaskedImageParam(title=_("Image"), icon="image.png")
        data, filename, title = self._get_image_data(
            data, filename, title, to_grayscale=True
        )
        assert data.ndim == 2, "Data must have 2 dimensions"
        if pixel_size is None:
            assert center_on is None, (
                "Ambiguous parameters: both `center_on`"
                " and `xdata`/`ydata` were specified"
            )
            xmin, xmax = xdata
            ymin, ymax = ydata
        else:
            xmin, xmax, ymin, ymax = self.compute_bounds(data, pixel_size, center_on)
        self.__set_image_param(
            param,
            title,
            alpha_mask,
            alpha,
            interpolation,
            background=background_color,
            colormap=colormap,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            show_mask=show_mask,
            fill_value=fill_value,
            xformat=xformat,
            yformat=yformat,
            zformat=zformat,
        )
        image = MaskedImageNan(data, mask, param)
        image.set_filename(filename)
        if eliminate_outliers is not None:
            image.set_lut_range(lut_range_threshold(image, 256, eliminate_outliers))
        return image


make = PlotItemBuilder()


class SlicePrefs:
    i0 = 0
    i1 = 1
    i2 = 2
    i3 = 0
    absmins = [0, 0, 0]  # les valeurs min des points
    absmaxs = [1, 1, 1]  # les valeurs max des points (borne exclue)
    mins = [0, 0, 0]  # les valeurs min de la range pour les slices
    maxs = [0, 0, 0]  # les valeurs max de la range pour les slices (borne exclue)
    steps = [1, 1, 1]
    labels = ["x", "y", "z"]
    ranges = [1.0, 1.0, 1.0]
    pos1 = [0.0, 0.0, 0.0]  # position en bas de tige
    pos2 = [0.0, 0.0, 1.0]  # position en haut de tige
    stepw = 1  # width of a slice
    stepn = 1  # number of slices
    do_it = False


class MoveHandler(DragHandler):
    def __init__(self, filter, btn, mods=Qt.NoModifier, start_state=0):
        super(MoveHandler, self).__init__(filter, btn, mods, start_state)
        self.avoid_null_shape = False

    def set_shape(self, shape, setup_shape_cb=None, avoid_null_shape=False):
        # called at initialization of the tool
        self.shape = shape
        self.setup_shape_cb = setup_shape_cb
        self.avoid_null_shape = avoid_null_shape

    def start_tracking(self, filter, event):
        self.shape.attach(filter.plot)
        self.shape.setZ(filter.plot.get_max_z() + 1)
        if self.avoid_null_shape:
            self.start -= QPoint(1, 1)
        pt = event.pos()
        self.shape.set_local_center(pt)
        if self.setup_shape_cb is not None:
            self.setup_shape_cb(self.shape)
        self.shape.show()
        filter.plot.replot()
        self.emit(SIG_START_TRACKING, filter, event)

    def stop_tracking(
        self, filter, event
    ):  # when the mouse press button is released before moving
        self.shape.detach()
        self.emit(SIG_STOP_TRACKING, filter, event)
        filter.plot.replot()

    def start_moving(self, filter, event):
        pass

    def move(self, filter, event):
        pt = event.pos()
        self.shape.set_local_center(pt)
        self.emit(SIG_MOVE, filter, event)
        filter.plot.replot()

    def stop_moving(
        self, filter, event
    ):  # when the mouse press button is released after moving
        self.emit(SIG_STOP_MOVING, filter, event)
        self.shape.detach()
        filter.plot.replot()


class RectangleCentredShape(RectangleShape):
    CLOSED = True

    def __init__(self, xc=0, yc=0, hsize=1, vsize=1, shapeparam=None):
        super(RectangleCentredShape, self).__init__(shapeparam=shapeparam)
        self.is_ellipse = False
        self.hsize = hsize
        self.vsize = vsize
        self.set_center(xc, yc)

    def set_size(self, hsize, vsize):
        self.hsize = hsize
        self.vsize = vsize
        x0, y0, x1, y1 = self.get_rect()
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        self.set_center(xc, yc)

    def set_local_center(self, pos):
        # set center from position in canvas units
        xc, yc = canvas_to_axes(self, pos)
        self.set_center(xc, yc)

    def set_center(self, xc, yc):
        x0 = xc - self.hsize / 2.0
        y0 = yc - self.vsize / 2.0
        x1 = xc + self.hsize / 2.0
        y1 = yc + self.vsize / 2.0
        self.set_rect(x0, y0, x1, y1)


class EllipseCentredShape(EllipseShape):
    CLOSED = True

    def __init__(self, xc=0, yc=0, hsize=1, vsize=1, shapeparam=None):
        super(EllipseCentredShape, self).__init__(shapeparam=shapeparam)
        self.is_ellipse = True
        self.hsize = hsize
        self.vsize = vsize
        self.set_center(xc, yc)

    def set_size(self, hsize, vsize):
        self.hsize = hsize
        self.vsize = vsize
        x0, y0, x1, y1 = self.get_rect()
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        self.set_center(xc, yc)

    def set_local_center(self, pos):
        # set center from position in canvas units
        xc, yc = canvas_to_axes(self, pos)
        self.set_center(xc, yc)

    def set_center(self, xc, yc):
        x0 = xc - self.hsize / 2.0
        y0 = yc - self.vsize / 2.0
        x1 = xc + self.hsize / 2.0
        y1 = yc + self.vsize / 2.0
        self.set_rect(x0, y0, x1, y1)


class SquareCentredShape(RectangleShape):
    CLOSED = True

    def __init__(self, xc=0, yc=0, size=1, shapeparam=None):
        super(SquareCentredShape, self).__init__(shapeparam=shapeparam)
        self.is_ellipse = False
        self.size = size
        self.set_center(xc, yc)

    def set_size(self, size):
        self.size = size
        x0, y0, x1, y1 = self.get_rect()
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        self.set_center(xc, yc)

    def set_local_center(self, pos):
        # set center from position in canvas units
        xc, yc = canvas_to_axes(self, pos)
        self.set_center(xc, yc)

    def set_center(self, xc, yc):
        x0 = xc - self.size / 2.0
        y0 = yc - self.size / 2.0
        x1 = xc + self.size / 2.0
        y1 = yc + self.size / 2.0
        self.set_rect(x0, y0, x1, y1)


class DrawingTool(InteractiveTool):
    TITLE = _("Drawing")
    ICON = "pencil.png"
    CURSOR = Qt.CrossCursor
    AVOID_NULL_SHAPE = False

    def __init__(
        self,
        manager,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title=None,
        icon=None,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(DrawingTool, self).__init__(
            manager,
            toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None

        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def reset(self):
        self.shape = None
        self.current_handle = None

    def setup_shape(self, shape):
        """To be reimplemented"""
        shape.setTitle(self.TITLE)

    def create_shape(self):
        shape = PointShape(0, 0)
        self.setup_shape(shape)
        return shape

    def set_shape_style(self, shape):
        shape.set_style(self.shape_style_sect, self.shape_style_key)

    def setup_filter(self, baseplot):
        filter = baseplot.filter
        start_state = filter.new_state()

        self.shape = self.get_shape()

        handler = MoveHandler(filter, Qt.LeftButton, start_state=start_state)
        handler.set_shape(
            self.shape, self.setup_shape, avoid_null_shape=self.AVOID_NULL_SHAPE
        )
        self.connect(handler, SIG_START_TRACKING, self.start)
        self.connect(handler, SIG_MOVE, self.move)
        self.connect(handler, SIG_STOP_NOT_MOVING, self.stop)
        self.connect(handler, SIG_STOP_MOVING, self.stop)
        self.connect(handler, SIG_STOP_TRACKING, self.stop)
        return setup_standard_tool_filter(filter, start_state)

    def get_shape(self):
        """Reimplemented RectangularActionTool method"""
        shape = self.create_shape()
        self.setup_shape(shape)
        return shape

    def validate(self, filter, event):
        super(DrawingTool, self).validate(filter, event)
        if self.handle_final_shape_cb is not None:
            self.handle_final_shape_cb(self.shape)
        self.reset()

    def start(self, filter, event):
        pass

    def move(self, filter, event):
        pass

    def stop(self, filter, event):
        pass


class CircularDrawingTool(DrawingTool):
    TITLE = _("Drawing")
    ICON = "circlebrush.png"
    size = 10

    def set_size(self, size):
        self.size = size
        if self.shape is not None:
            self.shape.set_size(self.size)

    def create_shape(self):
        shape = EllipseCentredShape(0, 0, self.size)
        self.setup_shape(shape)
        return shape


class EllipseDrawingTool(DrawingTool):
    TITLE = _("Drawing")
    ICON = "brush.jpg"
    hpixelsize = 1  # size in image pixels
    vpixelsize = 1
    hsize = 1
    vsize = 1

    def set_pixel_size(self, hpixelsize, vpixelsize=None):
        hsize = self.hsize / (self.hpixelsize - 0.5)
        vsize = self.vsize / (self.vpixelsize - 0.5)
        self.hpixelsize = hpixelsize
        if vpixelsize is None:
            self.vpixelsize = hpixelsize  # square shape
        else:
            self.vpixelsize = vpixelsize
        self.set_size(hsize, vsize)

    def set_size(self, hsize, vsize):
        # size in image coordinates
        self.hsize = hsize * (self.hpixelsize - 0.5)
        self.vsize = vsize * (self.vpixelsize - 0.5)
        if self.shape is not None:
            self.shape.set_size(self.hsize, self.vsize)

    def create_shape(self):
        shape = EllipseCentredShape(0, 0, self.hsize, self.vsize)
        self.setup_shape(shape)
        return shape


class SquareDrawingTool(DrawingTool):
    TITLE = _("Drawing")
    ICON = "brush.jpg"
    size = 10

    def set_size(self, size):
        self.size = size
        if self.shape is not None:
            self.shape.set_size(self.size)

    def create_shape(self):
        shape = SquareCentredShape(0, 0, self.size)
        self.setup_shape(shape)
        return shape


class RectangleDrawingTool(DrawingTool):
    TITLE = _("Drawing")
    ICON = "brush.jpg"
    hpixelsize = 1  # size in image pixels
    vpixelsize = 1
    hsize = 1
    vsize = 1

    def set_pixel_size(self, hpixelsize, vpixelsize=None):
        hsize = self.hsize / (self.hpixelsize - 0.5)
        vsize = self.vsize / (self.vpixelsize - 0.5)
        self.hpixelsize = hpixelsize
        if vpixelsize is None:
            self.vpixelsize = hpixelsize  # square shape
        else:
            self.vpixelsize = vpixelsize
        self.set_size(hsize, vsize)

    def set_size(self, hsize, vsize):
        # size in image coordinates
        self.hsize = hsize * (self.hpixelsize - 0.5)
        self.vsize = vsize * (self.vpixelsize - 0.5)
        if self.shape is not None:
            self.shape.set_size(self.hsize, self.vsize)

    def create_shape(self):
        shape = RectangleCentredShape(0, 0, self.hsize, self.vsize)
        self.setup_shape(shape)
        return shape


class RectangleEraserTool(RectangleDrawingTool):
    TITLE = _("Rectangle eraser")
    ICON = "eraser.png"
    CURSOR = Qt.CrossCursor
    AVOID_NULL_SHAPE = False
    SHAPE_STYLE_SECT = "plot"
    SHAPE_STYLE_KEY = "shape/drag"

    def __init__(
        self,
        manager,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title=None,
        icon=None,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(RectangleEraserTool, self).__init__(
            manager,
            toolbar_id=toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None
        self.hsize = 10
        self.vsize = 10

        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def start(self, filter, event):
        self.emit(SIGNAL("suppress_area()"))

    def stop(self, filter, event):
        self.emit(SIG_STOP_MOVING)

    def move(self, filter, event):
        """moving while holding the button down lets the user
        position the last created point
        """
        self.emit(SIGNAL("suppress_area()"))


class CircularMaskTool(CircularDrawingTool):
    TITLE = _("Circular masking brush")
    ICON = "circle_sponge.png"
    CURSOR = Qt.CrossCursor
    AVOID_NULL_SHAPE = False
    SHAPE_STYLE_SECT = "plot"
    SHAPE_STYLE_KEY = "shape/drag"

    def __init__(
        self,
        manager,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title=None,
        icon=None,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(CircularMaskTool, self).__init__(
            manager,
            toolbar_id=toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None
        self.masked_image = None  # associated masked image item
        self.size = 10
        self.mode = True

        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def set_mode(self, mode):
        # set the action mask/unmask
        self.mode = mode

    def start(self, filter, event):
        self.masked_image = self.find_masked_image(filter.plot)
        if self.masked_image is not None:
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )

            self.masked_image.plot().replot()

    def find_masked_image(self, plot):
        item = plot.get_active_item()
        if isinstance(item, MaskedImageItem):
            return item
        else:
            items = [
                item for item in plot.get_items() if isinstance(item, MaskedImageItem)
            ]
            if items:
                return items[-1]

    def move(self, filter, event):
        """moving while holding the button down lets the user
        position the last created point
        """
        if self.masked_image is not None:
            # mask = self.masked_image.get_mask()
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            self.masked_image.plot().replot()
        """
                x0, y0, x1, y1 = shape.get_rect()
                self.masked_image.mask_circular_area(x0, y0, x1, y1,inside=inside)
        """


class EllipseMaskTool(EllipseDrawingTool):
    TITLE = _("Ellipse masking brush")
    ICON = "circle_sponge.png"
    CURSOR = Qt.CrossCursor
    AVOID_NULL_SHAPE = False
    SHAPE_STYLE_SECT = "plot"
    SHAPE_STYLE_KEY = "shape/drag"

    def __init__(
        self,
        manager,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title=None,
        icon=None,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(EllipseMaskTool, self).__init__(
            manager,
            toolbar_id=toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None
        self.masked_image = None  # associated masked image item
        self.mode = True

        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def set_mode(self, mode):
        # set the action mask/unmask
        self.mode = mode

    def start(self, filter, event):
        self.masked_image = self.find_masked_image(filter.plot)
        if self.masked_image is not None:
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_circular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )

            self.masked_image.plot().replot()

    def find_masked_image(self, plot):
        item = plot.get_active_item()
        if isinstance(item, MaskedImageItem):
            return item
        else:
            items = [
                item for item in plot.get_items() if isinstance(item, MaskedImageItem)
            ]
            if items:
                return items[-1]

    def move(self, filter, event):
        """moving while holding the button down lets the user
        position the last created point
        """
        if self.masked_image is not None:
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            self.masked_image.plot().replot()


class RectangleMaskTool(RectangleDrawingTool):
    TITLE = _("Square masking brush")
    ICON = "square_sponge.png"
    CURSOR = Qt.CrossCursor
    AVOID_NULL_SHAPE = False
    SHAPE_STYLE_SECT = "plot"
    SHAPE_STYLE_KEY = "shape/drag"

    def __init__(
        self,
        manager,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title=None,
        icon=None,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(RectangleMaskTool, self).__init__(
            manager,
            toolbar_id=toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None
        self.masked_image = None  # associated masked image item
        self.mode = True

        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def set_mode(self, mode):
        # set the action mask/unmask
        self.mode = mode

    def start(self, filter, event):
        self.masked_image = self.find_masked_image(filter.plot)
        if self.masked_image is not None:
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )

            self.masked_image.plot().replot()

    def find_masked_image(self, plot):
        item = plot.get_active_item()
        if isinstance(item, MaskedImageItem):
            return item
        else:
            items = [
                item for item in plot.get_items() if isinstance(item, MaskedImageItem)
            ]
            if items:
                return items[-1]

    def move(self, filter, event):
        """moving while holding the button down lets the user
        position the last created point
        """
        if self.masked_image is not None:
            x0, y0, x1, y1 = self.shape.get_rect()
            if self.mode:
                self.masked_image.mask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            else:
                self.masked_image.unmask_rectangular_area(
                    x0, y0, x1, y1, trace=False, inside=True
                )
            self.masked_image.plot().replot()


class SetSliceWindow(QDialog):
    # definit une fenetre pour rentrer les parametres de dialogue de construction de map 3D
    def __init__(self, prefs):
        self.prefs = prefs
        super(
            SetSliceWindow, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.setWindowTitle("Parameters for slice generation")
        self.setFixedSize(QSize(450, 160))
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.lab0 = QLabel("Direction", self)
        self.lab1 = QLabel("Slice", self)
        self.lab2 = QLabel("Scan", self)
        self.lab3 = QLabel("Raw sum", self)
        self.lab4 = QLabel("Min (incl.)", self)
        self.lab5 = QLabel("Max (excl.)", self)

        self.xlab = QLabel(prefs.labels[0], self)
        self.ylab = QLabel(prefs.labels[1], self)
        self.zlab = QLabel(prefs.labels[2], self)

        self.lab6 = QLabel("Steps", self)
        self.lab7 = QLabel("Width", self)

        self.xbox1 = QCheckBox(self)
        self.ybox1 = QCheckBox(self)
        self.zbox1 = QCheckBox(self)
        self.group1 = QButtonGroup(self)
        self.group1.addButton(self.xbox1, 1)
        self.group1.addButton(self.ybox1, 2)
        self.group1.addButton(self.zbox1, 3)
        self.group1.button(prefs.i1 + 1).setChecked(True)

        self.xbox2 = QCheckBox(self)
        self.ybox2 = QCheckBox(self)
        self.zbox2 = QCheckBox(self)
        self.group2 = QButtonGroup(self)
        self.group2.addButton(self.xbox2, 1)
        self.group2.addButton(self.ybox2, 2)
        self.group2.addButton(self.zbox2, 3)
        self.group2.button(prefs.i2 + 1).setChecked(True)

        self.xbox3 = QCheckBox(self)
        self.ybox3 = QCheckBox(self)
        self.zbox3 = QCheckBox(self)
        self.group3 = QButtonGroup(self)
        self.group3.addButton(self.xbox3, 1)
        self.group3.addButton(self.ybox3, 2)
        self.group3.addButton(self.zbox3, 3)
        self.group3.button(prefs.i3 + 1).setChecked(True)

        self.xmin = QDoubleSpinBox(self)
        self.xmin.setRange(prefs.absmins[0], prefs.absmaxs[0])
        self.xmin.setSingleStep(prefs.steps[0])
        self.xmin.setDecimals(max(2, int(1 - np.log10(prefs.steps[0]))))
        self.xmin.setValue(prefs.mins[0])
        self.xmax = QDoubleSpinBox(self)
        self.xmax.setRange(prefs.absmins[0], prefs.absmaxs[0])
        self.xmax.setSingleStep(prefs.steps[0])
        self.xmax.setDecimals(max(2, int(1 - np.log10(prefs.steps[0]))))
        self.xmax.setValue(prefs.maxs[0])

        self.ymin = QDoubleSpinBox(self)
        self.ymin.setRange(prefs.absmins[1], prefs.absmaxs[1])
        self.ymin.setSingleStep(prefs.steps[1])
        self.ymin.setDecimals(max(2, int(1 - np.log10(prefs.steps[1]))))
        self.ymin.setValue(prefs.mins[1])
        self.ymax = QDoubleSpinBox(self)
        self.ymax.setRange(prefs.absmins[1], prefs.absmaxs[1])
        self.ymax.setSingleStep(prefs.steps[1])
        self.ymax.setDecimals(max(2, int(1 - np.log10(prefs.steps[1]))))
        self.ymax.setValue(prefs.maxs[1])

        self.zmin = QDoubleSpinBox(self)
        self.zmin.setRange(prefs.absmins[2], prefs.absmaxs[2])
        self.zmin.setSingleStep(prefs.steps[2])
        self.zmin.setDecimals(max(2, int(1 - np.log10(prefs.steps[2]))))
        self.zmin.setValue(prefs.mins[2])
        self.zmax = QDoubleSpinBox(self)
        self.zmax.setRange(prefs.absmins[2], prefs.absmaxs[2])
        self.zmax.setSingleStep(prefs.steps[2])
        self.zmax.setDecimals(max(2, int(1 - np.log10(prefs.steps[2]))))
        self.zmax.setValue(prefs.maxs[2])

        stepn = int(
            (prefs.maxs[prefs.i1] - prefs.mins[prefs.i1]) / prefs.steps[prefs.i1]
        )
        if stepn == 0:
            stepn = 1
        stepw = (prefs.maxs[prefs.i1] - prefs.mins[prefs.i1]) / stepn

        self.stepn = QSpinBox(self)
        self.stepn.setMinimum(1)
        self.stepn.setMaximum(stepn)
        self.stepn.setValue(stepn)

        self.stepw = QLineEdit(self)
        self.stepw.setText("%f" % (stepw))

        self.OK = QPushButton(self)
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setText("Cancel")

        self.layout.addWidget(self.lab0, 0, 0)
        self.layout.addWidget(self.lab1, 0, 1)
        self.layout.addWidget(self.lab2, 0, 2)
        self.layout.addWidget(self.lab3, 0, 3)
        self.layout.addWidget(self.lab4, 0, 4)
        self.layout.addWidget(self.lab5, 0, 5)

        self.layout.addWidget(self.xlab, 1, 0)
        self.layout.addWidget(self.xbox1, 1, 1)
        self.layout.addWidget(self.xbox2, 1, 2)
        self.layout.addWidget(self.xbox3, 1, 3)
        self.layout.addWidget(self.xmin, 1, 4)
        self.layout.addWidget(self.xmax, 1, 5)

        self.layout.addWidget(self.ylab, 2, 0)
        self.layout.addWidget(self.ybox1, 2, 1)
        self.layout.addWidget(self.ybox2, 2, 2)
        self.layout.addWidget(self.ybox3, 2, 3)
        self.layout.addWidget(self.ymin, 2, 4)
        self.layout.addWidget(self.ymax, 2, 5)

        self.layout.addWidget(self.zlab, 3, 0)
        self.layout.addWidget(self.zbox1, 3, 1)
        self.layout.addWidget(self.zbox2, 3, 2)
        self.layout.addWidget(self.zbox3, 3, 3)
        self.layout.addWidget(self.zmin, 3, 4)
        self.layout.addWidget(self.zmax, 3, 5)

        self.layout.addWidget(self.lab6, 4, 0)
        self.layout.addWidget(self.stepn, 4, 1)
        self.layout.addWidget(self.lab7, 4, 2)
        self.layout.addWidget(self.stepw, 4, 3)
        self.layout.addWidget(self.OK, 4, 4)
        self.layout.addWidget(self.Cancel, 4, 5)

        for i in range(1, 4):
            self.layout.setColumnMinimumWidth(i, 80)
        for i in range(6):
            self.layout.setColumnStretch(i, 1)

        QObject.connect(self.Cancel, SIGNAL("clicked()"), self.closewin)
        QObject.connect(self.OK, SIGNAL("clicked()"), self.appl)
        for button in [
            self.xbox1,
            self.ybox1,
            self.zbox1,
            self.xbox2,
            self.ybox2,
            self.zbox2,
            self.xbox3,
            self.ybox3,
            self.zbox3,
        ]:
            QObject.connect(button, SIGNAL("clicked()"), self.validate)
        for button in [
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
        ]:
            QObject.connect(button, SIGNAL("valueChanged(double)"), self.round_values)
        QObject.connect(
            self.stepn, SIGNAL("valueChanged(int)"), self.compute_step_width
        )
        QObject.connect(
            self.stepw, SIGNAL("textEdited(QString)"), self.compute_step_number
        )  # only when user change the text!

        QObject.connect(self.Cancel, SIGNAL("clicked()"), self.closewin)

        self.exec_()

    def round_values(self, x):
        i1 = round((self.xmin.value() - self.prefs.absmins[0]) / self.prefs.steps[0])
        i2 = round((self.xmax.value() - self.prefs.absmins[0]) / self.prefs.steps[0])
        self.xmin.setValue(self.prefs.absmins[0] + i1 * self.prefs.steps[0])
        self.xmax.setValue(self.prefs.absmins[0] + i2 * self.prefs.steps[0])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        i1 = round((self.ymin.value() - self.prefs.absmins[1]) / self.prefs.steps[1])
        i2 = round((self.ymax.value() - self.prefs.absmins[1]) / self.prefs.steps[1])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        self.ymin.setValue(self.prefs.absmins[1] + i1 * self.prefs.steps[1])
        self.ymax.setValue(self.prefs.absmins[1] + i2 * self.prefs.steps[1])

        i1 = round((self.zmin.value() - self.prefs.absmins[2]) / self.prefs.steps[2])
        i2 = round((self.zmax.value() - self.prefs.absmins[2]) / self.prefs.steps[2])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        self.zmin.setValue(self.prefs.absmins[2] + i1 * self.prefs.steps[2])
        self.zmax.setValue(self.prefs.absmins[2] + i2 * self.prefs.steps[2])

    def compute_step_number(self, text):
        i1 = self.group1.checkedId() - 1
        try:
            mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
            maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]
            stepw = abs(float(self.stepw.text()))
            stepn = int((maxs[i1] - mins[i1]) / stepw)
            stepw = (maxs[i1] - mins[i1]) / stepn
            self.stepn.setValue(stepn)
            self.stepw.setText("%f" % (stepw))
        except Exception:
            print("problem in compute_step_number")
            pass

    def compute_step_width(self, ii):
        i1 = self.group1.checkedId() - 1
        try:
            mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
            maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]
            stepn = self.stepn.value()
            stepw = (maxs[i1] - mins[i1]) / stepn
            self.stepw.setText("%f" % (stepw))
        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a number")
            return

    def validate(self):
        i1 = self.group1.checkedId() - 1
        i2 = self.group2.checkedId() - 1
        if i2 == i1:
            i2 = (i1 + 1) % 3
            self.group2.button(i2 + 1).setChecked(True)
        i3 = 3 - (i1 + i2)
        self.group3.button(i3 + 1).setChecked(True)

        mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
        maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]

        stepn = int((maxs[i1] - mins[i1]) / self.prefs.steps[i1])
        if stepn == 0:
            stepn = 1
        stepw = (maxs[i1] - mins[i1]) / stepn

        self.stepn.setMaximum(stepn)
        self.stepn.setValue(stepn)

        self.stepw.setText("%f" % (stepw))

    def appl(self):
        self.prefs.i1 = self.group1.checkedId() - 1
        self.prefs.i2 = self.group2.checkedId() - 1
        self.prefs.i3 = self.group3.checkedId() - 1

        try:
            self.prefs.mins[0] = self.xmin.value()
            self.prefs.mins[1] = self.ymin.value()
            self.prefs.mins[2] = self.zmin.value()
            self.prefs.maxs[0] = self.xmax.value()
            self.prefs.maxs[1] = self.ymax.value()
            self.prefs.maxs[2] = self.zmax.value()

            self.prefs.stepn = self.stepn.value()
            self.prefs.stepw = (
                self.prefs.maxs[self.prefs.i1] - self.prefs.mins[self.prefs.i1]
            ) / self.prefs.stepn

        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a number")
            return

        if (
            self.prefs.mins[0] >= self.prefs.maxs[0]
            or self.prefs.mins[1] >= self.prefs.maxs[1]
            or self.prefs.mins[2] >= self.prefs.maxs[2]
        ):
            QMessageBox.about(
                self, "Error", "Minimum values must be lower than maximum ones"
            )
            return

        self.close()
        self.prefs.do_it = True

    def closewin(self):
        self.close()
        self.prefs.do_it = False


class Set2DSliceWindow(QDialog):
    # definit une fenetre pour rentrer les parametres de dialogue de construction d'une serie de slices
    def __init__(self, prefs):
        self.prefs = prefs
        super(
            Set2DSliceWindow, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.setWindowTitle("Parameters for slice generation")
        self.setFixedSize(QSize(450, 160))
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.lab0 = QLabel("Direction", self)
        self.lab1 = QLabel("Slice", self)
        self.lab4 = QLabel("Min (incl.)", self)
        self.lab5 = QLabel("Max (excl.)", self)

        self.xlab = QLabel(prefs.labels[0], self)
        self.ylab = QLabel(prefs.labels[1], self)
        self.zlab = QLabel(prefs.labels[2], self)

        self.lab6 = QLabel("Steps", self)
        self.lab7 = QLabel("Width", self)

        self.xbox1 = QCheckBox(self)
        self.ybox1 = QCheckBox(self)
        self.zbox1 = QCheckBox(self)
        self.group1 = QButtonGroup(self)
        self.group1.addButton(self.xbox1, 1)
        self.group1.addButton(self.ybox1, 2)
        self.group1.addButton(self.zbox1, 3)
        self.group1.button(prefs.i1 + 1).setChecked(True)

        self.xmin = QDoubleSpinBox(self)
        self.xmin.setRange(prefs.absmins[0], prefs.absmaxs[0])
        self.xmin.setSingleStep(prefs.steps[0])
        self.xmin.setDecimals(max(2, int(1 - np.log10(prefs.steps[0]))))
        self.xmin.setValue(prefs.mins[0])
        self.xmax = QDoubleSpinBox(self)
        self.xmax.setRange(prefs.absmins[0], prefs.absmaxs[0])
        self.xmax.setSingleStep(prefs.steps[0])
        self.xmax.setDecimals(max(2, int(1 - np.log10(prefs.steps[0]))))
        self.xmax.setValue(prefs.maxs[0])

        self.ymin = QDoubleSpinBox(self)
        self.ymin.setRange(prefs.absmins[1], prefs.absmaxs[1])
        self.ymin.setSingleStep(prefs.steps[1])
        self.ymin.setDecimals(max(2, int(1 - np.log10(prefs.steps[1]))))
        self.ymin.setValue(prefs.mins[1])
        self.ymax = QDoubleSpinBox(self)
        self.ymax.setRange(prefs.absmins[1], prefs.absmaxs[1])
        self.ymax.setSingleStep(prefs.steps[1])
        self.ymax.setDecimals(max(2, int(1 - np.log10(prefs.steps[1]))))
        self.ymax.setValue(prefs.maxs[1])

        self.zmin = QDoubleSpinBox(self)
        self.zmin.setRange(prefs.absmins[2], prefs.absmaxs[2])
        self.zmin.setSingleStep(prefs.steps[2])
        self.zmin.setDecimals(max(2, int(1 - np.log10(prefs.steps[2]))))
        self.zmin.setValue(prefs.mins[2])
        self.zmax = QDoubleSpinBox(self)
        self.zmax.setRange(prefs.absmins[2], prefs.absmaxs[2])
        self.zmax.setSingleStep(prefs.steps[2])
        self.zmax.setDecimals(max(2, int(1 - np.log10(prefs.steps[2]))))
        self.zmax.setValue(prefs.maxs[2])

        stepn = int(
            (prefs.maxs[prefs.i1] - prefs.mins[prefs.i1]) / prefs.steps[prefs.i1]
        )
        if stepn == 0:
            stepn = 1
        stepw = (prefs.maxs[prefs.i1] - prefs.mins[prefs.i1]) / stepn

        self.stepn = QSpinBox(self)
        self.stepn.setMinimum(1)
        self.stepn.setMaximum(stepn)
        self.stepn.setValue(stepn)

        self.stepw = QLineEdit(self)
        self.stepw.setText("%f" % (stepw))

        self.OK = QPushButton(self)
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setText("Cancel")

        self.layout.addWidget(self.lab0, 0, 0)
        self.layout.addWidget(self.lab1, 0, 1)
        self.layout.addWidget(self.lab4, 0, 4)
        self.layout.addWidget(self.lab5, 0, 5)

        self.layout.addWidget(self.xlab, 1, 0)
        self.layout.addWidget(self.xbox1, 1, 1)
        self.layout.addWidget(self.xmin, 1, 4)
        self.layout.addWidget(self.xmax, 1, 5)

        self.layout.addWidget(self.ylab, 2, 0)
        self.layout.addWidget(self.ybox1, 2, 1)
        self.layout.addWidget(self.ymin, 2, 4)
        self.layout.addWidget(self.ymax, 2, 5)

        self.layout.addWidget(self.zlab, 3, 0)
        self.layout.addWidget(self.zbox1, 3, 1)
        self.layout.addWidget(self.zmin, 3, 4)
        self.layout.addWidget(self.zmax, 3, 5)

        self.layout.addWidget(self.lab6, 4, 0)
        self.layout.addWidget(self.stepn, 4, 1)
        self.layout.addWidget(self.lab7, 4, 2)
        self.layout.addWidget(self.stepw, 4, 3)
        self.layout.addWidget(self.OK, 4, 4)
        self.layout.addWidget(self.Cancel, 4, 5)

        for i in range(1, 4):
            self.layout.setColumnMinimumWidth(i, 80)
        for i in range(6):
            self.layout.setColumnStretch(i, 1)

        QObject.connect(self.Cancel, SIGNAL("clicked()"), self.closewin)
        QObject.connect(self.OK, SIGNAL("clicked()"), self.appl)
        for button in [self.xbox1, self.ybox1, self.zbox1]:
            QObject.connect(button, SIGNAL("clicked()"), self.validate)
        for button in [
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
        ]:
            QObject.connect(button, SIGNAL("valueChanged(double)"), self.round_values)
        QObject.connect(
            self.stepn, SIGNAL("valueChanged(int)"), self.compute_step_width
        )
        QObject.connect(
            self.stepw, SIGNAL("textEdited(QString)"), self.compute_step_number
        )  # only when user change the text!

        QObject.connect(self.Cancel, SIGNAL("clicked()"), self.closewin)

        self.exec_()

    def round_values(self, x):
        i1 = round((self.xmin.value() - self.prefs.absmins[0]) / self.prefs.steps[0])
        i2 = round((self.xmax.value() - self.prefs.absmins[0]) / self.prefs.steps[0])
        self.xmin.setValue(self.prefs.absmins[0] + i1 * self.prefs.steps[0])
        self.xmax.setValue(self.prefs.absmins[0] + i2 * self.prefs.steps[0])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        i1 = round((self.ymin.value() - self.prefs.absmins[1]) / self.prefs.steps[1])
        i2 = round((self.ymax.value() - self.prefs.absmins[1]) / self.prefs.steps[1])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        self.ymin.setValue(self.prefs.absmins[1] + i1 * self.prefs.steps[1])
        self.ymax.setValue(self.prefs.absmins[1] + i2 * self.prefs.steps[1])

        i1 = round((self.zmin.value() - self.prefs.absmins[2]) / self.prefs.steps[2])
        i2 = round((self.zmax.value() - self.prefs.absmins[2]) / self.prefs.steps[2])
        if i1 >= i2:
            if i2 == 0:
                i1 = 0
                i2 = 1
            else:
                i1 = i2 - 1
        self.zmin.setValue(self.prefs.absmins[2] + i1 * self.prefs.steps[2])
        self.zmax.setValue(self.prefs.absmins[2] + i2 * self.prefs.steps[2])

    def compute_step_number(self, text):
        i1 = self.group1.checkedId() - 1
        try:
            mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
            maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]
            stepw = abs(float(self.stepw.text()))
            stepn = int((maxs[i1] - mins[i1]) / stepw)
            stepw = (maxs[i1] - mins[i1]) / stepn
            self.stepn.setValue(stepn)
            self.stepw.setText("%f" % (stepw))
        except Exception:
            print("problem in compute_step_number")
            pass

    def compute_step_width(self, ii):
        i1 = self.group1.checkedId() - 1
        try:
            mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
            maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]
            stepn = self.stepn.value()
            stepw = (maxs[i1] - mins[i1]) / stepn
            self.stepw.setText("%f" % (stepw))
        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a number")
            return

    def validate(self):
        i1 = self.group1.checkedId() - 1

        mins = [self.xmin.value(), self.ymin.value(), self.zmin.value()]
        maxs = [self.xmax.value(), self.ymax.value(), self.zmax.value()]

        stepn = int((maxs[i1] - mins[i1]) / self.prefs.steps[i1])
        if stepn == 0:
            stepn = 1
        stepw = (maxs[i1] - mins[i1]) / stepn

        self.stepn.setMaximum(stepn)
        self.stepn.setValue(stepn)

        self.stepw.setText("%f" % (stepw))

    def appl(self):
        self.prefs.i1 = self.group1.checkedId() - 1

        try:
            self.prefs.mins[0] = self.xmin.value()
            self.prefs.mins[1] = self.ymin.value()
            self.prefs.mins[2] = self.zmin.value()
            self.prefs.maxs[0] = self.xmax.value()
            self.prefs.maxs[1] = self.ymax.value()
            self.prefs.maxs[2] = self.zmax.value()

            self.prefs.stepn = self.stepn.value()
            self.prefs.stepw = (
                self.prefs.maxs[self.prefs.i1] - self.prefs.mins[self.prefs.i1]
            ) / self.prefs.stepn

        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a number")
            return

        if (
            self.prefs.mins[0] >= self.prefs.maxs[0]
            or self.prefs.mins[1] >= self.prefs.maxs[1]
            or self.prefs.mins[2] >= self.prefs.maxs[2]
        ):
            QMessageBox.about(
                self, "Error", "Minimum values must be lower than maximum ones"
            )
            return

        self.close()
        self.prefs.do_it = True

    def closewin(self):
        self.close()
        self.prefs.do_it = False


class SetSliceWindow2(QDialog):
    # definit une fenetre pour rentrer les parametres de dialogue de construction de slice le long d'une rod
    def __init__(self, prefs):
        self.prefs = prefs
        super(
            SetSliceWindow2, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.setWindowTitle("Parameters for slice generation")
        # self.setFixedSize(QSize(450, 160))
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.lab01 = QLabel("Direction", self)
        self.lab02 = QLabel("Range/width", self)
        self.lab03 = QLabel("Start", self)
        self.lab04 = QLabel("Stop", self)
        self.lab05 = QLabel("Steps", self)

        self.lab10 = QLabel("Slice", self)
        self.lab20 = QLabel("Scan", self)
        self.lab30 = QLabel("Raw sum", self)

        self.lab41 = QLabel(prefs.labels[0], self)
        self.lab42 = QLabel(prefs.labels[1], self)
        self.lab43 = QLabel(prefs.labels[2], self)

        self.lab50 = QLabel("Pos. 1", self)
        self.lab60 = QLabel("Pos. 2", self)

        self.combo11 = QComboBox(self)
        self.combo21 = QComboBox(self)
        self.combo31 = QComboBox(self)

        for combo in [self.combo11, self.combo21, self.combo31]:
            combo.addItem(prefs.labels[0])
            combo.addItem(prefs.labels[1])
            combo.addItem(prefs.labels[2])

        self.box12 = QLineEdit(self)
        self.box22 = QDoubleSpinBox(self)
        self.box32 = QDoubleSpinBox(self)

        self.box13 = QDoubleSpinBox(self)  # start
        self.box14 = QDoubleSpinBox(self)  # stop
        self.box15 = QSpinBox(self)  # steps

        self.box15.setMinimum(1)

        self.entry51 = QLineEdit(self)
        self.entry52 = QLineEdit(self)
        self.entry53 = QLineEdit(self)
        self.entry61 = QLineEdit(self)
        self.entry62 = QLineEdit(self)
        self.entry63 = QLineEdit(self)

        self.OK = QPushButton(self)
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setText("Cancel")

        self.layout.addWidget(self.lab01, 0, 1)
        self.layout.addWidget(self.lab02, 0, 2)
        self.layout.addWidget(self.lab03, 0, 3)
        self.layout.addWidget(self.lab04, 0, 4)
        self.layout.addWidget(self.lab05, 0, 5)
        self.layout.addWidget(self.lab10, 1, 0)
        self.layout.addWidget(self.lab20, 2, 0)
        self.layout.addWidget(self.lab30, 3, 0)
        self.layout.addWidget(self.lab41, 4, 1)
        self.layout.addWidget(self.lab42, 4, 2)
        self.layout.addWidget(self.lab43, 4, 3)
        self.layout.addWidget(self.lab50, 5, 0)
        self.layout.addWidget(self.lab60, 6, 0)

        self.layout.addWidget(self.combo11, 1, 1)
        self.layout.addWidget(self.combo21, 2, 1)
        self.layout.addWidget(self.combo31, 3, 1)

        self.layout.addWidget(self.box12, 1, 2)
        self.layout.addWidget(self.box22, 2, 2)
        self.layout.addWidget(self.box32, 3, 2)
        self.layout.addWidget(self.box13, 1, 3)
        self.layout.addWidget(self.box14, 1, 4)
        self.layout.addWidget(self.box15, 1, 5)

        self.layout.addWidget(self.entry51, 5, 1)
        self.layout.addWidget(self.entry52, 5, 2)
        self.layout.addWidget(self.entry53, 5, 3)
        self.layout.addWidget(self.entry61, 6, 1)
        self.layout.addWidget(self.entry62, 6, 2)
        self.layout.addWidget(self.entry63, 6, 3)

        self.layout.addWidget(self.OK, 7, 0)
        self.layout.addWidget(self.Cancel, 7, 1)

        self.combo11.setCurrentIndex(prefs.i2)
        self.combo21.setCurrentIndex(prefs.i1)
        self.combo31.setCurrentIndex(prefs.i0)

        self.set_slice_direction(prefs.i2)
        self.set_scan_direction(prefs.i1)
        self.set_raw_direction(prefs.i0)

        self.entry51.setText("%f" % ((self.prefs.mins[0] + self.prefs.maxs[0]) / 2.0))
        self.entry61.setText("%f" % ((self.prefs.mins[0] + self.prefs.maxs[0]) / 2.0))
        self.entry52.setText("%f" % ((self.prefs.mins[1] + self.prefs.maxs[1]) / 2.0))
        self.entry62.setText("%f" % ((self.prefs.mins[1] + self.prefs.maxs[1]) / 2.0))
        self.entry53.setText("%f" % (self.prefs.mins[2]))
        self.entry63.setText("%f" % (self.prefs.maxs[2]))

        for i in range(1, 4):
            self.layout.setColumnMinimumWidth(i, 80)
        for i in range(6):
            self.layout.setColumnStretch(i, 1)

        QObject.connect(self.Cancel, SIGNAL("clicked()"), self.closewin)
        QObject.connect(self.OK, SIGNAL("clicked()"), self.appl)
        QObject.connect(
            self.combo11, SIGNAL("activated(int)"), self.set_slice_direction
        )
        QObject.connect(self.combo21, SIGNAL("activated(int)"), self.set_scan_direction)
        QObject.connect(self.combo31, SIGNAL("activated(int)"), self.set_raw_direction)
        for box in [self.box13, self.box14]:
            QObject.connect(box, SIGNAL("valueChanged(double)"), self.round_values)
        QObject.connect(
            self.box12, SIGNAL("textEdited(QString)"), self.set_steps
        )  # only when user change the text!
        QObject.connect(self.box15, SIGNAL("valueChanged(int)"), self.set_width)

        self.exec_()

    def set_steps(self, x):
        i2 = self.combo11.currentIndex()
        width = float(self.box12.text())
        if width < self.prefs.steps[i2]:
            width = self.prefs.steps[i2]

        vmin = self.box13.value()
        vmax = self.box14.value()
        step = int((vmax - vmin) / width)
        if step < 0:
            step = 1
        width = (vmax - vmin) / step

        self.box15.setValue(step)
        self.box12.setText("%f" % width)

    def set_width(self, x):
        i2 = self.combo11.currentIndex()
        vmin = self.box13.value()
        vmax = self.box14.value()
        step = self.box15.value()
        width = (vmax - vmin) / step

        if width < self.prefs.steps[i2]:
            step = int((vmax - vmin) / self.prefs.steps[i2])
            width = (vmax - vmin) / step
            self.box15.setValue(step)

        self.box15.setValue(step)
        self.box12.setText("%f" % width)

    def round_values(self, x):
        i2 = self.combo11.currentIndex()
        x1 = (self.box13.value() - self.prefs.absmins[i2]) / self.prefs.steps[i2]
        x2 = (self.box14.value() - self.prefs.absmins[i2]) / self.prefs.steps[i2]

        j1 = round(x1)
        j2 = round(x2)
        if j1 >= j2:
            if j2 == 0:
                j1 = 0
                j2 = 1
            else:
                j1 = j2 - 1

        vmin = self.prefs.absmins[i2] + j1 * self.prefs.steps[i2]
        vmax = self.prefs.absmins[i2] + j2 * self.prefs.steps[i2]
        self.box13.setValue(vmin)
        self.box14.setValue(vmax)
        width = (vmax - vmin) / self.box15.value()

        if width < self.prefs.steps[i2]:
            step = int((vmax - vmin) / self.prefs.steps[i2])
            width = (vmax - vmin) / step
            self.box15.setValue(step)

        self.box12.setText("%f" % width)

    def set_slice_direction(self, i2):
        # called when a combobox has been changed
        stepmax = int(
            (self.prefs.absmaxs[i2] - self.prefs.absmins[i2]) / self.prefs.steps[i2]
        )
        self.box15.setMaximum(stepmax)
        step = int(
            (self.prefs.absmaxs[i2] - self.prefs.absmins[i2]) / self.prefs.steps[i2]
        )
        self.box15.setValue(step)

        self.box13.setRange(self.prefs.absmins[i2], self.prefs.absmaxs[i2])
        self.box13.setValue(self.prefs.mins[i2])
        self.box13.setSingleStep(self.prefs.steps[i2])
        self.box13.setDecimals(max(2, int(1 - np.log10(self.prefs.steps[i2]))))

        self.box14.setRange(self.prefs.absmins[i2], self.prefs.absmaxs[i2])
        self.box14.setValue(self.prefs.maxs[i2])
        self.box14.setSingleStep(self.prefs.steps[i2])
        self.box14.setDecimals(max(2, int(1 - np.log10(self.prefs.steps[i2]))))

        self.box12.setText("%f" % self.prefs.steps[i2])

    def set_scan_direction(self, i1):
        self.box22.setRange(0, self.prefs.absmaxs[i1] - self.prefs.absmins[i1])
        self.box22.setSingleStep(self.prefs.steps[i1])
        self.box22.setDecimals(max(2, int(1 - np.log10(self.prefs.steps[i1]))))
        self.box22.setValue(self.prefs.maxs[i1] - self.prefs.mins[i1])

    def set_raw_direction(self, i0):
        self.box32.setRange(0, self.prefs.absmaxs[i0] - self.prefs.absmins[i0])
        self.box32.setSingleStep(self.prefs.steps[i0])
        self.box32.setDecimals(max(2, int(1 - np.log10(self.prefs.steps[i0]))))
        self.box32.setValue(self.prefs.maxs[i0] - self.prefs.mins[i0])

    def appl(self):

        self.prefs.i2 = self.combo11.currentIndex()  # slice integration
        self.prefs.i1 = self.combo21.currentIndex()  # scan direction
        self.prefs.i0 = self.combo31.currentIndex()  # raw sum direction

        try:
            self.prefs.ranges[2] = float(self.box12.text())
            self.prefs.ranges[1] = self.box22.value()
            self.prefs.ranges[0] = self.box32.value()
            self.prefs.stepn = self.box15.value()
            self.prefs.mins[2] = self.box13.value()
            self.prefs.maxs[2] = self.box14.value()

            self.prefs.pos1[0] = float(self.entry51.text())
            self.prefs.pos1[1] = float(self.entry52.text())
            self.prefs.pos1[2] = float(self.entry53.text())

            self.prefs.pos2[0] = float(self.entry61.text())
            self.prefs.pos2[1] = float(self.entry62.text())
            self.prefs.pos2[2] = float(self.entry63.text())

        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a number")
            return

        if self.prefs.mins[2] >= self.prefs.maxs[2]:
            QMessageBox.about(
                self, "Error", "Minimum values must be lower than maximum ones"
            )
            return

        self.close()
        print(self.prefs.stepn)
        self.prefs.do_it = True

    def closewin(self):
        self.close()
        self.prefs.do_it = False


class FitTool(CommandTool):
    def __init__(
        self, manager, title=None, icon=None, tip=None, toolbar_id=DefaultToolbarID
    ):
        if title == None:
            title = "Fit curve"
        if icon == None:
            icon = get_icon("curve.png")
        super(FitTool, self).__init__(manager, title, icon, toolbar_id=toolbar_id)

    def activate_command(self, plot, checked):
        """Activate tool"""
        self.emit(SIG_VALIDATE_TOOL)


class RunTool(CommandTool):
    def __init__(
        self, manager, title=None, icon=None, tip=None, toolbar_id=DefaultToolbarID
    ):
        if title == None:
            title = "Apply"
        if icon == None:
            icon = get_icon("apply.png")
        super(RunTool, self).__init__(manager, title, icon, toolbar_id=toolbar_id)

    def setup_context_menu(self, menu, plot):
        pass

    def activate_command(self, plot, checked):
        """Activate tool"""
        self.emit(SIG_VALIDATE_TOOL)


# in this module we define a widget for showing the progression of calculations
class ProgressBar(QWidget):
    def __init__(self, title):
        QWidget.__init__(self)
        self.setWindowTitle(title)
        self.setFixedSize(QSize(200, 80))

        self.progressbar = QProgressBar(self)
        self.progressbar.setGeometry(10, 10, 180, 30)

        self.cancelbtn = QPushButton("Cancel", self)
        self.cancelbtn.setGeometry(10, 40, 100, 30)

        QObject.connect(self.cancelbtn, SIGNAL("clicked()"), self.cancelwin)
        self.stop = False

    def cancelwin(self):
        self.stop = True

    def update_progress(self, x):
        # print self.progressbar.value()
        self.progressbar.setValue(x * 100)
        qApp.processEvents()


class IntSpinSliderBox(QWidget):
    # a convenient widget that combines a slider and an integer  spinbox
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.label = QLabel(self)
        self.slider = QSlider(self)
        self.slider.setOrientation(0x1)
        self.spinbox = QSpinBox(self)

        self.setRange(0, 100)
        self.setSingleStep(1)

        hBox = QHBoxLayout(self)
        hBox.addWidget(self.label)
        hBox.addWidget(self.slider)
        hBox.addWidget(self.spinbox)

        self.connect(self.spinbox, SIG_INT_VALUE_CHANGED, self.update_from_spinbox)
        self.connect(self.slider, SIG_INT_VALUE_CHANGED, self.update_from_slider)

    def setText(self, text):
        self.label.setText(text)

    def setRange(self, _min, _max):
        self.spinbox.setRange(_min, _max)
        self.slider.setMinimum(_min)
        self.slider.setMaximum(_max)

    def setSingleStep(self, step):
        self.spinbox.setSingleStep(step)
        self.slider.setTickInterval(step)

    def update_from_spinbox(self, i):
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.emit(SIG_INT_VALUE_CHANGED, i)

    def update_from_slider(self, i):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(i)
        self.spinbox.blockSignals(False)
        self.emit(SIG_INT_VALUE_CHANGED, i)

    def setValue(self, i):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(i)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.slider.blockSignals(False)
        if i != self.value():
            self.emit(SIG_INT_VALUE_CHANGED, i)

    def value(self):
        return self.spinbox.value()


class DoubleSpinSliderBox(QWidget):
    # a convenient widget that combines a slider and a double spinbox
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.label = QLabel(self)
        self.slider = QSlider(self)
        self.slider.setOrientation(0x1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.spinbox = QDoubleSpinBox(self)
        self.setRange(0.0, 100.0)
        self.setSingleStep(1.0)
        self.setSingleStep(1.0)

        hBox = QHBoxLayout(self)
        hBox.addWidget(self.label)
        hBox.addWidget(self.slider)
        hBox.addWidget(self.spinbox)

        self.connect(self.spinbox, SIG_DOUBLE_VALUE_CHANGED, self.update_from_spinbox)
        self.connect(self.slider, SIG_INT_VALUE_CHANGED, self.update_from_slider)
        self.connect(
            self.slider, SIG_SLIDER_PRESSED, lambda: self.emit(SIG_SLIDER_PRESSED)
        )
        self.connect(
            self.slider, SIG_SLIDER_RELEASED, lambda: self.emit(SIG_SLIDER_RELEASED)
        )

    def setText(self, text):
        self.label.setText(text)

    def setRange(self, _min, _max):
        if _max > _min:
            self.spinbox.setRange(_min, _max)
            self._min = _min
            self._max = _max
            self.scale = 100.0 / (self._max - self._min)

    def setSingleStep(self, step):
        self.spinbox.setSingleStep(step)

    def setDecimals(self, i):
        self.spinbox.setDecimals(i)

    def update_from_spinbox(self, x):
        self.slider.blockSignals(True)
        i = int((x - self._min) * self.scale)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.emit(SIG_SLIDER_PRESSED)
        self.emit(SIG_DOUBLE_VALUE_CHANGED, x)
        self.emit(SIG_SLIDER_RELEASED)

    def update_from_slider(self, i):
        self.spinbox.blockSignals(True)
        x = float(i) / self.scale + self._min
        self.spinbox.setValue(x)
        self.spinbox.blockSignals(False)
        self.emit(SIG_DOUBLE_VALUE_CHANGED, x)

    def setValue(self, x):
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.spinbox.setValue(x)
        i = int((x - self._min) * self.scale)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        if x != self.value():
            self.emit(SIG_SLIDER_PRESSED)
            self.emit(SIG_DOUBLE_VALUE_CHANGED, x)
            self.emit(SIG_SLIDER_RELEASED)

    def value(self):
        return self.spinbox.value()


def couple_doublespinsliders(spsl1, spsl2):
    QObject.connect(
        spsl1, SIG_DOUBLE_VALUE_CHANGED, lambda x: spsl2.setValue(max(x, spsl2.value()))
    )
    QObject.connect(
        spsl2, SIG_DOUBLE_VALUE_CHANGED, lambda x: spsl1.setValue(min(x, spsl1.value()))
    )


class ImageMaskingWidget(PanelWidget):
    __implements__ = (IPanel,)
    PANEL_ID = ID_IMAGEMASKING
    PANEL_TITLE = "image masking"
    PANEL_ICON = None  # string

    def __init__(self, parent=None):
        self._mask_shapes = {}
        self._mask_already_restored = {}

        super(ImageMaskingWidget, self).__init__(parent)
        self.setMinimumSize(QSize(250, 500))

        self.masked_image = None  # associated masked image item

        self.manager = None  # manager for the associated image plot

        self.toolbar = toolbar = QToolBar(self)
        toolbar.setOrientation(Qt.Horizontal)

        self.vBox = QVBoxLayout(self)  # central widget

        gbox1 = QGroupBox(self)
        gbox1.setTitle(_("Mask image with shape tools"))

        self.vBox1 = QVBoxLayout(gbox1)

        self.insidebutton = QCheckBox(self)
        self.insidebutton.setText(_("Inside"))
        self.outsidebutton = QCheckBox(self)
        self.outsidebutton.setText(_("Outside"))

        self.group1 = QButtonGroup(self)
        self.group1.addButton(self.insidebutton)
        self.group1.addButton(self.outsidebutton)
        self.insidebutton.setChecked(True)

        hBox1 = QHBoxLayout()
        hBox1.addWidget(self.insidebutton)
        hBox1.addWidget(self.outsidebutton)

        self.maskbutton = QCheckBox(self)
        self.maskbutton.setText(_("Mask"))
        self.unmaskbutton = QCheckBox(self)
        self.unmaskbutton.setText(_("Unmask"))

        self.group2 = QButtonGroup(self)
        self.group2.addButton(self.maskbutton)
        self.group2.addButton(self.unmaskbutton)
        self.maskbutton.setChecked(True)

        hBox2 = QHBoxLayout()
        hBox2.addWidget(self.maskbutton)
        hBox2.addWidget(self.unmaskbutton)

        self.brushsizelabel = QLabel(self)
        self.brushsizelabel.setText(_("Brush size"))

        self.hbrushsize = QSpinBox(self)
        self.hbrushsize.setRange(1, 100)

        self.vbrushsize = QSpinBox(self)
        self.vbrushsize.setRange(1, 100)

        hBox3 = QHBoxLayout()
        hBox3.addWidget(self.brushsizelabel)
        hBox3.addWidget(self.hbrushsize)
        hBox3.addWidget(self.vbrushsize)

        self.singleshapebutton = QCheckBox(self)
        self.singleshapebutton.setText(_("Single Shape"))
        self.singleshapebutton.setChecked(False)

        self.autoupdatebutton = QCheckBox(self)
        self.autoupdatebutton.setText(_("Auto update"))
        self.autoupdatebutton.setChecked(True)

        self.applymaskbutton = QPushButton(self)
        self.applymaskbutton.setText(_("Apply mask"))

        hBox4 = QHBoxLayout()
        hBox4.addWidget(self.autoupdatebutton)
        hBox4.addWidget(self.applymaskbutton)

        self.showshapesbutton = QCheckBox(self)
        self.showshapesbutton.setText(_("Show shapes"))
        self.removeshapesbutton = QPushButton(self)
        self.removeshapesbutton.setText(_("Remove shapes"))

        hBox5 = QHBoxLayout()
        hBox5.addWidget(self.showshapesbutton)
        hBox5.addWidget(self.removeshapesbutton)

        self.vBox1.addWidget(toolbar)
        self.vBox1.addLayout(hBox1)
        self.vBox1.addLayout(hBox2)
        self.vBox1.addLayout(hBox3)
        self.vBox1.addWidget(self.singleshapebutton)
        self.vBox1.addLayout(hBox4)
        self.vBox1.addLayout(hBox5)

        gbox2 = QGroupBox(self)
        gbox2.setTitle(_("Mask operations"))

        self.gr0 = QGridLayout(gbox2)

        self.showmaskbutton = QCheckBox(self)
        self.showmaskbutton.setText(_("Show mask"))

        self.clearmaskbutton = QPushButton(self)
        self.clearmaskbutton.setText(_("Clear mask"))

        self.invertmaskbutton = QPushButton(self)
        self.invertmaskbutton.setText(_("Invert mask"))

        self.dilatationbutton = QPushButton(self)
        self.dilatationbutton.setText(_("Dilatation"))

        self.erosionbutton = QPushButton(self)
        self.erosionbutton.setText(_("Erosion"))

        self.sizebox = QSpinBox(self)
        self.sizebox.setRange(1, 100)

        self.gr0.addWidget(self.invertmaskbutton, 0, 0)
        self.gr0.addWidget(self.clearmaskbutton, 1, 0)
        self.gr0.addWidget(self.showmaskbutton, 2, 0)
        self.gr0.addWidget(self.dilatationbutton, 0, 1)
        self.gr0.addWidget(self.erosionbutton, 1, 1)
        self.gr0.addWidget(self.sizebox, 2, 1)

        gbox3 = QGroupBox(self)
        gbox3.setTitle(_("Mask image from threshold"))

        self.vBox3 = QVBoxLayout(gbox3)  # set VBox to central widget

        self.minspinslider = DoubleSpinSliderBox(self)
        self.minspinslider.setText("Min")
        self.minspinslider.setRange(0.0, 100.0)
        self.minspinslider.setDecimals(1)
        self.minspinslider.setSingleStep(0.1)
        self.minspinslider.setValue(0.0)

        self.maxspinslider = DoubleSpinSliderBox(self)
        self.maxspinslider.setText("Max")
        self.maxspinslider.setRange(0.0, 100.0)
        self.maxspinslider.setDecimals(1)
        self.maxspinslider.setSingleStep(0.1)
        self.maxspinslider.setValue(100.0)

        couple_doublespinsliders(self.minspinslider, self.maxspinslider)

        # self.vBox2.addLayout(hBox7)
        self.vBox3.addWidget(self.minspinslider)
        self.vBox3.addWidget(self.maxspinslider)

        self.vBox.addWidget(gbox1)
        self.vBox.addWidget(gbox2)
        self.vBox.addWidget(gbox3)

    def register_plot(self, baseplot):
        self._mask_shapes.setdefault(baseplot, [])
        self.connect(baseplot, SIG_ITEMS_CHANGED, self.items_changed)
        self.connect(baseplot, SIG_ITEM_SELECTION_CHANGED, self.item_selection_changed)

    def register_panel(self, manager):
        """Register panel to plot manager"""
        self.manager = manager
        default_toolbar = self.manager.get_default_toolbar()
        self.manager.add_toolbar(self.toolbar, "masking shapes")
        self.manager.set_default_toolbar(default_toolbar)
        for plot in manager.get_plots():
            self.register_plot(plot)

    def configure_panel(self):
        """Configure panel"""
        self.ellipse_mask_tool = self.manager.add_tool(
            EllipseMaskTool, toolbar_id="masking shapes"
        )
        self.rect_mask_tool = self.manager.add_tool(
            RectangleMaskTool, toolbar_id="masking shapes"
        )
        self.rect_tool = self.manager.add_tool(
            RectangleTool,
            toolbar_id="masking shapes",
            handle_final_shape_cb=lambda shape: self.handle_shape(shape),
            title=_("Mask rectangular area"),
            icon="mask_rectangle_grey.png",
        )

        self.ellipse_tool = self.manager.add_tool(
            CircleTool,
            toolbar_id="masking shapes",
            handle_final_shape_cb=lambda shape: self.handle_shape(shape),
            title=_("Mask circular area"),
            icon="mask_circle_grey.png",
        )

        self.polygon_tool = self.manager.add_tool(
            FreeFormTool,
            toolbar_id="masking shapes",
            handle_final_shape_cb=lambda shape: self.handle_shape(shape),
            title=_("Mask polygonal area"),
            icon="mask_polygon_grey.png",
        )

        self.run_tool = self.manager.add_tool(
            RunTool, title=_("Remove masked area"), toolbar_id="masking shapes"
        )

        self.setup_actions()
        self.hbrushsize.setValue(1)
        self.vbrushsize.setValue(1)
        self.set_brush_size(1)
        self.showmaskbutton.setChecked(True)
        self.showshapesbutton.setChecked(True)

    def setup_actions(self):
        # QObject.connect(self.maskbutton, SIG_STATE_CHANGED, self.set_mode)
        QObject.connect(self.hbrushsize, SIG_INT_VALUE_CHANGED, self.set_brush_size)
        QObject.connect(self.vbrushsize, SIG_INT_VALUE_CHANGED, self.set_brush_size)
        QObject.connect(self.applymaskbutton, SIG_CLICKED, self.apply_mask)
        QObject.connect(self.showshapesbutton, SIG_STATE_CHANGED, self.show_shapes)
        QObject.connect(self.removeshapesbutton, SIG_CLICKED, self.remove_all_shapes)
        QObject.connect(self.clearmaskbutton, SIG_CLICKED, self.clear_mask)
        QObject.connect(self.showmaskbutton, SIG_STATE_CHANGED, self.show_mask)
        QObject.connect(self.invertmaskbutton, SIG_CLICKED, self.invert_mask)
        QObject.connect(self.dilatationbutton, SIG_CLICKED, self.mask_dilatation)
        QObject.connect(self.erosionbutton, SIG_CLICKED, self.mask_erosion)
        QObject.connect(
            self.minspinslider, SIG_DOUBLE_VALUE_CHANGED, self.update_threshold
        )
        QObject.connect(
            self.maxspinslider, SIG_DOUBLE_VALUE_CHANGED, self.update_threshold
        )

    def set_mode(self, i):
        self.ellipse_mask_tool.set_mode(self.maskbutton.isChecked())
        self.rect_mask_tool.set_mode(self.maskbutton.isChecked())

    def set_brush_size(self, i=1):
        # self.cmt.set_size(i)
        i = self.hbrushsize.value()
        j = self.vbrushsize.value()
        self.ellipse_mask_tool.set_pixel_size(i, j)
        self.rect_mask_tool.set_pixel_size(i, j)

    def update_threshold(self, x):
        plot = self.get_active_plot()
        if self.masked_image is None:
            return
        pmin = self.minspinslider.value()
        pmax = self.maxspinslider.value()
        vmin = np.percentile(
            self.masked_image.data[np.isfinite(self.masked_image.data)], pmin
        )
        vmax = np.percentile(
            self.masked_image.data[np.isfinite(self.masked_image.data)], pmax
        )
        self.masked_image.data.mask = np.ma.nomask
        np.ma.masked_less(self.masked_image.data, vmin, copy=False)
        np.ma.masked_greater(self.masked_image.data, vmax, copy=False)
        self.show_mask(True)
        plot.replot()

    def mask_dilatation(self):
        plot = self.get_active_plot()
        if self.masked_image is None:
            return

        radius = self.sizebox.value()
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        struct = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.bool_)

        self.masked_image.data.mask = (
            signal.fftconvolve(self.masked_image.data.mask, struct, "same") > 0.5
        )  # much better than ndimage.binary_dilatation

        self.show_mask(True)
        plot.replot()

    def mask_erosion(self):
        plot = self.get_active_plot()
        if self.masked_image is None:
            return

        radius = self.sizebox.value()
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        struct = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.bool_)

        self.masked_image.data.mask = (
            signal.fftconvolve(
                np.logical_not(self.masked_image.data.mask), struct, "same"
            )
            <= 0.5
        )  # much better than ndimage.binary_dilatation

        self.show_mask(True)
        plot.replot()

    def get_active_plot(self):
        return self.manager.get_active_plot()

    def handle_shape(self, shape):
        shape.set_style("plot", "shape/mask")
        shape.shapeparam.label = "mask shape"
        shape.set_private(True)
        plot = self.manager.get_active_plot()
        plot.set_active_item(shape)
        if self.singleshapebutton.isChecked():
            self.remove_shapes()
            if self.masked_image is not None:
                self.masked_image.unmask_all()
        self._mask_shapes[plot] += [
            (shape, self.insidebutton.isChecked(), self.maskbutton.isChecked())
        ]

        if self.autoupdatebutton.isChecked():
            self.apply_shape_mask(
                shape, self.insidebutton.isChecked(), self.maskbutton.isChecked()
            )

    def show_mask(self, state):
        if self.masked_image is not None:
            self.masked_image.set_mask_visible(state)

    def invert_mask(self):
        if self.masked_image is not None:
            self.masked_image.set_mask(np.logical_not(self.masked_image.get_mask()))
            self.masked_image.plot().replot()

    def apply_shape_mask(self, shape, inside, mask):
        if self.masked_image is None:
            return
        if isinstance(shape, RectangleShape):
            self.masked_image.align_rectangular_shape(shape)
            x0, y0, x1, y1 = shape.get_rect()
            if mask:
                self.masked_image.mask_rectangular_area(x0, y0, x1, y1, inside=inside)
            else:
                self.masked_image.unmask_rectangular_area(x0, y0, x1, y1, inside=inside)
        elif isinstance(shape, EllipseShape):
            x0, y0, x1, y1 = shape.get_rect()
            if mask:
                self.masked_image.mask_circular_area(x0, y0, x1, y1, inside=inside)
            else:
                self.masked_image.unmask_circular_area(x0, y0, x1, y1, inside=inside)
        elif isinstance(shape, PolygonShape):
            if mask:
                self.masked_image.mask_polygonal_area(shape.points, inside=inside)
                text = (
                    "adding masked polygonal area to"
                    + self.masked_image.imageparam.label
                )
                for pt in shape.points:
                    text += "(%f,%f)," % (pt[0], pt[1])
                text += " inside=" + str(inside)
            else:
                self.masked_image.unmask_polygonal_area(shape.points, inside=inside)
                text = (
                    "adding unmasked polygonal area to"
                    + self.masked_image.imageparam.label
                )
                for pt in shape.points:
                    text += "(%f,%f)," % (pt[0], pt[1])
                text += " inside=" + str(inside)

    def apply_mask(self):
        plot = self.get_active_plot()
        for shape, inside, mask in self._mask_shapes[plot]:
            self.apply_shape_mask(shape, inside, mask)

        self.show_mask(True)
        # self.masked_image.set_mask(mask)
        plot.replot()
        # self.emit(SIG_APPLIED_MASK_TOOL)

    def remove_all_shapes(self):
        message = _("Do you really want to remove all masking shapes?")
        plot = self.get_active_plot()
        answer = QMessageBox.warning(
            plot,
            _("Remove all masking shapes"),
            message,
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.remove_shapes()

    def remove_shapes(self):

        plot = self.get_active_plot()
        plot.del_items(
            [shape for shape, _inside, mask in self._mask_shapes[plot]]
        )  # remove shapes
        self._mask_shapes[plot] = []
        plot.replot()

    def show_shapes(self, state):
        plot = self.get_active_plot()
        if plot is not None:
            for shape, _inside, mask in self._mask_shapes[plot]:
                shape.setVisible(state)
            plot.replot()

    def create_shapes_from_masked_areas(self):
        plot = self.get_active_plot()
        self._mask_shapes[plot] = []
        for area in self.masked_image.get_masked_areas():
            if area.geometry == "rectangular":
                shape = RectangleShape()
                shape.set_points(area.pts)
                self.masked_image.align_rectangular_shape(shape)
            elif area.geometry == "circular":
                shape = EllipseShape()
                shape.set_points(area.pts)
            elif area.geometry == "polygonal":
                shape = PolygonShape()
                shape.set_points(area.pts)

            shape.set_style("plot", "shape/custom_mask")
            shape.set_private(True)
            self._mask_shapes[plot] += [(shape, area.inside, area.mask)]
            plot.blockSignals(True)
            plot.add_item(shape)
            plot.blockSignals(False)

    def find_masked_image(self, plot):
        item = plot.get_active_item()
        if isinstance(item, MaskedImageNan):
            return item
        else:
            items = [
                item for item in plot.get_items() if isinstance(item, MaskedImageNan)
            ]
            if items:
                return items[-1]

    def set_masked_image(self, plot):
        self.masked_image = self.find_masked_image(plot)
        if self.masked_image is not None and not self._mask_already_restored:
            self.create_shapes_from_masked_areas()
            self._mask_already_restored = True

    def items_changed(self, plot):
        self.set_masked_image(plot)
        self._mask_shapes[plot] = [
            (shape, inside, mask)
            for shape, inside, mask in self._mask_shapes[plot]
            if shape.plot() is plot
        ]
        self.update_status(plot)

    def item_selection_changed(self, plot):
        self.set_masked_image(plot)
        self.update_status(plot)

    def reset_mask(self, plot):
        # remove shapes prior opening a masked image with masked areas
        self._mask_shapes[plot] = []
        self._mask_already_restored = False

    def clear_mask(self):
        if self.masked_image is None:
            return
        message = _("Do you really want to clear the mask?")
        plot = self.get_active_plot()
        answer = QMessageBox.warning(
            plot, _("Clear mask"), message, QMessageBox.Yes | QMessageBox.No
        )
        if answer == QMessageBox.Yes:
            self.masked_image.unmask_all()

            plot.replot()

    def update_status(self, plot):
        # self.action.setEnabled(self.masked_image is not None)
        pass

    def activate_command(self, plot, checked):
        """Activate tool"""
        pass


class ProjectionWidget(PanelWidget):
    PANEL_ID = 999

    def __init__(self, parent=None):
        super(ProjectionWidget, self).__init__(parent)

        self.manager = None  # manager for the associated image plot
        self.local_manager = PlotManager(self)  # local manager for the histogram plot

        self.setMinimumWidth(180)
        self.dockwidget = None

        VBox = QVBoxLayout()
        self.setLayout(VBox)

        self.zlog = QCheckBox("log(I)", self)
        self.integral = QCheckBox("integral", self)
        self.plane = QCheckBox(
            "plane selection", self
        )  # select a specific plane instead of a range

        self.bg = QCheckBox("subtract background", self)

        self.swap = QCheckBox("swap", self)
        self.autoscale = QCheckBox("autoscale", self)
        self.autoscale.setChecked(True)
        self.show_contributions = QCheckBox("show contributions", self)

        self.xplot = BinoPlot()
        self.yplot = BinoPlot()
        self.zplot = BinoPlot()

        self.local_manager.add_plot(self.xplot)
        self.local_manager.add_plot(self.yplot)
        self.local_manager.add_plot(self.zplot)

        self.xcurve = make.binocurve([0, 1], [0, 0])
        self.xcurve.xlabel = "X"
        self.ycurve = make.binocurve([0, 1], [0, 0])
        self.ycurve.xlabel = "Y"
        self.zcurve = make.binocurve([0, 1], [0, 0])
        self.zcurve.xlabel = "Z"

        self.xrange = XRangeSelection2(0, 1)
        self.yrange = XRangeSelection2(0, 1)
        self.zrange = XRangeSelection2(0, 1)

        self.xbgrange1 = BgRangeSelection(
            0, 1
        )  # range selection for background subtraction
        self.ybgrange1 = BgRangeSelection(0, 1)
        self.zbgrange1 = BgRangeSelection(0, 1)

        self.xbgrange2 = BgRangeSelection(
            0, 1
        )  # range selection for background subtraction
        self.ybgrange2 = BgRangeSelection(0, 1)
        self.zbgrange2 = BgRangeSelection(0, 1)

        self.xbgrange1.setVisible(False)
        self.ybgrange1.setVisible(False)
        self.zbgrange1.setVisible(False)
        self.xbgrange2.setVisible(False)
        self.ybgrange2.setVisible(False)
        self.zbgrange2.setVisible(False)

        self.xmarker = Marker(
            constraint_cb=lambda x, y: self.xcurve.get_closest_x(x, y)
        )
        self.xmarker.set_markerstyle("|")
        self.ymarker = Marker(
            constraint_cb=lambda x, y: self.ycurve.get_closest_x(x, y)
        )
        self.ymarker.set_markerstyle("|")
        self.zmarker = Marker(
            constraint_cb=lambda x, y: self.zcurve.get_closest_x(x, y)
        )
        self.zmarker.set_markerstyle("|")
        self.xmarker.setVisible(False)  # default projection along h
        self.ymarker.setVisible(False)
        self.zmarker.setVisible(False)

        self.xplot.add_item(self.xbgrange1)
        self.xplot.add_item(self.xbgrange2)
        self.xplot.add_item(self.xrange)
        self.xplot.add_item(self.xcurve)
        self.xplot.add_item(self.xmarker)
        self.xplot.replot()

        self.yplot.add_item(self.ybgrange1)
        self.yplot.add_item(self.ybgrange2)
        self.yplot.add_item(self.yrange)
        self.yplot.add_item(self.ycurve)
        self.yplot.add_item(self.ymarker)
        self.yplot.replot()

        self.zplot.add_item(self.zbgrange1)
        self.zplot.add_item(self.zbgrange2)
        self.zplot.add_item(self.zrange)
        self.zplot.add_item(self.zcurve)
        self.zplot.add_item(self.zmarker)
        self.zplot.replot()

        self.xplot.set_active_item(self.xrange)
        self.yplot.set_active_item(self.yrange)
        self.zplot.set_active_item(self.zrange)

        HBox0a = QHBoxLayout()
        HBox0b = QHBoxLayout()
        HBox0c = QHBoxLayout()
        HBox1 = QHBoxLayout()
        HBox2 = QHBoxLayout()
        HBox3 = QHBoxLayout()

        self.projx = QCheckBox(self)
        self.projx.setText("X")
        self.projy = QCheckBox(self)
        self.projy.setText("Y")
        self.projz = QCheckBox(self)
        self.projz.setText("Z")

        self.projx.setChecked(True)
        self.axis = 0

        self.projgroup = QButtonGroup(VBox)
        self.projgroup.setExclusive(True)
        self.projgroup.addButton(self.projx)
        self.projgroup.addButton(self.projy)
        self.projgroup.addButton(self.projz)

        HBox1.addWidget(self.projx)
        HBox2.addWidget(self.projy)
        HBox3.addWidget(self.projz)

        self.xmin = QLineEdit(self)
        self.ymin = QLineEdit(self)
        self.zmin = QLineEdit(self)

        HBox1.addWidget(self.xmin)
        HBox2.addWidget(self.ymin)
        HBox3.addWidget(self.zmin)

        self.xmax = QLineEdit(self)
        self.ymax = QLineEdit(self)
        self.zmax = QLineEdit(self)

        HBox1.addWidget(self.xmax)
        HBox2.addWidget(self.ymax)
        HBox3.addWidget(self.zmax)

        HBox0a.addWidget(self.zlog)
        HBox0a.addWidget(self.integral)
        HBox0a.addWidget(self.plane)

        HBox0b.addWidget(self.bg)
        HBox0c.addWidget(self.swap)
        HBox0c.addWidget(self.autoscale)
        HBox0c.addWidget(self.show_contributions)

        VBox.addLayout(HBox0a)
        VBox.addLayout(HBox0b)
        VBox.addLayout(HBox0c)
        VBox.addLayout(HBox1)
        VBox.addWidget(self.xplot)
        VBox.addLayout(HBox2)
        VBox.addWidget(self.yplot)
        VBox.addLayout(HBox3)
        VBox.addWidget(self.zplot)

        lman = self.local_manager
        lman.add_tool(SelectTool)
        lman.add_tool(BasePlotMenuTool, "item")
        lman.add_tool(BasePlotMenuTool, "axes")
        lman.add_tool(BasePlotMenuTool, "grid")
        lman.add_tool(AntiAliasingTool)
        lman.get_default_tool().activate()
        self.setup_connect()
        self.active_plot = self.xplot
        self.prefs = SlicePrefs()

    def keyPressEvent(self, event):
        if type(event) == QKeyEvent:
            # here accept the event and do something
            # print event.key()
            # print self.active_plot
            event.accept()
        else:
            event.ignore()

    def get_plot(self):
        return self.manager.get_active_plot()

    def register_panel(self, manager):
        self.manager = manager

    def configure_panel(self):
        pass

    def setup_connect(self):
        QObject.connect(self.xplot, SIG_RANGE_CHANGED, self.selection_changed)
        QObject.connect(self.yplot, SIG_RANGE_CHANGED, self.selection_changed)
        QObject.connect(self.zplot, SIG_RANGE_CHANGED, self.selection_changed)
        QObject.connect(self.xplot, SIG_MARKER_CHANGED, self.selection_changed)
        QObject.connect(self.yplot, SIG_MARKER_CHANGED, self.selection_changed)
        QObject.connect(self.zplot, SIG_MARKER_CHANGED, self.selection_changed)
        QObject.connect(self.xplot, SIG_ITEM_SELECTION_CHANGED, self.set_active_plot)
        QObject.connect(self.yplot, SIG_ITEM_SELECTION_CHANGED, self.set_active_plot)
        QObject.connect(self.zplot, SIG_ITEM_SELECTION_CHANGED, self.set_active_plot)
        QObject.connect(self.xplot, SIGNAL("Move(int)"), self.move_xcursor)
        QObject.connect(self.yplot, SIGNAL("Move(int)"), self.move_ycursor)
        QObject.connect(self.zplot, SIGNAL("Move(int)"), self.move_zcursor)
        QObject.connect(self.xmin, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.xmax, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.ymin, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.ymax, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.zmin, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.zmax, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.xmin, SIGNAL("returnPressed()"), self.value_changed)
        QObject.connect(self.zlog, SIGNAL("clicked()"), self.log_changed)
        QObject.connect(self.swap, SIGNAL("clicked()"), self.other_changed)
        QObject.connect(
            self.show_contributions, SIGNAL("clicked()"), self.other_changed
        )
        QObject.connect(self.plane, SIGNAL("clicked()"), self.plane_changed)
        QObject.connect(self.bg, SIGNAL("clicked()"), self.bg_changed)
        QObject.connect(self.projx, SIGNAL("clicked()"), lambda: self.axis_changed(0))
        QObject.connect(self.projy, SIGNAL("clicked()"), lambda: self.axis_changed(1))
        QObject.connect(self.projz, SIGNAL("clicked()"), lambda: self.axis_changed(2))

    def log_changed(self):
        if self.zlog.isChecked():
            self.xplot.set_axis_scale("left", "log")
            self.yplot.set_axis_scale("left", "log")
            self.zplot.set_axis_scale("left", "log")
        else:
            self.xplot.set_axis_scale("left", "lin")
            self.yplot.set_axis_scale("left", "lin")
            self.zplot.set_axis_scale("left", "lin")
        self.xplot.replot()
        self.yplot.replot()
        self.zplot.replot()
        self.emit(SIGNAL("projection_changed()"))

    def update_bg_range(self):
        # we show background for the projected curve
        # we update position of the bg to be in adequation with the range selected
        self.xbgrange1.setVisible(False)
        self.ybgrange1.setVisible(False)
        self.zbgrange1.setVisible(False)
        self.xbgrange2.setVisible(False)
        self.ybgrange2.setVisible(False)
        self.zbgrange2.setVisible(False)

        dxs2 = self.spacings[self.axis] / 2.0
        if self.axis == 0:
            bgrange1 = self.xbgrange1
            bgrange2 = self.xbgrange2
            if self.plane.isChecked():
                xmarker = self.xmarker.xValue()
                xrmin, xrmax = xmarker - dxs2, xmarker + dxs2
            else:
                xrmin, xrmax = self.xrange.get_range()
            xdata = self.xcurve.get_data()[0]

        elif self.axis == 1:
            bgrange1 = self.ybgrange1
            bgrange2 = self.ybgrange2
            if self.plane.isChecked():
                xmarker = self.ymarker.xValue()
                xrmin, xrmax = xmarker - dxs2, xmarker + dxs2
            else:
                xrmin, xrmax = self.yrange.get_range()
            xdata = self.ycurve.get_data()[0]

        else:
            bgrange1 = self.zbgrange1
            bgrange2 = self.zbgrange2
            if self.plane.isChecked():
                xmarker = self.zmarker.xValue()
                xrmin, xrmax = xmarker - dxs2, xmarker + dxs2
            else:
                xrmin, xrmax = self.zrange.get_range()
            xdata = self.zcurve.get_data()[0]

        bgrange1.setVisible(True)
        bgrange2.setVisible(True)

        if len(xdata) > 0:
            xmin = min(xdata)
            xmax = max(xdata)
            if (
                xmin > xrmin
            ):  # instead of starting from first point of the curve, we start from a lowest value
                xmin = xrmin
            if xmax < xrmax:
                xmax = xrmax
            bgrange1.set_range(xmin - dxs2, xrmin, dosignal=False)
            bgrange2.set_range(xmax + dxs2, xrmax, dosignal=False)
        else:  # no data, we choose arbitrary 10% tail
            bgrange1.set_range(xrmin - 0.1 * (xrmax - xrmin), xrmin, dosignal=False)
            bgrange2.set_range(xrmax, xrmax + 0.1 * (xrmax - xrmin), dosignal=False)

    def bg_changed(self):
        if self.bg.isChecked():
            # we show background for the projected curve
            self.update_bg_range()

        else:
            self.xbgrange1.setVisible(False)
            self.ybgrange1.setVisible(False)
            self.zbgrange1.setVisible(False)
            self.xbgrange2.setVisible(False)
            self.ybgrange2.setVisible(False)
            self.zbgrange2.setVisible(False)

        self.emit(SIGNAL("projection_changed()"))

    def other_changed(self, i=0):
        self.emit(SIGNAL("projection_changed()"))

    def set_active_plot(self, plot):
        self.active_plot = plot
        self.emit(SIGNAL("active_plot_changed()"))

    def selection_changed(self, plot):
        # a range selection has been modified
        xmin, xmax = self.xrange.get_range()
        ymin, ymax = self.yrange.get_range()
        zmin, zmax = self.zrange.get_range()
        self.selection_ranges = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
        self.set_values(self.selection_ranges)
        self.emit(SIGNAL("projection_changed()"))
        pass

    def value_changed(self):
        # a range value has been entered
        xmin = float(self.xmin.text())
        xmax = float(self.xmax.text())
        ymin = float(self.ymin.text())
        ymax = float(self.ymax.text())
        zmin = float(self.zmin.text())
        zmax = float(self.zmax.text())
        self.selection_ranges = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
        self.set_ranges(self.selection_ranges)
        self.emit(SIGNAL("projection_changed()"))
        pass

    def plane_changed(self):
        # change between range and plane selection modes
        if self.bg.isChecked():
            self.update_bg_range()

        if self.plane.isChecked():
            self.xrange.setVisible(False)
            self.yrange.setVisible(False)
            self.zrange.setVisible(False)
            self.set_marker_position()
        else:
            self.xmarker.setVisible(False)
            self.ymarker.setVisible(False)
            self.zmarker.setVisible(False)
            self.xrange.setVisible(True)
            self.yrange.setVisible(True)
            self.zrange.setVisible(True)
        self.xplot.replot()
        self.yplot.replot()
        self.zplot.replot()
        self.emit(SIGNAL("projection_changed()"))

    def move_xcursor(self, i):
        if self.plane.isChecked():
            if self.axis == 0:  # we move the cursor
                xdata, ydata = self.xcurve.get_data()
                if len(xdata) > 0:  # put marker at the middle of the curve
                    x, y = self.xmarker.get_pos()
                    x, y = self.xcurve.get_closest_x(x + i * self.spacings[0], y)
                    self.xmarker.setValue(x, y)
                    self.emit(SIGNAL("projection_changed()"))
        else:  # we move the range
            _min, _max = self.xrange.get_range()
            self.xrange.set_range(
                _min + i * self.spacings[0], _max + i * self.spacings[0]
            )

    def move_ycursor(self, i):
        if self.plane.isChecked():
            if self.axis == 1:  # we move the cursor
                xdata, ydata = self.ycurve.get_data()
                if len(xdata) > 0:  # put marker at the middle of the curve
                    x, y = self.ymarker.get_pos()
                    x, y = self.ycurve.get_closest_x(x + i * self.spacings[1], y)
                    self.ymarker.setValue(x, y)
                    self.emit(SIGNAL("projection_changed()"))
        else:  # we move the range
            _min, _max = self.yrange.get_range()
            self.yrange.set_range(
                _min + i * self.spacings[1], _max + i * self.spacings[1]
            )

    def move_zcursor(self, i):
        if self.plane.isChecked():
            if self.axis == 2:  # we move the cursor
                xdata, ydata = self.zcurve.get_data()
                if len(xdata) > 0:  # put marker at the middle of the curve
                    x, y = self.zmarker.get_pos()
                    x, y = self.zcurve.get_closest_x(x + i * self.spacings[2], y)
                    self.zmarker.setValue(x, y)
                    self.emit(SIGNAL("projection_changed()"))
        else:  # we move the range
            _min, _max = self.zrange.get_range()
            self.zrange.set_range(
                _min + i * self.spacings[2], _max + i * self.spacings[2]
            )

    def set_marker_position(self):
        if self.axis == 0:
            self.xmarker.setVisible(True)
            self.ymarker.setVisible(False)
            self.zmarker.setVisible(False)
            self.xplot.select_some_items([self.xmarker])
            xdata, ydata = self.xcurve.get_data()
            if len(xdata) > 0:  # put marker at the middle of the curve
                x, y = self.xmarker.get_pos()
                x, y = self.xcurve.get_closest_x(x, y)
                self.xmarker.setValue(x, y)

        elif self.axis == 1:
            self.xmarker.setVisible(False)
            self.ymarker.setVisible(True)
            self.zmarker.setVisible(False)
            self.yplot.select_some_items([self.ymarker])
            xdata, ydata = self.ycurve.get_data()
            if len(xdata) > 0:
                x, y = self.ymarker.get_pos()
                x, y = self.ycurve.get_closest_x(x, y)
                self.ymarker.setValue(x, y)
        else:
            self.xmarker.setVisible(False)
            self.ymarker.setVisible(False)
            self.zmarker.setVisible(True)
            self.zplot.select_some_items([self.zmarker])
            xdata, ydata = self.zcurve.get_data()
            if len(xdata) > 0:
                x, y = self.zmarker.get_pos()
                x, y = self.zcurve.get_closest_x(x, y)
                self.zmarker.setValue(x, y)

    def axis_changed(self, axis):
        # a projection axis has been modified
        self.axis = axis
        if self.bg.isChecked():
            self.update_bg_range()
        if self.plane.isChecked():
            self.xmarker.setVisible(axis == 0)
            self.ymarker.setVisible(axis == 1)
            self.zmarker.setVisible(axis == 2)
            self.set_marker_position()
            self.emit(SIGNAL("projection_changed()"))
            self.set_marker_position()  # needed to ajust y value
            if axis == 0:
                self.xplot.replot()
            elif axis == 1:
                self.yplot.replot()
            else:
                self.zplot.replot()

        else:
            self.emit(SIGNAL("projection_changed()"))

    def set_ranges_and_values(self, ranges, labels=None, spacings=[1, 1, 1]):
        # adjust ranges and values with new data
        if labels is not None:
            self.projx.setText(labels[0])
            self.projy.setText(labels[1])
            self.projz.setText(labels[2])
            self.xcurve.xlabel = labels[0]
            self.ycurve.xlabel = labels[1]
            self.zcurve.xlabel = labels[2]

        self.selection_ranges = np.array(ranges)
        self.ranges = np.array(ranges)
        self.set_ranges(ranges, spacings)
        self.set_values(ranges)

    def set_ranges(self, ranges, spacings=None, dosignal=False):
        if spacings is not None:
            self.spacings = spacings  # intervals between points along x,y,z
        # selection
        self.xrange.set_range(
            ranges[0][0] - self.spacings[0] / 2.0,
            ranges[0][1] + self.spacings[0] / 2.0,
            dosignal=dosignal,
        )
        self.yrange.set_range(
            ranges[1][0] - self.spacings[1] / 2.0,
            ranges[1][1] + self.spacings[1] / 2.0,
            dosignal=dosignal,
        )
        self.zrange.set_range(
            ranges[2][0] - self.spacings[2] / 2.0,
            ranges[2][1] + self.spacings[2] / 2.0,
            dosignal=dosignal,
        )

        # background left
        self.xbgrange1.set_range(
            ranges[0][0] - 3.0 * self.spacings[0] / 2.0,
            ranges[0][0] - self.spacings[0] / 2.0,
            dosignal=dosignal,
        )
        self.ybgrange1.set_range(
            ranges[1][0] - 3.0 * self.spacings[1] / 2.0,
            ranges[1][0] - self.spacings[1] / 2.0,
            dosignal=dosignal,
        )
        self.zbgrange1.set_range(
            ranges[2][0] - 3.0 * self.spacings[2] / 2.0,
            ranges[2][0] - self.spacings[2] / 2.0,
            dosignal=dosignal,
        )

        # background right
        self.xbgrange2.set_range(
            ranges[0][1] + self.spacings[0] / 2.0,
            ranges[0][1] + 3.0 * self.spacings[0] / 2.0,
            dosignal=dosignal,
        )
        self.ybgrange2.set_range(
            ranges[1][1] + self.spacings[1] / 2.0,
            ranges[1][1] + 3.0 * self.spacings[1] / 2.0,
            dosignal=dosignal,
        )
        self.zbgrange2.set_range(
            ranges[2][1] + self.spacings[2] / 2.0,
            ranges[2][1] + 3.0 * self.spacings[2] / 2.0,
            dosignal=dosignal,
        )

        self.xplot.replot()
        self.yplot.replot()
        self.zplot.replot()

    def set_values(self, ranges):
        self.xmin.setText("%f" % ranges[0][0])
        self.xmax.setText("%f" % ranges[0][1])
        self.ymin.setText("%f" % ranges[1][0])
        self.ymax.setText("%f" % ranges[1][1])
        self.zmin.setText("%f" % ranges[2][0])
        self.zmax.setText("%f" % ranges[2][1])


class Image3DDialog(ImageDialog):
    def __init__(
        self, parent=None,
    ):
        defaultoptions = {
            "show_contrast": True,
            "show_xsection": False,
            "show_ysection": False,
            "lock_aspect_ratio": False,
        }
        ImageDialog.__init__(self, edit=False, toolbar=True, options=defaultoptions)
        self.fittool = self.add_tool(FitTool)
        self.fittool.connect(self.fittool, SIG_VALIDATE_TOOL, self.set_fit)
        self.rectangleerasertool = self.add_tool(RectangleEraserTool)
        self.rectangleerasertool.connect(
            self.rectangleerasertool,
            SIGNAL("suppress_area()"),
            self.suppress_rectangular_area,
        )
        self.image = None
        self.xcurve = None
        self.ycurve = None
        self.zcurve = None
        self.xvalues = []
        self.yvalues = []
        self.zvalues = []
        self.data = []
        self.isafit = 0

        self.sliceprefs = SlicePrefs()
        self.laplaceparam = LaplaceParam()
        self.filterparam = FilterParam()

        self.create_menubar()
        self.preferences = Preferences()
        self.make_default_image()
        QObject.connect(self.p_panel, SIGNAL("projection_changed()"), self.update_image)
        QObject.connect(
            self.imagemasking.run_tool, SIG_VALIDATE_TOOL, self.delete_masked_values
        )

    def register_image_tools(self):
        ImageDialog.register_image_tools(self)
        self.openfiletool = self.add_tool(OpenFileTool)
        self.openfiletool.connect(
            self.openfiletool, SIGNAL("openfile(QString*)"), self.open_file
        )

    def create_menubar(self):
        self.menubar = QMenuBar(self)
        self.layout().setMenuBar(self.menubar)

        """***************File Menu*******************************"""

        self.menuFile = QMenu("File", self.menubar)

        self.actionOpen = QAction("Open", self.menuFile)
        self.actionOpen.setShortcut("Ctrl+o")

        self.actionQuit = QAction("Quit", self.menuFile)
        self.actionQuit.setShortcut("Ctrl+Q")

        self.menuFile.addActions((self.actionOpen, self.actionQuit))

        """***************Operations Menu************************"""

        self.menuOperations = QMenu("Operations", self.menubar)

        self.menuSlices = QMenu("Slices", self.menuOperations)
        self.action2D = QAction("Slice rod", self.menuSlices)
        self.actionStraight = QAction("Scan straight rod", self.menuSlices)
        self.actionTilted = QAction("Scan tilted rod", self.menuSlices)
        self.menuSlices.addActions(
            (self.action2D, self.actionStraight, self.actionTilted)
        )

        self.menuInterpolation = QMenu("Interpolation", self.menuOperations)
        self.actionGriddata = QAction("Scipy griddata", self.menuInterpolation)
        self.actionLaplace = QAction("Laplace equation", self.menuInterpolation)
        self.actionFilling = QAction("Filling", self.menuInterpolation)
        self.menuInterpolation.addActions(
            (self.actionGriddata, self.actionLaplace, self.actionFilling)
        )

        self.actionFilter = QAction("Filter data", self.menuOperations)

        self.menuOperations.addMenu(self.menuSlices)
        self.menuOperations.addMenu(self.menuInterpolation)
        self.menuOperations.addAction(self.actionFilter)

        """***************Settings Menu************************"""
        self.menuSettings = QMenu("Setting", self.menubar)
        self.actionPreferences = QAction("Preferences", self.menuSettings)
        self.menuSettings.addAction(self.actionPreferences)

        """**********************************************"""

        self.menubar.addMenu(self.menuFile)
        self.menubar.addMenu(self.menuOperations)
        self.menubar.addMenu(self.menuSettings)

        QObject.connect(self.actionOpen, SIGNAL("triggered()"), self.get_open_filename)
        QObject.connect(self.actionQuit, SIGNAL("triggered()"), self.close)
        QObject.connect(self.action2D, SIGNAL("triggered()"), self.make_2D_slices)
        QObject.connect(
            self.actionStraight, SIGNAL("triggered()"), self.make_straight_slices
        )
        QObject.connect(
            self.actionTilted, SIGNAL("triggered()"), self.make_tilted_slices
        )
        QObject.connect(
            self.actionGriddata, SIGNAL("triggered()"), self.griddata_interpolation
        )
        QObject.connect(
            self.actionLaplace, SIGNAL("triggered()"), self.laplace_interpolation
        )
        QObject.connect(self.actionFilling, SIGNAL("triggered()"), self.filling)
        QObject.connect(self.actionFilter, SIGNAL("triggered()"), self.do_filter)
        QObject.connect(
            self.actionPreferences, SIGNAL("triggered()"), self.set_preferences
        )

    def create_plot(self, options, row=0, column=0, rowspan=1, columnspan=1):
        ImageDialog.create_plot(self, options, row, column, rowspan, columnspan)
        # ra_panel = ObliqueCrossSection(self)
        # splitter = self.plot_widget.xcsw_splitter
        # splitter.addWidget(ra_panel)
        # splitter.setStretchFactor(splitter.count()-1, 1)
        # splitter.setSizes(list(splitter.sizes())+[2])
        # self.add_panel(ra_panel)

        self.p_panel = ProjectionWidget(self)
        self.imagemasking = ImageMaskingWidget(self)

        splitter = self.plot_widget.ycsw_splitter
        splitter.addWidget(self.p_panel)
        splitter.addWidget(self.imagemasking)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes(list(splitter.sizes()) + [2])

        self.add_panel(self.p_panel)
        self.add_panel(self.imagemasking)

    def make_default_image(self):
        self.image = make.maskedimagenan(
            np.zeros((1, 1)), interpolation="nearest", xdata=[0, 0], ydata=[0, 0]
        )
        self.imagemasking.masked_image = self.image
        self.image.set_mask_visible(self.imagemasking.showmaskbutton.isChecked())
        self.get_plot().add_item(self.image)

    def update_image(self):
        axis = self.p_panel.axis
        integral = self.p_panel.integral.isChecked()
        bg = self.p_panel.bg.isChecked()
        if self.data is not None:
            if self.p_panel.plane.isChecked():
                self.select_plane(axis, bg=bg)  # show intensity for selected plane
            else:
                self.make_projection(
                    axis, integral=integral, bg=bg
                )  # show integrated intensity for selected range

            # pixel size calculation
            xmin, xmax = self.image.get_xdata()
            dx = (xmax - xmin) / float(self.image.data.shape[1])
            ymin, ymax = self.image.get_ydata()
            dy = (ymax - ymin) / float(self.image.data.shape[0])
            self.rectangleerasertool.set_size(dx, dy)
            self.imagemasking.rect_mask_tool.set_size(dx, dy)
            self.imagemasking.ellipse_mask_tool.set_size(dx, dy)

    def get_open_filename(self):
        self.openfiletool.activate()

    def open_file(self, filename):
        print(self.openfiletool.directory)
        filename = str(filename)
        try:
            hfile = tables.openFile(filename)
        except:
            QMessageBox.about(self, "Error", "unable to open file")
            return

        print(hfile.root.binoculars.counts.shape)

        try:
            self.data = np.array(hfile.root.binoculars.counts.read(), np.float32)
            self.contributions = np.array(
                hfile.root.binoculars.contributions.read(), np.float32
            )

        except tables.exceptions.NoSuchNodeError:
            QMessageBox.about(self, "Error", "file is empty")
            return

        except MemoryError:
            try:
                self.data = np.array(hfile.root.binoculars.counts.read(), np.float16)
                self.contributions = np.array(
                    hfile.root.binoculars.contributions.read(), np.float16
                )

            except MemoryError:
                QMessageBox.about(self, "Error", "file is too big")

        axes = hfile.root.binoculars.axes

        self.labels = []
        # print self.axes._v_children,labels
        for leaf in axes._f_listNodes():
            self.labels.append(leaf.name)

        # print self.axes._f_listNodes()
        self.x_label, self.y_label, self.z_label = self.labels
        self.x_values = axes._v_children[self.x_label].read()
        self.y_values = axes._v_children[self.y_label].read()
        self.z_values = axes._v_children[self.z_label].read()

        self.values = np.array([self.x_values, self.y_values, self.z_values])
        ranges = self.values[:, 1:3]
        spacings = self.values[:, 3]
        self.p_panel.set_ranges_and_values(ranges, self.labels, spacings)

        if self.image.plot() is None:
            self.make_default_image()

        self.update_image()
        self.setWindowTitle(filename.split("/")[-1])

        mins = [
            round(self.x_values[1], 12),
            round(self.y_values[1], 12),
            round(self.z_values[1], 12),
        ]
        maxs = [
            round(self.x_values[2], 12),
            round(self.y_values[2], 12),
            round(self.z_values[2], 12),
        ]
        steps = [
            round(self.x_values[3], 12),
            round(self.y_values[3], 12),
            round(self.z_values[3], 12),
        ]
        maxs = maxs + steps  # borne exclue

        self.update_sliceprefs(
            absranges=[mins, maxs], ranges=[mins, maxs], steps=steps, labels=self.labels
        )

        # self.slicetool2.update_pref(absranges=[mins,maxs],steps=steps,labels=self.labels)

        # interpolation de self.data

    def set_preferences(self):
        doit = self.preferences.edit()
        if doit:
            self.rectangleerasertool.set_pixel_size(self.preferences.eraser_size)

    def suppress_rectangular_area(self):
        # we first suppress pixels on the drawn image, then apply the result to the data
        if self.image is not None:
            x0, y0, x1, y1 = self.rectangleerasertool.shape.get_rect()
            self.image.suppress_rectangular_area(x0, y0, x1, y1)
            plot = self.get_plot()
            plot.replot()

    def delete_masked_values(self):

        axis = self.p_panel.axis

        ix0, ix1 = self.showndatalimits[0]
        iy0, iy1 = self.showndatalimits[1]
        iz0, iz1 = self.showndatalimits[2]

        mask = self.image.data.mask
        if self.p_panel.swap.isChecked():
            mask = mask.transpose()
        if axis == 0:
            for ix in range(ix0, ix1):
                self.data[ix, iy0:iy1, iz0:iz1] = np.where(
                    mask, 0.0, self.data[ix, iy0:iy1, iz0:iz1]
                )
                self.contributions[ix, iy0:iy1, iz0:iz1] = np.where(
                    mask, 0.0, self.contributions[ix, iy0:iy1, iz0:iz1]
                )
        elif axis == 1:
            for iy in range(iy0, iy1):
                self.data[ix0:ix1, iy, iz0:iz1] = np.where(
                    mask, 0.0, self.data[ix0:ix1, iy, iz0:iz1]
                )
                self.contributions[ix0:ix1, iy, iz0:iz1] = np.where(
                    mask, 0.0, self.contributions[ix0:ix1, iy, iz0:iz1]
                )
        else:
            for iz in range(iz0, iz1):
                self.data[ix0:ix1, iy0:iy1, iz] = np.where(
                    mask, 0.0, self.data[ix0:ix1, iy0:iy1, iz]
                )
                self.contributions[ix0:ix1, iy0:iy1, iz] = np.where(
                    mask, 0.0, self.contributions[ix0:ix1, iy0:iy1, iz]
                )
        self.update_image()

    def laplace_interpolation(self):

        # first we select point to interpolate
        doit = self.laplaceparam.edit()
        if not doit:
            return

        if self.laplaceparam.doslice:

            i0 = self.laplaceparam.cutoff

            progressbar = ProgressBar("progression...")
            progressbar.show()

            sld = np.array(self.data)  # copy to save
            slc = np.array(self.contributions)  # copy to save
            kmax = self.data.shape[2]

            for k in range(kmax):
                progressbar.update_progress(k / float(kmax))

                cont = self.contributions[:, :, k]
                if np.sum(cont) > 0:
                    # we interpolate independantly each plane

                    mpoints = np.nonzero(cont < i0)  # values to interpolate
                    dt = 0.5

                    sle = np.nan_to_num(
                        self.data[:, :, k] / cont
                    )  # normalized values, 0 if no contributions

                    # first we fill with default
                    a = sle.shape
                    # a=float(a[0]*a[1]*a[2])
                    a = float(a[0] * a[1])

                    # sle[mpoints]=self.laplaceparam.fill           #starting point
                    # self.contributions[mpoints]=self.laplaceparam.fillc           #starting point

                    mpoints = np.nonzero(cont == 0)  # values to interpolate
                    mpointsc = np.nonzero(cont == 0)  # copy
                    x = len(mpoints[0])

                    while x > 0:
                        z = uniform_filter(cont)[mpoints]
                        y = uniform_filter(sle * cont)[mpoints]

                        sle[mpoints] = y / z  # replace with mean values
                        sle[mpoints] = np.nan_to_num(sle[mpoints])

                        cont[mpoints] = np.where(
                            sle[mpoints] > 0, self.laplaceparam.fillc, 0
                        )  # replace with default value fillc

                        mpoints = np.nonzero(cont == 0)  # values to interpolate
                        x = len(mpoints[0])
                        if progressbar.stop:
                            x = 0

                    mpoints = mpointsc
                    delta = laplace(sle, mode="nearest")
                    max0 = np.max(np.abs(delta[mpoints]))

                    sle[mpoints] += delta[mpoints] * dt

                    for i in range(self.laplaceparam.steps):
                        delta = laplace(sle, mode="nearest")
                        max1 = np.max(np.abs(delta[mpoints]))

                        if max1 < max0:
                            dt = dt * self.laplaceparam.increase
                        else:
                            dt = dt * self.laplaceparam.decrease

                        sle[mpoints] += delta[mpoints] * dt

                    self.data[:, :, k] = sle * self.contributions[:, :, k]

                    if progressbar.stop:
                        break

            progressbar.close()

            self.update_image()
            i = QMessageBox.question(
                self,
                "what should I do",
                "what should I do",
                "apply",
                "continue",
                "cancel",
            )

            if i == 2:  # do not change and stop iterating
                self.data = sld
                self.contributions = slc
                self.update_image()

        else:

            i0 = self.laplaceparam.cutoff

            mpoints = np.nonzero(self.contributions < i0)  # values to interpolate
            dt = 0.5

            sld = np.array(self.data)  # copy to save
            slc = np.array(self.contributions)

            sle = np.nan_to_num(
                self.data / slc
            )  # normalized values, 0 if no contributions

            # first we fill with default
            a = sle.shape
            # a=float(a[0]*a[1]*a[2])
            a = float(a[0] * a[1] * a[2])

            # sle[mpoints]=self.laplaceparam.fill           #starting point
            # self.contributions[mpoints]=self.laplaceparam.fillc           #starting point

            progressbar = ProgressBar("filling first...")
            progressbar.show()

            mpoints = np.nonzero(slc == 0)  # values to interpolate
            mpointsc = np.nonzero(slc == 0)  # copy
            x = len(mpoints[0])

            while x > 0:
                progressbar.update_progress(1.0 - x / a)
                z = uniform_filter(self.contributions)[mpoints]
                y = uniform_filter(sle * self.contributions)[mpoints]

                sle[mpoints] = y / z  # replace with mean values
                sle[mpoints] = np.nan_to_num(sle[mpoints])

                self.contributions[mpoints] = np.where(
                    sle[mpoints] > 0, self.laplaceparam.fillc, 0
                )  # replace with default value fillc

                mpoints = np.nonzero(self.contributions == 0)  # values to interpolate
                x = len(mpoints[0])
                if progressbar.stop:
                    x = 0

            progressbar.close()

            mpoints = mpointsc
            delta = laplace(sle, mode="nearest")
            max0 = np.max(np.abs(delta[mpoints]))

            sle[mpoints] += delta[mpoints] * dt

            while doit:
                progressbar = ProgressBar("laplace transformation...")
                progressbar.show()
                for i in range(self.laplaceparam.steps):
                    delta = laplace(sle, mode="nearest")
                    max1 = np.max(np.abs(delta[mpoints]))

                    if max1 < max0:
                        dt = dt * self.laplaceparam.increase
                    else:
                        dt = dt * self.laplaceparam.decrease

                    sle[mpoints] += delta[mpoints] * dt

                    x = i / float(self.laplaceparam.steps)
                    progressbar.update_progress(x)

                    if progressbar.stop:
                        break

                self.data = sle * self.contributions
                self.update_image()
                txt = "Maximum laplacian value=%f" % max1
                i = QMessageBox.question(
                    self, "what should I do", txt, "apply", "continue", "cancel"
                )

                if i == 2:  # do not change and stop iterating
                    self.data = sld
                    self.contributions = slc
                    self.update_image()
                    doit = False
                elif i == 0:
                    doit = False  # stop iterating

    def filling(self):
        # fill pixels with no contributions with pixels where

        # select one plane
        slc = np.array(self.contributions)
        sle = np.nan_to_num(self.data / slc)

        a = sle.shape
        # a=float(a[0]*a[1]*a[2])
        a = float(a[0] * a[1] * a[2])

        progressbar = ProgressBar("filling...")
        progressbar.show()

        mpoints = np.nonzero(slc == 0)  # values to interpolate
        x = len(mpoints[0])

        while x > 0:
            progressbar.update_progress(1.0 - x / a)
            z = uniform_filter(slc)[mpoints]
            y = uniform_filter(sle * slc)[mpoints]

            sle[mpoints] = y / z  # replace with mean values
            sle[mpoints] = np.nan_to_num(sle[mpoints])

            slc[mpoints] = 10 * z / z  # replace with default value=10
            slc[mpoints] = np.nan_to_num(slc[mpoints])

            mpoints = np.nonzero(slc == 0)  # values to interpolate
            x = len(mpoints[0])
            if progressbar.stop:
                x = 0

        self.contributions = slc
        self.data = sle * slc
        self.update_image()

    def griddata_interpolation(self):
        """old version on slices
        x,y = np.indices(self.data.shape[0:2])
        nz=self.data.shape[2]
        print "interpolation"
        for iz in range(nz):
          print iz+1,' / ',nz
          sld=self.data[:,:,iz]
          slc=self.contributions[:,:,iz]

          if np.sum(slc)>0:
            sle=sld/slc  #intensite normalisees
            #interpolate contributions
            slc[np.isnan(sle)]=griddata((x[~np.isnan(sle)], y[~np.isnan(sle)]), # points we know
                                        slc[~np.isnan(sle)],                    # values we know
                                        (x[np.isnan(sle)], y[np.isnan(sle)]),
                                        method='linear',fill_value=0.)  #fill with 0 for points outside of the convex hull of the input points
            #interpolate normalized intensities and multiply by contributions
            sld[np.isnan(sle)]=slc[np.isnan(sle)]*griddata((x[~np.isnan(sle)], y[~np.isnan(sle)]), # points we know
                                        sle[~np.isnan(sle)],                    # values we know
                                        (x[np.isnan(sle)], y[np.isnan(sle)]),
                                        method='linear',fill_value=0.)

            self.data[:,:,iz]=sld
            self.contributions[:,:,iz]=slc
        """
        """new version: 3D interpolation"""
        x, y, z = np.indices(self.data.shape)

        sle = self.data / self.contributions  # intensite normalisees
        hascont = self.contributions > 0
        # interpolate contributions

        print("interpolate contributions")
        self.contributions[~hascont] = griddata(
            (x[hascont], y[hascont], z[hascont]),  # points we know
            self.contributions[hascont],  # values we know
            (x[~hascont], y[~hascont], z[~hascont]),
            method="nearest",
            fill_value=0.0,
        )  # fill with 0 for points outside of the convex hull of the input points

        # nterpolate normalized intensities and multiply by
        # contributions
        print("interpolate counts")
        self.data[~hascont] = self.contributions[~hascont] * griddata(
            (x[hascont], y[hascont], z[hascont]),  # points we know
            sle[hascont],  # values we know
            (x[~hascont], y[~hascont], z[~hascont]),
            method="nearest",
            fill_value=0.0,
        )
        self.update_image()

    def do_filter(self):
        doit = self.filterparam.edit()
        if not doit:
            return

        mpoints = np.nonzero(
            self.contributions < self.filterparam.cutoff
        )  # values to remove

        self.contributions[mpoints] = 0
        self.data[mpoints] = 0
        self.update_image()

    def select_plane(self, axis, bg=False):
        # show a given plane (with orientation given axis and position given by corresponding marker)
        with np.errstate(invalid="ignore"):
            xmin = self.x_values[1] - self.x_values[3] / 2.0  # raw table limits
            xmax = self.x_values[2] + self.x_values[3] / 2.0
            ymin = self.y_values[1] - self.y_values[3] / 2.0
            ymax = self.y_values[2] + self.y_values[3] / 2.0
            zmin = self.z_values[1] - self.z_values[3] / 2.0
            zmax = self.z_values[2] + self.z_values[3] / 2.0

            if axis == 0:
                x0 = self.p_panel.xmarker.xValue()
                ix0 = int(round((x0 - self.x_values[1]) / self.x_values[3]))
                if ix0 < 0:
                    ix0 = 0
                if ix0 > self.data.shape[0]:
                    ix0 = self.data.shape[0]

                if self.p_panel.show_contributions.isChecked():
                    proj = self.contributions[ix0, :, :]
                    projx = np.sum(self.contributions, axis=(1, 2))
                    projy = np.sum(self.contributions[ix0, :, :], axis=1)
                    projz = np.sum(self.contributions[ix0, :, :], axis=0)
                else:
                    proj = self.data[ix0, :, :] / self.contributions[ix0, :, :]
                    projx = np.sum(self.data, axis=(1, 2)) / np.sum(
                        self.contributions, axis=(1, 2)
                    )
                    projy = np.sum(self.data[ix0, :, :], axis=1) / np.sum(
                        self.contributions[ix0, :, :], axis=1
                    )
                    projz = np.sum(self.data[ix0, :, :], axis=0) / np.sum(
                        self.contributions[ix0, :, :], axis=0
                    )
                self.showndatalimits = [
                    [ix0, ix0 + 1],
                    [0, self.data.shape[1]],
                    [0, self.data.shape[2]],
                ]  # limits shown

            elif axis == 1:
                y0 = self.p_panel.ymarker.xValue()
                iy0 = int(round((y0 - self.y_values[1]) / self.y_values[3]))
                if iy0 < 0:
                    iy0 = 0
                if iy0 > self.data.shape[1]:
                    iy0 = self.data.shape[1]

                if self.p_panel.show_contributions.isChecked():
                    proj = self.contributions[:, iy0, :]
                    projx = np.sum(self.contributions[:, iy0, :], axis=1)
                    projy = np.sum(self.contributions, axis=(0, 2))
                    projz = np.sum(self.contributions[:, iy0, :], axis=0)
                else:
                    proj = self.data[:, iy0, :] / self.contributions[:, iy0, :]
                    projx = np.sum(self.data[:, iy0, :], axis=1) / np.sum(
                        self.contributions[:, iy0, :], axis=1
                    )
                    projy = np.sum(self.data, axis=(0, 2)) / np.sum(
                        self.contributions, axis=(0, 2)
                    )
                    projz = np.sum(self.data[:, iy0, :], axis=0) / np.sum(
                        self.contributions[:, iy0, :], axis=0
                    )
                self.showndatalimits = [
                    [0, self.data.shape[0]],
                    [iy0, iy0 + 1],
                    [0, self.data.shape[2]],
                ]  # limits shown
            else:
                z0 = self.p_panel.zmarker.xValue()
                iz0 = int(round((z0 - self.z_values[1]) / self.z_values[3]))
                if iz0 < 0:
                    iz0 = 0
                if iz0 > self.data.shape[2]:
                    iz0 = self.data.shape[2]

                if self.p_panel.show_contributions.isChecked():
                    proj = self.contributions[:, :, iz0]
                    projx = np.sum(self.contributions[:, :, iz0], axis=1)
                    projy = np.sum(self.contributions[:, :, iz0], axis=0)
                    projz = np.sum(self.contributions, axis=(0, 1))
                else:
                    proj = self.data[:, :, iz0] / self.contributions[:, :, iz0]
                    projx = np.sum(self.data[:, :, iz0], axis=1) / np.sum(
                        self.contributions[:, :, iz0], axis=1
                    )
                    projy = np.sum(self.data[:, :, iz0], axis=0) / np.sum(
                        self.contributions[:, :, iz0], axis=0
                    )
                    projz = np.sum(self.data, axis=(0, 1)) / np.sum(
                        self.contributions, axis=(0, 1)
                    )
                self.showndatalimits = [
                    [0, self.data.shape[0]],
                    [0, self.data.shape[1]],
                    [iz0, iz0 + 1],
                ]  # limits shown

            if bg:
                # i3,i4,i5,i6 indices for bg subraction
                if axis == 0:
                    vv = self.x_values
                    x3, x4 = self.p_panel.xbgrange1.get_range()
                    x5, x6 = self.p_panel.xbgrange2.get_range()
                elif axis == 1:
                    vv = self.y_values
                    x3, x4 = self.p_panel.ybgrange1.get_range()
                    x5, x6 = self.p_panel.ybgrange2.get_range()
                else:
                    vv = self.z_values
                    x3, x4 = self.p_panel.zbgrange1.get_range()
                    x5, x6 = self.p_panel.zbgrange2.get_range()
                ibgs = []
                for x in [x3, x4, x5, x6]:
                    ibg = int(np.floor((x - vv[1]) / vv[3]) + 1)
                    if ibg < 0:
                        ibg = 0
                    if ibg > self.data.shape[axis]:
                        ibg = self.data.shape[axis]
                    ibgs.append(ibg)
                i3, i4, i5, i6 = ibgs
                if i3 > i4:
                    i3, i4 = i4, i3
                if i5 > i6:
                    i5, i6 = i6, i5
                # we subtract background:
                if axis == 0:
                    bg_value = (
                        np.sum(self.data[i3:i4, :, :], axis=axis)
                        + np.sum(self.data[i5:i6, :, :], axis=axis)
                    ) / (
                        np.sum(self.contributions[i3:i4, :, :], axis=axis)
                        + np.sum(self.contributions[i5:i6, :, :], axis=axis)
                    )
                elif axis == 1:
                    bg_value = (
                        np.sum(self.data[:, i3:i4, :], axis=axis)
                        + np.sum(self.data[:, i5:i6, :], axis=axis)
                    ) / (
                        np.sum(self.contributions[:, i3:i4, :], axis=axis)
                        + np.sum(self.contributions[:, i5:i6, :], axis=axis)
                    )
                else:
                    bg_value = (
                        np.sum(self.data[:, :, i3:i4], axis=axis)
                        + np.sum(self.data[:, :, i5:i6], axis=axis)
                    ) / (
                        np.sum(self.contributions[:, :, i3:i4], axis=axis)
                        + np.sum(self.contributions[:, :, i5:i6], axis=axis)
                    )
                bg_value = np.nan_to_num(bg_value)  # replace nan numbers with 0.
                proj = proj - bg_value

            ai = self.x_values[1]
            af = (
                self.x_values[2] + self.x_values[3] / 2.0
            )  # pour eviter les problemes d'arrondi. arange s'arrete tout seul
            da = self.x_values[3]
            xpts = np.arange(ai, af, da)
            ai = self.y_values[1]
            af = self.y_values[2] + self.y_values[3] / 2.0
            da = self.y_values[3]
            ypts = np.arange(ai, af, da)
            ai = self.z_values[1]
            af = self.z_values[2] + self.z_values[3] / 2.0
            da = self.z_values[3]
            zpts = np.arange(ai, af, da)

            if axis == 0:
                xdata, ydata = (zmin, zmax), (ymin, ymax)
                xlabel, ylabel = self.z_label, self.y_label
            elif axis == 1:
                xdata, ydata = (zmin, zmax), (xmin, xmax)
                xlabel, ylabel = self.z_label, self.x_label
            else:
                xdata, ydata = (ymin, ymax), (xmin, xmax)
                xlabel, ylabel = self.y_label, self.x_label

            plot = self.get_plot()

            if self.p_panel.zlog.isChecked():
                v = np.nanmin(
                    proj[np.array(proj, bool)]
                )  # minimum of the nonzero values of the array
                proj = np.log(proj + v)

            lutrange = self.image.get_lut_range()
            if self.p_panel.swap.isChecked():
                plot.set_axis_title("bottom", ylabel)
                plot.set_axis_title("left", xlabel)
                self.image.set_data(np.transpose(proj))
                self.image.set_xdata(ydata[0], ydata[1])
                self.image.set_ydata(xdata[0], xdata[1])
            else:
                plot.set_axis_title("bottom", xlabel)
                plot.set_axis_title("left", ylabel)
                self.image.set_data(proj)
                self.image.set_xdata(xdata[0], xdata[1])
                self.image.set_ydata(ydata[0], ydata[1])

            self.image.imageparam.update_param(self.image)
            self.image.imageparam.update_image(self.image)

            if self.p_panel.autoscale.isChecked():
                plot.do_autoscale()
            else:
                self.image.set_lut_range(lutrange)
                plot.replot()
            plot.emit(SIG_ITEM_SELECTION_CHANGED, plot)

            self.p_panel.xcurve.set_data(
                xpts[np.isfinite(projx)], projx[np.isfinite(projx)]
            )
            if self.p_panel.autoscale.isChecked():
                self.p_panel.xplot.do_autoscale()
            else:
                self.p_panel.xplot.replot()
            self.p_panel.ycurve.set_data(
                ypts[np.isfinite(projy)], projy[np.isfinite(projy)]
            )

            if self.p_panel.autoscale.isChecked():
                self.p_panel.yplot.do_autoscale()
            else:
                self.p_panel.yplot.replot()
            self.p_panel.zcurve.set_data(
                zpts[np.isfinite(projz)], projz[np.isfinite(projz)]
            )
            if self.p_panel.autoscale.isChecked():
                self.p_panel.zplot.do_autoscale()
            else:
                self.p_panel.zplot.replot()

    def make_projection(self, axis, integral=False, bg=False):
        # axis: axis along which the projection will be made
        # integral: use raw sum instead of normalized counts
        # bg: subtract background, with bgts from each side
        with np.errstate(invalid="ignore"):
            xmin = self.p_panel.selection_ranges[0][0]
            xmax = self.p_panel.selection_ranges[0][1]
            ymin = self.p_panel.selection_ranges[1][0]
            ymax = self.p_panel.selection_ranges[1][1]
            zmin = self.p_panel.selection_ranges[2][0]
            zmax = self.p_panel.selection_ranges[2][1]
            ix1 = int(np.floor((xmin - self.x_values[1]) / self.x_values[3]) + 1)
            ix2 = int(np.floor((xmax - self.x_values[1]) / self.x_values[3]) + 1)
            iy1 = int(np.floor((ymin - self.y_values[1]) / self.y_values[3]) + 1)
            iy2 = int(np.floor((ymax - self.y_values[1]) / self.y_values[3]) + 1)
            iz1 = int(np.floor((zmin - self.z_values[1]) / self.z_values[3]) + 1)
            iz2 = int(np.floor((zmax - self.z_values[1]) / self.z_values[3]) + 1)

            if ix1 > ix2:
                ix2, ix1 = ix1, ix2
            if iy1 > iy2:
                iy2, iy1 = iy1, iy2
            if iz1 > iz2:
                iz2, iz1 = iz1, iz2

            if ix1 < 0:
                ix1 = 0
            if ix2 < 0:
                ix2 = 0
            if iy1 < 0:
                iy1 = 0
            if iy2 < 0:
                iy2 = 0
            if iz1 < 0:
                iz1 = 0
            if iz2 < 0:
                iz2 = 0

            if ix1 > self.data.shape[0]:
                ix1 = self.data.shape[0]
            if ix2 > self.data.shape[0]:
                ix2 = self.data.shape[0]
            if iy1 > self.data.shape[1]:
                iy1 = self.data.shape[1]
            if iy2 > self.data.shape[1]:
                iy2 = self.data.shape[1]
            if iz1 > self.data.shape[2]:
                iz1 = self.data.shape[2]
            if iz2 > self.data.shape[2]:
                iz2 = self.data.shape[2]

            if ix1 == ix2:
                return
            if iy1 == iy2:
                return
            if iz1 == iz2:
                return

            if self.p_panel.show_contributions.isChecked():
                proj = np.sum(self.contributions[ix1:ix2, iy1:iy2, iz1:iz2], axis=axis)
                projx = np.sum(self.contributions[:, iy1:iy2, iz1:iz2], axis=(1, 2))
                projy = np.sum(self.contributions[ix1:ix2, :, iz1:iz2], axis=(0, 2))
                projz = np.sum(self.contributions[ix1:ix2, iy1:iy2, :], axis=(0, 1))
            else:
                proj = np.sum(self.data[ix1:ix2, iy1:iy2, iz1:iz2], axis=axis) / np.sum(
                    self.contributions[ix1:ix2, iy1:iy2, iz1:iz2], axis=axis
                )
                projx = np.sum(self.data[:, iy1:iy2, iz1:iz2], axis=(1, 2)) / np.sum(
                    self.contributions[:, iy1:iy2, iz1:iz2], axis=(1, 2)
                )
                projy = np.sum(self.data[ix1:ix2, :, iz1:iz2], axis=(0, 2)) / np.sum(
                    self.contributions[ix1:ix2, :, iz1:iz2], axis=(0, 2)
                )
                projz = np.sum(self.data[ix1:ix2, iy1:iy2, :], axis=(0, 1)) / np.sum(
                    self.contributions[ix1:ix2, iy1:iy2, :], axis=(0, 1)
                )
                if bg:

                    # we subtract background:
                    if axis == 0:
                        vv = self.x_values
                        x3, x4 = self.p_panel.xbgrange1.get_range()
                        x5, x6 = self.p_panel.xbgrange2.get_range()
                    elif axis == 1:
                        vv = self.y_values
                        x3, x4 = self.p_panel.ybgrange1.get_range()
                        x5, x6 = self.p_panel.ybgrange2.get_range()
                    else:
                        vv = self.z_values
                        x3, x4 = self.p_panel.zbgrange1.get_range()
                        x5, x6 = self.p_panel.zbgrange2.get_range()
                    ibgs = []
                    for x in [x3, x4, x5, x6]:
                        ibg = int(np.floor((x - vv[1]) / vv[3]) + 1)
                        if ibg < 0:
                            ibg = 0
                        if ibg > self.data.shape[axis]:
                            ibg = self.data.shape[axis]
                        ibgs.append(ibg)
                    i3, i4, i5, i6 = ibgs
                    if i3 > i4:
                        i3, i4 = i4, i3
                    if i5 > i6:
                        i5, i6 = i6, i5

                    if axis == 0:
                        bg_value = (
                            np.sum(self.data[i3:i4, iy1:iy2, iz1:iz2], axis=axis)
                            + np.sum(self.data[i5:i6, iy1:iy2, iz1:iz2], axis=axis)
                        ) / (
                            np.sum(
                                self.contributions[i3:i4, iy1:iy2, iz1:iz2], axis=axis
                            )
                            + np.sum(
                                self.contributions[i5:i6, iy1:iy2, iz1:iz2], axis=axis
                            )
                        )
                    elif axis == 1:
                        bg_value = (
                            np.sum(self.data[ix1:ix2, i3:i4, iz1:iz2], axis=axis)
                            + np.sum(self.data[ix1:ix2, i5:i6, iz1:iz2], axis=axis)
                        ) / (
                            np.sum(
                                self.contributions[ix1:ix2, i3:i4, iz1:iz2], axis=axis
                            )
                            + np.sum(
                                self.contributions[ix1:ix2, i5:i6, iz1:iz2], axis=axis
                            )
                        )
                    else:
                        bg_value = (
                            np.sum(self.data[ix1:ix2, iy1:iy2, i3:i4], axis=axis)
                            + np.sum(self.data[ix1:ix2, iy1:iy2, i5:i6], axis=axis)
                        ) / (
                            np.sum(
                                self.contributions[ix1:ix2, iy1:iy2, i3:i4], axis=axis
                            )
                            + np.sum(
                                self.contributions[ix1:ix2, iy1:iy2, i5:i6], axis=axis
                            )
                        )
                    bg_value = np.nan_to_num(bg_value)  # replace nan numbers with 0.
                    proj = proj - bg_value

            if integral:
                # we replace averaged value by integrated value
                projx = projx * (ymax - ymin) * (zmax - zmin)
                projy = projy * (xmax - xmin) * (zmax - zmin)
                projz = projz * (xmax - xmin) * (ymax - ymin)

            xmin = (
                self.x_values[1] + (ix1 - 0.5) * self.x_values[3]
            )  # x_values are at the center of the pixels
            xmax = self.x_values[1] + (ix2 - 0.5) * self.x_values[3]
            ymin = self.y_values[1] + (iy1 - 0.5) * self.y_values[3]
            ymax = self.y_values[1] + (iy2 - 0.5) * self.y_values[3]
            zmin = self.z_values[1] + (iz1 - 0.5) * self.z_values[3]
            zmax = self.z_values[1] + (iz2 - 0.5) * self.z_values[3]

            self.showndatalimits = [[ix1, ix2], [iy1, iy2], [iz1, iz2]]  # limits shown
            if axis == 0:
                xdata, ydata = (zmin, zmax), (ymin, ymax)
                xlabel, ylabel = self.z_label, self.y_label
            elif axis == 1:
                xdata, ydata = (zmin, zmax), (xmin, xmax)
                xlabel, ylabel = self.z_label, self.x_label
            else:
                xdata, ydata = (ymin, ymax), (xmin, xmax)
                xlabel, ylabel = self.y_label, self.x_label

            xpts = self.x_values[1] + np.arange(self.data.shape[0]) * self.x_values[3]
            ypts = self.y_values[1] + np.arange(self.data.shape[1]) * self.y_values[3]
            zpts = self.z_values[1] + np.arange(self.data.shape[2]) * self.z_values[3]

            plot = self.get_plot()

            if self.p_panel.zlog.isChecked():
                v = np.nanmin(
                    proj[np.array(proj, bool)]
                )  # minimum of the nonzero values of the array
                proj = np.log(proj + v)

            lutrange = self.image.get_lut_range()
            if self.p_panel.swap.isChecked():
                plot.set_axis_title("bottom", ylabel)
                plot.set_axis_title("left", xlabel)
                self.image.set_data(np.transpose(proj))
                self.image.set_xdata(ydata[0], ydata[1])
                self.image.set_ydata(xdata[0], xdata[1])
            else:
                plot.set_axis_title("bottom", xlabel)
                plot.set_axis_title("left", ylabel)
                self.image.set_data(proj)
                self.image.set_xdata(xdata[0], xdata[1])
                self.image.set_ydata(ydata[0], ydata[1])

            self.image.imageparam.update_param(self.image)
            self.image.imageparam.update_image(self.image)

            if self.p_panel.autoscale.isChecked():
                plot.do_autoscale()
            else:
                self.image.set_lut_range(lutrange)
                plot.replot()
            plot.emit(SIG_ITEM_SELECTION_CHANGED, plot)

            self.p_panel.xcurve.set_data(
                xpts[np.isfinite(projx)], projx[np.isfinite(projx)]
            )
            if self.p_panel.autoscale.isChecked():
                self.p_panel.xplot.do_autoscale()
            else:
                self.p_panel.xplot.replot()
            self.p_panel.ycurve.set_data(
                ypts[np.isfinite(projy)], projy[np.isfinite(projy)]
            )

            if self.p_panel.autoscale.isChecked():
                self.p_panel.yplot.do_autoscale()
            else:
                self.p_panel.yplot.replot()
            self.p_panel.zcurve.set_data(
                zpts[np.isfinite(projz)], projz[np.isfinite(projz)]
            )
            if self.p_panel.autoscale.isChecked():
                self.p_panel.zplot.do_autoscale()
            else:
                self.p_panel.zplot.replot()

    def update_sliceprefs(self, absranges=None, ranges=None, steps=None, labels=None):
        if absranges is not None:
            # fixe les bornes des spinbox
            self.sliceprefs.absmins = absranges[0]
            self.sliceprefs.absmaxs = absranges[1]
        if ranges is not None:
            # determine les valeurs des spinbox
            self.sliceprefs.mins = ranges[0]
            self.sliceprefs.maxs = ranges[1]
        if steps is not None:
            self.sliceprefs.steps = steps
        if labels is not None:
            self.sliceprefs.labels = labels
        print(self.sliceprefs.mins, self.sliceprefs.maxs, self.sliceprefs.steps)

    def make_2D_slices(self):
        xmin = self.p_panel.selection_ranges[0][0]
        xmax = self.p_panel.selection_ranges[0][1]
        ymin = self.p_panel.selection_ranges[1][0]
        ymax = self.p_panel.selection_ranges[1][1]
        zmin = self.p_panel.selection_ranges[2][0]
        zmax = self.p_panel.selection_ranges[2][1]
        mins = [round(xmin, 12), round(ymin, 12), round(zmin, 12)]
        maxs = [round(xmax, 12), round(ymax, 12), round(zmax, 12)]

        self.update_sliceprefs(ranges=[mins, maxs])
        prefs = self.sliceprefs
        Set2DSliceWindow(self.sliceprefs)  # ouvre une fenetre de dialogue

        if not prefs.do_it:
            return
        # print prefs.mins
        # print prefs.maxs
        # print prefs.stepn,prefs.stepw
        # print prefs.i1,prefs.i2,prefs.i3
        i1 = prefs.i1  # slice direction
        tag1 = self.labels[i1]  # axis name

        ix1 = int(
            round((prefs.mins[0] - self.x_values[1]) / self.x_values[3])
        )  # borne inclue[
        ix2 = int(
            round((prefs.maxs[0] - self.x_values[1]) / self.x_values[3])
        )  # borne exclue[
        iy1 = int(round((prefs.mins[1] - self.y_values[1]) / self.y_values[3]))
        iy2 = int(round((prefs.maxs[1] - self.y_values[1]) / self.y_values[3]))
        iz1 = int(round((prefs.mins[2] - self.z_values[1]) / self.z_values[3]))
        iz2 = int(round((prefs.maxs[2] - self.z_values[1]) / self.z_values[3]))

        if ix1 >= ix2 or iy1 >= iy2 or iz1 >= iz2:
            return
        if ix1 < 0:
            ix1 = 0
        if iy1 < 0:
            iy1 = 0
        if iz1 < 0:
            iz1 = 0
        if ix2 > self.data.shape[0]:
            print("ix2 trop grand")
            ix2 = self.data.shape[0]
        if iy2 > self.data.shape[1]:
            print("iy2 trop grand")
            iy2 = self.data.shape[1]
        if iz2 > self.data.shape[2]:
            print("iz2 trop grand")
            iz2 = self.data.shape[2]

        mins = [self.x_values[1], self.y_values[1], self.z_values[1]]
        maxs = [self.x_values[2], self.y_values[2], self.z_values[2]]
        steps = [self.x_values[3], self.y_values[3], self.z_values[3]]

        if i1 != 2:
            QMessageBox.about(self, "not implemented yet", "change direction")
            print("not implemented yet")
            return

        print("mins,maxs")
        print(mins)
        print(maxs)
        print("-------------")

        x_range = [mins[0] + steps[0] * ix1, mins[0] + steps[0] * ix2]
        y_range = [mins[1] + steps[1] * iy1, mins[1] + steps[1] * iy2]

        vmin = prefs.mins[2]

        v1 = vmin
        j1 = int(round((v1 - mins[2]) / steps[2]))  # borne [

        fitwindow = Fit2DWindow()
        fitwindow.cwd = self.openfiletool.directory

        for i in range(prefs.stepn):
            v2 = vmin + (i + 1) * prefs.stepw
            j2 = int(round((v2 - mins[i1]) / steps[i1]))  # borne [
            proj = np.sum(
                self.data[ix1:ix2, iy1:iy2, j1:j2], axis=2
            )  # we do not multiply by stepw in order to have the density along the rod, as usual
            projcont = np.sum(self.contributions[ix1:ix2, iy1:iy2, j1:j2], axis=2)

            data = proj / projcont

            weights = projcont / np.sqrt(proj + 1)  # 1/errors

            # in that case the error bar is 1 count

            Q = mins[2] + float(j1 + j2 - 1) * steps[2] / 2.0
            title = tag1 + "=%f" % (Q)
            # we transpose data to have y as first direction (line number)
            if self.p_panel.swap.isChecked():  # x along
                fitwindow.add_data(
                    np.transpose(data),
                    x_range,
                    y_range,
                    weights,
                    title=title,
                    xlabel=self.x_label,
                    ylabel=self.y_label,
                    tags=[(tag1, Q)],
                )
            else:
                fitwindow.add_data(
                    data,
                    y_range,
                    x_range,
                    weights,
                    title=title,
                    xlabel=self.y_label,
                    ylabel=self.x_label,
                    tags=[(tag1, Q)],
                )
            j1 = j2

        fitwindow.setWindowTitle(self.windowTitle())
        fitwindow.show()

    def make_straight_slices(self):
        xmin = self.p_panel.selection_ranges[0][0]
        xmax = self.p_panel.selection_ranges[0][1]
        ymin = self.p_panel.selection_ranges[1][0]
        ymax = self.p_panel.selection_ranges[1][1]
        zmin = self.p_panel.selection_ranges[2][0]
        zmax = self.p_panel.selection_ranges[2][1]
        mins = [round(xmin, 12), round(ymin, 12), round(zmin, 12)]
        maxs = [round(xmax, 12), round(ymax, 12), round(zmax, 12)]

        self.update_sliceprefs(ranges=[mins, maxs])
        prefs = self.sliceprefs
        SetSliceWindow(self.sliceprefs)  # ouvre une fenetre de dialogue
        if not prefs.do_it:
            return

        # print prefs.mins
        # print prefs.maxs
        # print prefs.stepn,prefs.stepw
        # print prefs.i1,prefs.i2,prefs.i3
        i1, i2, i3 = prefs.i1, prefs.i2, prefs.i3

        ix1 = int(
            round((prefs.mins[0] - self.x_values[1]) / self.x_values[3])
        )  # borne inclue[
        ix2 = int(
            round((prefs.maxs[0] - self.x_values[1]) / self.x_values[3])
        )  # borne exclue[
        iy1 = int(round((prefs.mins[1] - self.y_values[1]) / self.y_values[3]))
        iy2 = int(round((prefs.maxs[1] - self.y_values[1]) / self.y_values[3]))
        iz1 = int(round((prefs.mins[2] - self.z_values[1]) / self.z_values[3]))
        iz2 = int(round((prefs.maxs[2] - self.z_values[1]) / self.z_values[3]))

        if ix1 >= ix2 or iy1 >= iy2 or iz1 >= iz2:
            return
        if ix1 < 0:
            ix1 = 0
        if iy1 < 0:
            iy1 = 0
        if iz1 < 0:
            iz1 = 0
        if ix2 > self.data.shape[0]:
            print("ix2 trop grand")
            ix2 = self.data.shape[0]
        if iy2 > self.data.shape[1]:
            print("iy2 trop grand")
            iy2 = self.data.shape[1]
        if iz2 > self.data.shape[2]:
            print("iz2 trop grand")
            iz2 = self.data.shape[2]

        imins = [ix1, iy1, iz1]
        imaxs = [ix2, iy2, iz2]
        mins = [self.x_values[1], self.y_values[1], self.z_values[1]]
        maxs = [self.x_values[2], self.y_values[2], self.z_values[2]]
        steps = [self.x_values[3], self.y_values[3], self.z_values[3]]

        print("mins,maxs")
        print(mins)
        print(maxs)
        print("-------------")

        ni3 = imaxs[i3] - imins[i3]
        # We do raw integration along direction i3, keeping a 3D array
        proj3 = (
            np.sum(self.data[ix1:ix2, iy1:iy2, iz1:iz2], axis=i3)
            * prefs.steps[i3]
            * ni3
        )
        # comme on va diviser par le nombre de contributions, il faut multiplier par la taille de la fenetre avant
        # qui est de prefs.steps[i3]*ni3
        # we have to take into account the number of time the pixels have been counted
        cont3 = np.sum(self.contributions[ix1:ix2, iy1:iy2, iz1:iz2], axis=i3)
        axes = [0, 1, 2]
        axes.remove(i3)
        # np.sum(self.contributions[ix1:ix2,iy1:iy2,iz1:iz2],axis=(prefs.i3),keepdims=True)
        # print 'ix1,ix2:',ix1,ix2
        # print 'iy1,iy2:',iy1,iy2
        # print 'iz1,iz2:',iz1,iz2

        # print "proj3",proj3.shape
        ii1 = axes.index(i1)  # index of the axis for 2D array to make sum
        # better would be to transpose at the beginning...
        if ii1 == 1:
            proj3 = proj3.transpose()
            cont3 = cont3.transpose()

        # we make slices along i1, now first direction of the 2D array
        vmin = prefs.mins[i1]
        # vmax=prefs.maxs[i1]
        v1 = vmin
        j1 = int(round((v1 - mins[i1]) / steps[i1]))  # borne [

        title = str(self.windowTitle())

        title = (
            title
            + "\nslices along "
            + self.labels[i1]
            + "\nraw sum along "
            + self.labels[i3]
            + " in the range [%g,%g[" % (prefs.mins[i3], prefs.maxs[i3])
        )

        self.slicewin = CurveDialog(
            edit=False,
            toolbar=True,
            wintitle="CurveDialog test",
            options=dict(title=title, xlabel=self.labels[i2], ylabel="intensity"),
        )

        plot = self.slicewin.get_plot()
        QObject.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.update_fit)

        self.init_fit()
        # scan abscisse
        wstep = steps[i2]
        wmin = mins[i2] + imins[i2] * wstep
        wmax = mins[i2] + imaxs[i2] * wstep
        pts = np.arange(
            wmin, wmax - wstep / 2.0, wstep
        )  # we put wmax-wstep to be sure not to have wmax included
        # print 'pts',wmin,wmax,pts.shape
        tag1 = self.labels[i1]
        tag2 = self.labels[i3] + "_min"
        tag3 = self.labels[i3] + "_max"

        zrgb = 255.0 / float(prefs.stepn - 1)

        Qs = []
        integrals = []  # intensity integrated along the window chosen
        errors = []

        for i in range(prefs.stepn):
            red = int(round(i * zrgb))
            green = 0
            blue = int(round((prefs.stepn - 1 - i) * zrgb))
            v2 = vmin + (i + 1) * prefs.stepw
            j2 = int(round((v2 - mins[i1]) / steps[i1]))  # borne [
            proj1 = np.sum(
                proj3[j1:j2], axis=0
            )  # we do not multiply by stepw in order to have the density along the rod, as usual
            # print 'proj1',proj1.shape
            cont1 = np.sum(cont3[j1:j2], axis=0)
            ydata = np.ma.masked_where(cont1 == 0, proj1)
            ydata = (
                ydata / cont1
            )  # on divise par les contributions donc on a une densite
            xdata = np.ma.masked_where(cont1 == 0, pts)

            Q = mins[i1] + float(j1 + j2 - 1) * steps[i1] / 2.0
            title = self.labels[i1] + "=%f" % (Q)
            item = make.curve(
                xdata.compressed(),
                ydata.compressed(),
                title=title,
                color=QColor(red, green, blue),
            )
            item.xlabel = self.labels[i2]
            item.tags = [[tag1, Q], [tag2, prefs.mins[i3]], [tag3, prefs.maxs[i3]]]

            Qs.append(Q)
            ncoupstot = np.sum(proj1) / (
                prefs.steps[i3] * ni3
            )  # nombre de coups detectes
            print(Q, ncoupstot)
            error = np.sqrt(ncoupstot)
            integral = np.sum(ydata) * wstep
            integrals.append(integral)
            errors.append(integral / error)

            plot.add_item(item)
            j1 = j2
            v1 = v2

        self.figfit.setsavedir(self.openfiletool.directory)
        self.slicewin.show()

        self.intwin = CurveDialog(
            edit=False,
            toolbar=True,
            wintitle="CurveDialog test",
            options=dict(title=title, xlabel=self.labels[i1], ylabel="intensity"),
        )
        item = make.error(Qs, integrals, None, errors)
        plot = self.intwin.get_plot()
        plot.add_item(item)
        self.intwin.show()

        self.slicewin.exec_()

    def make_tilted_slices(self):
        prefs = self.sliceprefs
        SetSliceWindow2(prefs)  # ouvre une fenetre de dialogue
        if not prefs.do_it:
            return

        print(prefs.i0, prefs.i1, prefs.i2)  # indices of directions
        print(prefs.ranges)  # range for raw_sum/range for scan/width og a slice
        print(prefs.stepn)  # number of slices
        print(prefs.mins[2], prefs.maxs[2])  # limit of slices
        print(prefs.pos1)  # 1 position in the rod
        print(prefs.pos2)  # 2 position in the rod

        # print prefs.mins
        # print prefs.maxs
        # print prefs.stepn,prefs.stepw
        # print prefs.i1,prefs.i2,prefs.i3
        i0, i1, i2 = (
            prefs.i0,
            prefs.i1,
            prefs.i2,
        )  # i0 slice direction,i1 scan direction, i2 raw sum direction
        if i1 == i0 or i1 == i2 or i2 == i0:
            print("two indices equal!!!")
            return

        pos1 = np.array(prefs.pos1)
        pos2 = np.array(prefs.pos2)
        if pos1[i2] == pos2[i2]:
            print("pos1[i2]==pos2[i2]=%f" % pos1[i2])
            return
        # we make a slice
        # mins=[self.x_values[1],self.y_values[1],self.z_values[1]]
        # maxs=[self.x_values[2],self.y_values[2],self.z_values[2]]
        # steps=[self.x_values[3],self.y_values[3],self.z_values[3]]
        # i2_min=int(round((prefs.mins[i2]-prefs.absmins[i2])/prefs.steps[i2]))
        # i2_max=int(round((prefs.maxs[i2]-prefs.absmins[i2])/prefs.steps[i2]))

        # we swap the array to have i0,i1,i2 coprresponding to axes 0,1,2

        data = self.data
        cont = self.contributions
        if i2 != 2:
            data = np.swapaxes(data, i2, 2)
            cont = np.swapaxes(cont, i2, 2)
        # pos1[i0],pos1[2]=pos1[2],pos1[i0]
        # pos2[i0],pos2[2]=pos2[2],pos2[i0]
        if i1 != 1:
            data = np.swapaxes(data, i1, 1)
            cont = np.swapaxes(cont, i1, 1)
        # pos1[i1],pos1[1]=pos1[1],pos1[i1]
        # pos2[i1],pos2[1]=pos2[1],pos2[i1]
        # that should be OK now! we have slice=last axes,raw_sum=first axe

        # number of integration points:
        kx = int(prefs.ranges[0] / prefs.steps[i0])
        ky = int(prefs.ranges[1] / prefs.steps[i1])
        print("number of points along raw_sum and scan directions", kx, ky)

        zmin = prefs.mins[2]
        zmax = prefs.maxs[2]
        z1 = zmin
        k1 = int(round((z1 - prefs.absmins[i2]) / prefs.steps[i2]))  # borne inf [

        title = str(self.windowTitle())

        title = (
            title
            + "\nslices along "
            + self.labels[i2]
            + "\nraw sum along "
            + self.labels[i0]
        )

        self.slicewin = CurveDialog(
            edit=False,
            toolbar=True,
            wintitle="CurveDialog test",
            options=dict(title=title, xlabel=self.labels[i0], ylabel="intensity"),
        )

        plot = self.slicewin.get_plot()
        QObject.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.update_fit)

        self.init_fit()
        if prefs.stepn > 1:
            zrgb = 255.0 / float(prefs.stepn - 1)
        else:
            zrgb = 0.0

        for istep in range(prefs.stepn):
            z2 = zmin + (istep + 1) * (zmax - zmin) / prefs.stepn
            k2 = int(round((z2 - prefs.absmins[i2]) / prefs.steps[i2]))  # borne sup [

            scan_values = np.zeros(ky)  # a scan
            scan_conts = np.zeros(ky)  # number of contributing original pixels

            red = int(round(istep * zrgb))
            green = 0
            blue = int(round((prefs.stepn - 1 - istep) * zrgb))

            print("slice between", k1, k2)
            for k in range(k1, k2):
                # center of the rod
                data_sli = data[:, :, k]
                cont_sli = cont[:, :, k]
                z = prefs.absmins[i2] + k * prefs.steps[i2]

                # center of the rod
                x = pos1[i0] + (pos2[i0] - pos1[i0]) * (z - pos1[i2]) / (
                    pos2[i2] - pos1[i2]
                )
                y = pos1[i1] + (pos2[i1] - pos1[i1]) * (z - pos1[i2]) / (
                    pos2[i2] - pos1[i2]
                )
                x1 = x - prefs.ranges[0] / 2
                x2 = x + prefs.ranges[0] / 2
                y1 = y - prefs.ranges[1] / 2
                y2 = y + prefs.ranges[1] / 2

                ix1 = int(round((x1 - prefs.absmins[i0]) / prefs.steps[i0]))
                ix2 = int(round((x2 - prefs.absmins[i0]) / prefs.steps[i0]))
                iy1 = int(round((y1 - prefs.absmins[i1]) / prefs.steps[i1]))
                iy2 = int(round((y2 - prefs.absmins[i1]) / prefs.steps[i1]))
                print("k=", k, "z=", z, "ix1,ix2,iy1,iy2=", ix1, ix2, iy1, iy2)

                if k == k1:
                    ix1deb = ix1
                    ix2deb = ix2
                    iy1deb = iy1
                    iy2deb = iy2
                    zdeb = z
                if k == k2 - 1:
                    ix1fin = ix1
                    ix2fin = ix2
                    iy1fin = iy1
                    iy2fin = iy2
                    zfin = z

                if ix1 < 0:
                    ix1 = 0  # the slice will be smaller
                if ix2 > data_sli.shape[0]:
                    ix2 = data_sli.shape[0]  # the slice will be smaller

                if iy1 >= 0 and iy2 <= data_sli.shape[1]:
                    scan_values = (
                        scan_values
                        + np.sum(data_sli[ix1:ix2, iy1:iy2], axis=0)
                        * (ix2 - ix1)
                        * prefs.ranges[0]
                    )
                    scan_conts = (
                        scan_conts
                        + np.sum(cont_sli[ix1:ix2, iy1:iy2], axis=0)
                        * (ix2 - ix1)
                        * prefs.ranges[0]
                    )
                else:
                    # we have to ajust the position of the slice
                    if iy1 < 0:
                        idec1 = -iy1
                        iy1 = 0
                    else:
                        idec1 = 0

                    if iy2 > data_sli.shape[1]:
                        idec2 = ky - (iy2 - data_sli.shape[1])
                        iy2 = data_sli.shape[1]
                    else:
                        idec2 = ky

                    scan_values[idec1:idec2] = (
                        scan_values[idec1:idec2]
                        + np.sum(data_sli[ix1:ix2, iy1:iy2], axis=0)
                        * (ix2 - ix1)
                        * prefs.ranges[0]
                    )
                    scan_conts[idec1:idec2] = scan_conts[idec1:idec2] + np.sum(
                        cont_sli[ix1:ix2, iy1:iy2], axis=0
                    )

            # xvalues
            x1deb = prefs.absmins[i0] + ix1deb * prefs.steps[i0]
            x1fin = prefs.absmins[i0] + ix1fin * prefs.steps[i0]
            x2deb = prefs.absmins[i0] + (ix2deb - 1) * prefs.steps[i0]
            x2fin = prefs.absmins[i0] + (ix2fin - 1) * prefs.steps[i0]
            x1 = (x1deb + x1fin) / 2.0
            x2 = (x2deb + x2fin) / 2.0
            xcen = (x1 + x2) / 2.0

            # yvalues
            y1deb = prefs.absmins[i1] + iy1deb * prefs.steps[i1]
            y1fin = prefs.absmins[i1] + iy1fin * prefs.steps[i1]
            y2deb = prefs.absmins[i1] + (iy2deb - 1) * prefs.steps[i1]
            y2fin = prefs.absmins[i1] + (iy2fin - 1) * prefs.steps[i1]
            y1 = (y1deb + y1fin) / 2.0
            y2 = (y2deb + y2fin) / 2.0
            ycen = (y1 + y2) / 2.0

            zcen = (zdeb + zfin) / 2.0

            abscissae = np.arange(ky) * prefs.steps[i1] + y1
            title = "scan_" + self.labels[i2] + "=%f" % zcen

            tag1 = self.labels[i0] + "_center"
            tag2 = self.labels[i1] + "_center"
            tag3 = self.labels[i2] + "_center"
            tag4 = self.labels[i0] + "_range"

            ydata = np.ma.masked_where(scan_conts == 0, scan_values)
            ydata = ydata / scan_conts
            xdata = np.ma.masked_where(scan_conts == 0, abscissae)

            item = make.curve(
                xdata.compressed(),
                ydata.compressed(),
                title=title,
                color=QColor(red, green, blue),
            )
            item.xlabel = self.labels[i1]

            item.tags = [
                [tag1, xcen],
                [tag2, ycen],
                [tag3, zcen],
                [tag4, prefs.ranges[0]],
            ]

            plot.add_item(item)

            k1 = k2

        self.figfit.setsavedir(self.openfiletool.directory)
        self.slicewin.show()
        self.slicewin.exec_()

    def init_fit(self):
        # ici on inclue une figure de fit provenant de grafit
        if self.isafit == 0:
            from grafit2.grafit2 import Ui_FitWindow

            #           import grafit2 as grafit
            self.figfit = Ui_FitWindow()
            #           self.figfit=grafit.Ui_FitWindow()
            self.figfit.setupUi()
            # tags est une liste de couples [non,valeur]
            self.figfit.settags([], saveint=True)
            self.isafit = 1
            # ici on redirige le signal de fermeture de la fenetre de fit
            self.figfit.closeEvent = self.close_fit

        self.figfit.move(10, 500)
        self.figfit.show()
        # self.update_fit()

    def set_fit(self):
        self.init_fit()
        QObject.connect(
            self.p_panel, SIGNAL("active_plot_changed()"), self.update_curve_fit
        )

    def update_curve_fit(self):
        plot = self.p_panel.active_plot
        curve = plot.get_items(item_type=ICurveItemType)[0]
        if self.isafit:
            self.show_curve_fit(curve)

    def show_curve_fit(self, item):
        if item is not None:
            # try:
            x, y = item.get_data()
            ylabel = "intensity"
            title = str(self.windowTitle())
            self.figfit.setvalexp(x, y, xlabel=item.xlabel, ylabel=ylabel, title=title)
            self.figfit.show()
        # except:
        # print "error in show_curve_fit, unable to display curve to be fitted"
        # pass

    def show_fit(self, item):
        if item is not None:
            try:
                x, y = item.get_data()
                ylabel = "intensity"
                self.figfit.setvalexp(
                    x,
                    y,
                    xlabel=item.xlabel,
                    ylabel=ylabel,
                    tags=item.tags,
                    title=item.curveparam.label,
                )
                self.figfit.show()
            except:
                print("error in show_fit, unable to display curve to be fitted")
                pass

    def update_fit(self, plot):
        itemselected = plot.get_active_item(force=False)
        if self.isafit:
            self.show_fit(itemselected)

    def close_fit(self, closeevent):
        self.isafit = 0


def test():
    """Test"""
    # -- Create QApplication
    import guidata

    _app = guidata.qapplication()
    # --

    win = Image3DDialog()
    # win.get_plot().manager.get_tool(AspectRatioTool).lock_action.setChecked(False)
    win.get_plot().set_aspect_ratio(lock=False)
    win.resize(1200, 800)
    # win.open_file('sample2a_295-295.hdf5')
    # win.openfiletool.directory='G:/Documents Geoffroy PREVOT/Ag-Si/SIXS-SiAg(110)-Juillet2019/binoculars V2'
    # win.open_file('G:/Documents Geoffroy PREVOT/Ag-Si/SIXS-SiAg(110)-Juillet2019/binoculars V2/sample2a_924-942_[m5.00-5.00,m1.55-m1.45,0.00-2.00].hdf5')
    win.openfiletool.directory = "G:/Documents Geoffroy PREVOT/ANR-Germanene-2017/Manip SOLEIL nov 2019/binoculars depot1 corrige/hdf5"
    win.open_file(
        "G:/Documents Geoffroy PREVOT/ANR-Germanene-2017/Manip SOLEIL nov 2019/binoculars depot1 corrige/hdf5/hdf5 in plan 0.1 et 2.5/0.1/Al111_depot1_258-258.hdf5"
    )

    win.setGeometry(10, 35, 1250, 900)
    # win.open_file('Au111_noir_na=2_nb=1_188-281.hdf5')
    # win.p_panel.set_ranges([[4.145,4.306],[-0.057,0.060],[0.05,3.90]],dosignal=True)
    # win.update_image()

    # win.open_file('test_2186-2201_180.hdf5')
    win.show()
    # SetSliceWindow2(win.slicetool.prefs)
    _app.exec_()


if __name__ == "__main__":
    from guidata.configtools import add_image_path

    abspath = osp.abspath(__file__)
    dirpath = osp.dirname(abspath)
    add_image_path(dirpath)
    # test()
