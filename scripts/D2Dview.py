# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:42:19 2014

@author: Prevot
"""
import sys
import os.path as osp
import pickle

import numpy as np
import scipy.optimize as opt

from os import getcwd, access, R_OK
from math import sqrt, sin, cos, pi

from guidata.qt.QtGui import (
    QPen,
    QBrush,
    QPolygonF,
    QCheckBox,
    QTransform,
    QPainter,
    QComboBox,
    QColor,
    QPushButton,
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
)
from guidata.qt.QtCore import Qt, QPointF, QSize, QRect, SIGNAL, QObject
from matplotlib.path import Path as mplPath
from guidata.utils import assert_interfaces_valid, update_dataset

# Local imports
from guiqwt.transitional import QwtSymbol
from guiqwt.config import CONF, _
from guiqwt.interfaces import (
    IBasePlotItem,
    IShapeItemType,
    ISerializableType,
    IColormapImageItemType,
)
from guiqwt.styles import ShapeParam, AxesShapeParam
from guiqwt.signals import SIG_ITEM_REMOVED, SIG_ITEMS_CHANGED
from guiqwt.baseplot import canvas_to_axes
from guiqwt.shapes import AbstractShape, PolygonShape, RectangleShape, PointShape

from guiqwt.plot import ImageDialog
from guiqwt.tools import RectangularShapeTool
from guiqwt.tools import CommandTool, DefaultToolbarID
from guiqwt.builder import make
from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import (
    BoolItem,
    FloatItem,
    IntItem,
    ImageChoiceItem,
    StringItem,
)
from PIL import Image

_fromUtf8 = lambda s: s

abspath = osp.abspath(__file__)
dirpath = osp.dirname(abspath)
abspath = getcwd()
reconstructionpath = osp.join(dirpath, "reconstruction.png")
distorsionpath = osp.join(dirpath, "distorsion.png")
gridpath = osp.join(dirpath, "grid.png")
peakfindpath = osp.join(dirpath, "peakfind.png")


cos_60 = 0.5
sin_60 = np.sqrt(3.0) / 2.0

DEFAULTS = {
    "plot": {
        "shape/gridshape/line/style": "SolidLine",
        "shape/gridshape/line/color": "#ff0000",
        "shape/gridshape/line/width": 1,
        "shape/gridshape/fill/style": "SolidPattern",
        "shape/gridshape/fill/color": "white",
        "shape/gridshape/fill/alpha": 0.1,
        "shape/gridshape/symbol/marker": "Ellipse",
        "shape/gridshape/symbol/size": 10,
        "shape/gridshape/symbol/edgecolor": "#ff0000",
        "shape/gridshape/symbol/facecolor": "#ff0000",
        "shape/gridshape/symbol/alpha": 0.0,
        "shape/gridshape/sel_line/style": "SolidLine",
        "shape/gridshape/sel_line/color": "#00ff00",
        "shape/gridshape/sel_line/width": 1,
        "shape/gridshape/sel_fill/style": "SolidPattern",
        "shape/gridshape/sel_fill/color": "white",
        "shape/gridshape/sel_fill/alpha": 0.1,
        "shape/gridshape/sel_symbol/marker": "Ellipse",
        "shape/gridshape/sel_symbol/size": 10,
        "shape/gridshape/sel_symbol/edgecolor": "#00aa00",
        "shape/gridshape/sel_symbol/facecolor": "#00ff00",
        "shape/gridshape/sel_symbol/alpha": 0.7,
        "shape/gridshape/xarrow_pen/style": "SolidLine",
        "shape/gridshape/xarrow_pen/color": "red",
        "shape/gridshape/xarrow_pen/width": 1,
        "shape/gridshape/xarrow_brush/color": "red",
        "shape/gridshape/xarrow_brush/alpha": 0.2,
        "shape/gridshape/yarrow_pen/style": "SolidLine",
        "shape/gridshape/yarrow_pen/color": "green",
        "shape/gridshape/yarrow_pen/width": 1,
        "shape/gridshape/yarrow_brush/color": "green",
        "shape/gridshape/yarrow_brush/alpha": 0.2,
        "shape/spot/line/style": "SolidLine",
        "shape/spot/line/color": "#48b427",
        "shape/spot/line/width": 1,
        "shape/spot/sel_line/style": "SolidLine",
        "shape/spot/sel_line/color": "#00ff00",
        "shape/spot/sel_line/width": 1,
        "shape/spot/fill/style": "NoBrush",
        "shape/spot/sel_fill/style": "NoBrush",
        "shape/spot/symbol/marker": "XCross",
        "shape/spot/symbol/size": 9,
        "shape/spot/symbol/edgecolor": "#48b427",
        "shape/spot/symbol/facecolor": "#48b427",
        "shape/spot/symbol/alpha": 1.0,
        "shape/spot/sel_symbol/marker": "XCross",
        "shape/spot/sel_symbol/size": 12,
        "shape/spot/sel_symbol/edgecolor": "#00aa00",
        "shape/spot/sel_symbol/facecolor": "#00ff00",
        "shape/spot/sel_symbol/alpha": 0.7,
        "shape/reconstructionshape/line/style": "NoPen",
        "shape/reconstructionshape/line/color": "#0000ff",
        "shape/reconstructionshape/line/width": 2,
        "shape/reconstructionshape/fill/style": "SolidPattern",
        "shape/reconstructionshape/fill/color": "white",
        "shape/reconstructionshape/fill/alpha": 0.1,
        "shape/reconstructionshape/symbol/marker": "Ellipse",
        "shape/reconstructionshape/symbol/size": 8,
        "shape/reconstructionshape/symbol/edgecolor": "#0000ff",
        "shape/reconstructionshape/symbol/facecolor": "#0000ff",
        "shape/reconstructionshape/symbol/alpha": 0.0,
        "shape/reconstructionshape/sel_line/style": "DashLine",
        "shape/reconstructionshape/sel_line/color": "#00ff00",
        "shape/reconstructionshape/sel_line/width": 1,
        "shape/reconstructionshape/sel_fill/style": "SolidPattern",
        "shape/reconstructionshape/sel_fill/color": "white",
        "shape/reconstructionshape/sel_fill/alpha": 0.1,
        "shape/reconstructionshape/sel_symbol/marker": "Ellipse",
        "shape/reconstructionshape/sel_symbol/size": 8,
        "shape/reconstructionshape/sel_symbol/edgecolor": "#00aa00",
        "shape/reconstructionshape/sel_symbol/facecolor": "#00ff00",
        "shape/reconstructionshape/sel_symbol/alpha": 0.7,
        "shape/reconstructionshape/xarrow_pen/style": "SolidLine",
        "shape/reconstructionshape/xarrow_pen/color": "red",
        "shape/reconstructionshape/xarrow_pen/width": 1,
        "shape/reconstructionshape/xarrow_brush/color": "red",
        "shape/reconstructionshape/xarrow_brush/alpha": 0.2,
        "shape/reconstructionshape/yarrow_pen/style": "SolidLine",
        "shape/reconstructionshape/yarrow_pen/color": "green",
        "shape/reconstructionshape/yarrow_pen/width": 1,
        "shape/reconstructionshape/yarrow_brush/color": "green",
        "shape/reconstructionshape/yarrow_brush/alpha": 0.2,
    },
}

CONF.update_defaults(DEFAULTS)

SPACEGROUP_CHOICES = [
    ("p1", _("p1"), "none.png"),
    ("p2", _("p2"), "none.png"),
    ("pm", _("pm"), "none.png"),
    ("pg", _("pg"), "none.png"),
    ("cm", _("cm"), "none.png"),
    ("p2mm", _("p2mm"), "none.png"),
    ("p2mg", _("p2mg"), "none.png"),
    ("p2gg", _("p2gg"), "none.png"),
    ("c2mm", _("c2mm"), "none.png"),
    ("p4", _("p4"), "none.png"),
    ("p4mm", _("p4mm"), "none.png"),
    ("p4gm", _("p4gm"), "none.png"),
    ("p3", _("p3"), "none.png"),
    ("p3m1", _("p3m1"), "none.png"),
    ("p31m", _("p31m"), "none.png"),
    ("p6", _("p6"), "none.png"),
    ("p6mm", _("p6mm"), "none.png"),
]


def xy_to_angle(ux, uy):
    return np.sqrt(ux ** 2 + uy ** 2), np.rad2deg(np.arctan2(uy, ux))


def angle_to_xy(r, t):
    tt = np.deg2rad(t)
    return r * np.cos(tt), r * np.sin(tt)


def corr_lin(p, x, y):
    # hi est soit h,soit k en pixel, retourne ki qui est soit kx, soit ky
    return p[0] + p[1] * x + p[2] * y


def corr_quad(p, x, y):
    # h et k sont les pixels, retourne la valeur soit de kx, soit de ky
    return p[0] + p[1] * x + p[2] * y + p[3] * x * x + p[4] * x * y + p[5] * x * y


def corr_cub(p, x, y):
    # h et k sont les pixels, retourne la valeur soit de kx, soit de ky
    return (
        p[0]
        + p[1] * x
        + p[2] * y
        + p[3] * x * x
        + p[4] * x * y
        + p[5] * x * y
        + p[6] * x * x * x
        + p[7] * x * x * y
        + p[8] * x * y * y
        + p[9] * y * y * y
    )


def erreur_lin(p, x, y, s):
    # x et y en pixel, s est la coordonnee corrigee soit x, soit y
    return corr_lin(p, x, y) - s


def erreur_quad(p, x, y, s):
    # x et y en pixel, s est la coordonnee corrigee soit x, soit y
    return corr_quad(p, x, y) - s


def erreur_cub(p, x, y, s):
    # x et y en pixel, s est la coordonnee corrigee soit x, soit y
    return corr_cub(p, x, y) - s


def Gaussian(x, amplitude, xo, sigma, offset):
    g = offset + amplitude * np.exp(-(((x - xo) / sigma) ** 2))
    return g.ravel()


def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (
        4 * sigma_y ** 2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def twoD_iso_Gaussian(x, y, amplitude, xo, yo, sigma, offset):
    xo = float(xo)
    yo = float(yo)
    a = 1.0 / (2 * sigma ** 2)
    g = offset + amplitude * np.exp(-a * (((x - xo) ** 2) + ((y - yo) ** 2)))
    return g.ravel()


class ReconstructionShapeParam(DataSet):
    A0 = FloatItem(_("Ax"), default=2.0)
    A1 = FloatItem(_("Ay"), default=0.0)
    B0 = FloatItem(_("Bx"), default=0.0)
    B1 = FloatItem(_("By"), default=2.0)
    order = IntItem(_("order"), default=1)

    def update_param(self, obj):
        self.A0 = obj.A[0]
        self.A1 = obj.A[1]
        self.B0 = obj.B[0]
        self.B1 = obj.B[1]
        self.order = obj.order

    def update_shape(self, obj):
        obj.A[0] = self.A0
        obj.A[1] = self.A1
        obj.B[0] = self.B0
        obj.B[1] = self.B1
        obj.order = self.order
        self.update_param(obj)


class GridShapeParam(DataSet):
    th = chr(952)
    O0 = FloatItem(_("Ox"), default=200.0)
    O1 = FloatItem(_("Oy"), default=200.0).set_pos(col=1)

    RT = BoolItem(_("(R," + th + ")"), default=False)
    vm = BoolItem(_("v master"), default=False).set_pos(col=1)

    ux = FloatItem(_("ux"), default=100.0)
    uR = FloatItem(_("uR"), default=100.0).set_pos(col=1)
    uy = FloatItem(_("uy"), default=100.0)
    uT = FloatItem(_("u" + th), default=100.0).set_pos(col=1)

    vx = FloatItem(_("vx"), default=100.0)
    vR = FloatItem(_("vR"), default=100.0).set_pos(col=1)

    vy = FloatItem(_("vy"), default=100.0)
    vT = FloatItem(_("v" + th), default=100.0).set_pos(col=1)

    order = IntItem(_("order"), default=1)
    sg = ImageChoiceItem(_("SpaceGroup"), SPACEGROUP_CHOICES, default="p4mm")

    def update_param(self, obj):
        self.O0 = obj.O[0]
        self.O1 = obj.O[1]
        self.ux = obj.u[0]
        self.uy = obj.u[1]
        self.vx = obj.v[0]
        self.vy = obj.v[1]
        self.uR = obj.uR
        self.uT = obj.uT
        self.vR = obj.vR
        self.vT = obj.vT
        self.order = obj.order
        self.sg = obj.sg

    def update_shape(self, obj):

        obj.O[0] = self.O0
        obj.O[1] = self.O1
        """
        if self.RT:
          self.ux,self.uy=angle_to_xy(self.uR,self.uT)
          self.vx,self.vy=angle_to_xy(self.vR,self.vT)
        else:
          self.uR,self.uT=xy_to_angle(self.ux,self.uy)
          self.vR,self.vT=xy_to_angle(self.vx,self.vy)
        """
        obj.u[0] = self.ux
        obj.u[1] = self.uy
        obj.v[0] = self.vx
        obj.v[1] = self.vy
        obj.uR = self.uR
        obj.uT = self.uT
        obj.vR = self.vR
        obj.vT = self.vT

        obj.order = self.order
        obj.sg = self.sg
        if self.vm:
            obj.set_space_group_constraints(master="v", RT=self.RT)
        else:
            obj.set_space_group_constraints(master="u", RT=self.RT)
        self.update_param(obj)


class SpotShapeParam(DataSet):
    x0 = FloatItem(_("x0"), default=200.0)
    y0 = FloatItem(_("y0"), default=200.0)
    h = FloatItem(_("h"), default=100.0)
    k = FloatItem(_("k"), default=100.0)
    gridname = StringItem(_("Grid"), default="Enter grid name")

    def update_param(self, obj):
        self.x0 = obj.x0
        self.y0 = obj.y0
        self.h = obj.h
        self.k = obj.k
        self.gridname = obj.gridname

    def update_shape(self, obj):
        obj.x0 = self.x0
        obj.y0 = self.y0
        obj.h = self.h
        obj.k = self.k
        obj.gridname = self.gridname
        self.update_param(obj)


class SpotShape(PointShape):
    # un point pour marquer une tache de diffraction
    # identique a PointShape avec h,k en plus et la reference a la reconstruction (grid qui est soit une gridshape,soit une reconstructionshape)
    CLOSED = False

    def __init__(
        self,
        x0=0,
        y0=0,
        shapeparam=None,
        spotshapeparam=None,
        gridname="",
        grid=None,
        h=None,
        k=None,
    ):
        super(SpotShape, self).__init__(x0, y0, shapeparam=shapeparam)
        self.gridname = gridname
        self.grid = grid
        self.h = h
        self.k = k
        self.x0 = x0
        self.y0 = y0

        if spotshapeparam is None:
            self.spotshapeparam = SpotShapeParam(_("SpotShape"), icon="point.png")
        else:
            self.spotshapeparam = spotshapeparam
            self.spotshapeparam.update_shape(self)

    def move_point_to(self, handle, pos, ctrl=None):
        pass
        # nx, ny = pos
        # self.points[0] = (nx, ny)

    def set_pos(self, x0, y0):
        """Set the point coordinates to (x, y)"""
        self.set_points([(x0, y0)])
        self.x0 = x0
        self.y0 = y0

    def __reduce__(self):
        self.shapeparam.update_param(self)
        state = (self.shapeparam, self.points, self.h, self.k, self.gridname, self.z())
        return (SpotShape, (), state)

    def __getstate__(self):
        return self.shapeparam, self.points, self.h, self.k, self.gridname, self.z()

    def __setstate__(self, state):
        param, points, h, k, gridname, z = state
        self.points = points
        self.x0 = points[0, 0]
        self.y0 = points[0, 1]
        self.h = h
        self.k = k
        self.setZ(z)
        self.shapeparam = param
        self.gridname = gridname
        self.shapeparam.update_shape(self)

    def get_item_parameters(self, itemparams):

        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)

        self.spotshapeparam = SpotShapeParam(_("SpotShape"), icon="point.png")

        self.spotshapeparam.update_param(self)
        itemparams.add("SpotShapeParam", self, self.spotshapeparam)

    def set_grid_from_gridname(self, gridname):
        # permet de recuperer le reseau a partir de son nom s'il existe
        items = list(
            [
                item
                for item in self.plot().get_items()
                if (
                    isinstance(item, GridShape) or isinstance(item, ReconstructionShape)
                )
            ]
        )
        itemnames = list([item.title().text() for item in items])
        if gridname in itemnames:
            self.grid = items[itemnames.index(self.gridname)]

    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"), visible_only=True)
        gridname = self.gridname
        self.shapeparam.update_shape(self)
        # on verifie que self.grid existe bien
        items = list(
            [
                item
                for item in self.plot().get_items()
                if (
                    isinstance(item, GridShape) or isinstance(item, ReconstructionShape)
                )
            ]
        )
        itemnames = list([item.title().text() for item in items])
        # modification des parametres specifiques de la gridshape
        self.spotshapeparam.update_shape(self)

        if self.gridname not in itemnames:
            params = PeakIdentificationParameters(self.plot())
            params.guess(self.x0, self.y0)
            Ok = PeakIdentificationWindow(params).exec_()

            if Ok:
                self.gridname = params.itemname
                self.grid = params.item
                self.h = params.h
                self.k = params.k
            else:
                self.gridname = gridname

        else:
            i = itemnames.index(self.gridname)
            self.grid = items[i]
        self.setTitle(self.gridname + "(%d,%d)" % (self.h, self.k))


assert_interfaces_valid(SpotShape)


class PointCloudShape(AbstractShape):
    __implements__ = (IBasePlotItem, ISerializableType)
    ADDITIONNAL_POINTS = 0  # Number of points which are not part of the shape
    LINK_ADDITIONNAL_POINTS = False  # Link additionnal points with dotted lines
    CLOSED = True

    def __init__(self, points=None, connects=None, closed=None, shapeparam=None):
        super(PointCloudShape, self).__init__(None)
        self.closed = self.CLOSED if closed is None else closed
        self.selected = False

        if shapeparam is None:
            self.shapeparam = ShapeParam(_("Shape"), icon="rectangle.png")
        else:
            self.shapeparam = shapeparam
            self.shapeparam.update_shape(self)

        self.pen = QPen()
        self.brush = QBrush()
        self.symbol = QwtSymbol.NoSymbol
        self.sel_pen = QPen()
        self.sel_brush = QBrush()
        self.sel_symbol = QwtSymbol.NoSymbol
        self.points = np.zeros((0, 2), float)
        if points is not None:
            self.set_points(points)
        if connects is not None:
            self.set_connects(connects)

    def types(self):
        return (IShapeItemType, ISerializableType)

    def __reduce__(self):
        self.shapeparam.update_param(self)
        state = (self.shapeparam, self.points, self.connects, self.closed, self.z())
        return (PointCloudShape, (), state)

    def __getstate__(self):
        return self.shapeparam, self.points, self.connects, self.closed, self.z()

    def __setstate__(self, state):
        param, points, connects, closed, z = state
        self.points = points
        self.connects = connects
        self.setZ(z)
        self.shapeparam = param
        self.shapeparam.update_shape(self)

    def serialize(self, writer):
        """Serialize object to HDF5 writer"""
        self.shapeparam.update_param(self)
        writer.write(self.shapeparam, group_name="shapeparam")
        writer.write(self.points, group_name="points")
        writer.write(self.closed, group_name="closed")
        writer.write(self.z(), group_name="z")

    def deserialize(self, reader):
        """Deserialize object from HDF5 reader"""
        self.closed = reader.read("closed")
        self.shapeparam = ShapeParam(_("Shape"), icon="rectangle.png")
        reader.read("shapeparam", instance=self.shapeparam)
        self.shapeparam.update_shape(self)
        self.points = reader.read(group_name="points", func=reader.read_array)
        self.setZ(reader.read("z"))

    # ----Public API-------------------------------------------------------------

    def set_style(self, section, option):
        self.shapeparam.read_config(CONF, section, option)
        self.shapeparam.update_shape(self)

    def set_color(self, color):
        # a convenient way to change all color attributes simultaneously with the same color
        self.pen.setColor(color)
        self.symbol.setPen(color)
        self.symbol.setBrush(color)
        # self.brush.setColor(color)
        self.sel_pen.setColor(color)
        self.sel_symbol.setPen(color)
        self.sel_symbol.setBrush(color)
        self.shapeparam.update_param(self)
        self.shapeparam.update_shape(self)
        # self.sel_brush.setColor(color)

    def set_points(self, points):
        self.points = np.array(points, float)
        assert self.points.shape[1] == 2

    def set_connects(self, connects):
        self.connects = np.array(connects, float)
        assert self.points.shape[1] == 2

    def get_points(self):
        """Return polygon points"""
        return self.points

    def get_bounding_rect_coords(self):
        """Return bounding rectangle coordinates (in plot coordinates)"""
        poly = QPolygonF()
        shape_points = self.points[: -self.ADDITIONNAL_POINTS]
        for i in range(shape_points.shape[0]):
            poly.append(QPointF(shape_points[i, 0], shape_points[i, 1]))
        return poly.boundingRect().getCoords()

    def transform_points(self, xMap, yMap):
        points = QPolygonF()
        for i in range(self.points.shape[0]):
            points.append(
                QPointF(
                    xMap.transform(self.points[i, 0]), yMap.transform(self.points[i, 1])
                )
            )
        return points

    def transform_connects(self, points):
        lines = list()
        for i in range(len(self.connects)):
            i1 = self.connects[i][0]
            i2 = self.connects[i][1]
            lines.append(points[i1])
            lines.append(points[i2])
        return lines

    def get_reference_point(self):
        if self.points.size:
            return self.points.mean(axis=0)

    def get_pen_brush(self, xMap, yMap):
        if self.selected:
            pen = self.sel_pen
            brush = self.sel_brush
            sym = self.sel_symbol
        else:
            pen = self.pen
            brush = self.brush
            sym = self.symbol
        if self.points.size > 0:
            x0, y0 = self.get_reference_point()
            xx0 = xMap.transform(x0)
            yy0 = yMap.transform(y0)
            try:
                # Optimized version in PyQt >= v4.5
                t0 = QTransform.fromTranslate(xx0, yy0)
            except AttributeError:
                # Fallback for PyQt <= v4.4
                t0 = QTransform().translate(xx0, yy0)
            tr = brush.transform()
            tr = tr * t0
            brush = QBrush(brush)
            brush.setTransform(tr)
        return pen, brush, sym

    def draw(self, painter, xMap, yMap, canvasRect):
        pen, brush, symbol = self.get_pen_brush(xMap, yMap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.setBrush(brush)
        points = self.transform_points(xMap, yMap)

        # connection des segments
        shape_lines = self.transform_connects(points)
        painter.drawLines(shape_lines)

        """
        if self.ADDITIONNAL_POINTS:
            shape_points = points[:-self.ADDITIONNAL_POINTS]
            other_points = points[-self.ADDITIONNAL_POINTS:]
        else:
            shape_points = points
            other_points = []
        if self.closed:
            painter.drawPolygon(shape_points)
        else:
            painter.drawPolyline(shape_points)
        """
        if symbol != QwtSymbol.NoSymbol:
            for i in range(points.size()):
                symbol.draw(painter, points[i].toPoint())
        """
        if self.LINK_ADDITIONNAL_POINTS and other_points:
            pen2 = painter.pen()
            pen2.setStyle(Qt.DotLine)
            painter.setPen(pen2)
            painter.drawPolyline(other_points)
        """

    def poly_hit_test(self, plot, ax, ay, pos):
        pos = QPointF(pos)
        dist = sys.maxint
        handle = -1
        Cx, Cy = pos.x(), pos.y()
        poly = QPolygonF()
        pts = self.points
        for i in range(pts.shape[0]):
            # On calcule la distance dans le repère du canvas
            px = plot.transform(ax, pts[i, 0])
            py = plot.transform(ay, pts[i, 1])
            if i < pts.shape[0] - self.ADDITIONNAL_POINTS:
                poly.append(QPointF(px, py))
            d = (Cx - px) ** 2 + (Cy - py) ** 2
            if d < dist:
                dist = d
                handle = i
        inside = poly.containsPoint(QPointF(Cx, Cy), Qt.OddEvenFill)
        return sqrt(dist), handle, inside, None

    def hit_test(self, pos):
        """return (dist, handle, inside)"""
        if not self.plot():
            return sys.maxint, 0, False, None
        return self.poly_hit_test(self.plot(), self.xAxis(), self.yAxis(), pos)

    def add_local_point(self, pos):
        pt = canvas_to_axes(self, pos)
        return self.add_point(pt)

    def add_point(self, pt):
        N, _ = self.points.shape
        self.points = np.resize(self.points, (N + 1, 2))
        self.points[N, :] = pt
        return N

    def del_point(self, handle):
        self.points = np.delete(self.points, handle, 0)
        if handle < len(self.points):
            return handle
        else:
            return self.points.shape[0] - 1

    def move_point_to(self, handle, pos, ctrl=None):
        # self.points[handle, :] = pos
        # ici il faut ecrire le comportement de la shape
        if handle == 0:
            pos0 = self.points[handle, :]
            dx = pos[0] - pos0[0]
            dy = pos[1] - pos0[1]
            for i in range(len(self.points)):
                self.points[i, 0] = self.points[i, 0] + dx
                self.points[i, 1] = self.points[i, 1] + dy
        else:
            # regarder uniquement s'il s'agit des axes
            x0 = self.points[0, 0]
            y0 = self.points[0, 1]
            xi = self.points[handle, 0]
            yi = self.points[handle, 1]

            r0 = np.sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0))
            xf = pos[0]
            yf = pos[1]

            rf = np.sqrt((xf - x0) * (xf - x0) + (yf - y0) * (yf - y0))
            fact = rf / r0

            for i in range(1, len(self.points)):
                self.points[i, 0] = (self.points[i, 0] - x0) * fact + x0
                self.points[i, 1] = (self.points[i, 1] - y0) * fact + y0

    def move_shape(self, old_pos, new_pos):
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        self.points += np.array([[dx, dy]])

    def get_item_parameters(self, itemparams):
        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)

    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"), visible_only=True)
        self.shapeparam.update_shape(self)


assert_interfaces_valid(PointCloudShape)


class GridShape(PointCloudShape, PolygonShape):
    # une classe particuliere de pointcloudshape
    def __init__(
        self,
        O=[0, 0],
        u=[100, 0],
        v=[0, 100],
        order=1,
        sg="p4",
        axesparam=None,
        shapeparam=None,
        gridshapeparam=None,
    ):
        super(GridShape, self).__init__(shapeparam=shapeparam)
        self.arrow_angle = 15  # degrees
        self.arrow_size = 10  # pixels
        self.x_pen = self.pen
        self.x_brush = self.brush
        self.y_pen = self.pen
        self.y_brush = self.brush
        self.set_movable(False)

        if axesparam is None:
            self.axesparam = AxesShapeParam(_("Axes"), icon="gtaxes.png")
        else:
            self.axesparam = axesparam
            self.axesparam.update_param(self)

        if shapeparam is None:
            self.shapeparam = ShapeParam(_("Shape"), icon="rectangle.png")
        else:
            self.shapeparam = shapeparam
            self.shapeparam.update_shape(self)

        if gridshapeparam is None:
            self.gridshapeparam = GridShapeParam(_("GridShape"), icon="rectangle.png")
        else:
            self.gridshapeparam = gridshapeparam
            self.gridshapeparam.update_shape(self)

        self.O = np.array(O, float)  # centre du reseau
        self.u = np.array(u, float)  # vecteur 1
        self.v = np.array(v, float)  # vecteur 2

        self.uR, self.uT = xy_to_angle(self.u[0], self.u[1])
        self.vR, self.vT = xy_to_angle(self.v[0], self.v[1])

        self.order = order  # nombre de mailles
        self.sg = sg  # space group
        self.slaves = list()

        self.set_nodes()
        self.set_points()
        self.set_connects()
        self.define_points()

    def __reduce__(self):
        self.shapeparam.update_param(self)
        state = (
            self.shapeparam,
            self.O,
            self.u,
            self.v,
            self.order,
            self.sg,
            self.slaves,
            self.z(),
        )
        return (GridShape, (), state)

    def __getstate__(self):
        return (
            self.shapeparam,
            self.O,
            self.u,
            self.v,
            self.order,
            self.sg,
            self.slaves,
            self.z(),
        )

    def __setstate__(self, state):
        # quand on fait un pickle, on charge aussi les slaves
        param, O, u, v, order, sg, slaves, z = state
        self.O = O
        self.u = u
        self.v = v
        self.order = order
        self.slaves = slaves
        for slave in self.slaves:
            slave.set_master(self)
        self.sg = sg
        self.setZ(z)
        self.shapeparam = param
        self.shapeparam.update_shape(self)

        self.set_nodes()
        self.set_points()
        self.set_connects()
        self.define_points()

    def add_slave(self, slave):
        self.slaves.append(slave)

    def remove_slave(self, slave):
        self.slaves.remove(slave)

    def set_nodes(self):
        self.nodes = list()
        # les trois premiers noeuds servent a definir le reseau
        self.nodes.append([0, 0])
        self.nodes.append([1, 0])
        self.nodes.append([0, 1])
        for i in range(-self.order, self.order + 1):
            for j in range(-self.order, self.order + 1):
                node = [i, j]
                if node not in self.nodes:
                    self.nodes.append(node)
        self.npts = len(self.nodes)

    def set_points(self):
        self.points = np.array(
            list(
                [
                    self.O[0] + self.u[0] * node[0] + self.v[0] * node[1],
                    self.O[1] + self.u[1] * node[0] + self.v[1] * node[1],
                ]
                for node in self.nodes
            ),
            float,
        )
        for slave in self.slaves:
            slave.set_points()

    def set_connects(self):
        self.connects = np.array(
            list(
                [i, j]
                for i in range(self.npts)
                for j in range(self.npts)
                if self.dist(i, j) == 1
            )
        )

    def define_points(self):
        # redefini les positions des points de la grille en fonction des vecteurs ux et uy
        for i in range(self.npts):
            node = self.nodes[i]
            self.points[i, 0] = self.O[0] + self.u[0] * node[0] + self.v[0] * node[1]
            self.points[i, 1] = self.O[1] + self.u[1] * node[0] + self.v[1] * node[1]

    def dist(self, i, j):
        ni = self.nodes[i]
        nj = self.nodes[j]
        return (ni[0] - nj[0]) ** 2 + (ni[1] - nj[1]) ** 2

    def set_space_group_constraints(self, master="u", RT=False):

        if RT:
            self.u[0], self.u[1] = angle_to_xy(self.uR, self.uT)
            self.v[0], self.v[1] = angle_to_xy(self.vR, self.vT)

        if master == "u":
            if self.sg in ["pm", "p2mm", "pg", "p2mg", "p2gg"]:
                # les axes doivent etre perpendiculaires
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(v2 / u2)
                self.v[0] = -self.u[1] * ratio
                self.v[1] = self.u[0] * ratio

            elif self.sg in ["cm", "c2mm"]:
                # les axes doivent etre orthonormes
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(u2 / v2)
                self.v[0] = self.v[0] * ratio
                self.v[1] = self.v[1] * ratio

            elif self.sg in ["p4", "p4mm", "p4gm"]:
                # les axes doivent etre orthonormes
                self.v[0] = -self.u[1]
                self.v[1] = self.u[0]

            elif self.sg in ["p3", "p3m1", "p31m", "p6", "p6mm"]:
                # les axes doivent etre orthonormes
                self.v[0] = self.u[0] * cos_60 - self.u[1] * sin_60
                self.v[1] = self.u[0] * sin_60 + self.u[1] * cos_60
        else:
            if self.sg in ["pm", "p2mm", "pg", "p2mg", "p2gg"]:
                # les axes doivent etre perpendiculaires
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(u2 / v2)
                self.u[0] = self.v[1] * ratio
                self.u[1] = -self.v[0] * ratio

            if self.sg in ["cm", "c2mm"]:
                # les axes doivent etre orthonormes
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(v2 / u2)
                self.u[0] = self.u[0] * ratio
                self.u[1] = self.u[1] * ratio

            if self.sg in ["p4", "p4mm", "p4gm"]:
                # les axes doivent etre orthonormes
                self.u[0] = self.v[1]
                self.u[1] = -self.v[0]

            if self.sg in ["p3", "p3m1", "p31m", "p6", "p6mm"]:
                # les axes doivent etre orthonormes
                self.u[0] = self.v[0] * cos_60 + self.v[1] * sin_60
                self.u[1] = -self.v[0] * sin_60 + self.v[1] * cos_60

        self.uR, self.uT = xy_to_angle(self.u[0], self.u[1])
        self.vR, self.vT = xy_to_angle(self.v[0], self.v[1])

    def move_point_to(self, handle, pos, ctrl=None):
        # self.points[handle, :] = pos
        # ici il faut ecrire le comportement de la shape
        if handle == 0:
            self.O = np.array(pos, float)
            self.define_points()

        elif handle == 1:
            # il s'agit de l'axe des x
            x0 = self.points[0, 0]
            y0 = self.points[0, 1]
            x1 = pos[0]
            y1 = pos[1]
            self.u[0] = x1 - x0
            self.u[1] = y1 - y0

            # ici on ecrit le comportement pour assurer que le groupe d'espace est le bon
            self.set_space_group_constraints("u")
            self.define_points()

        elif handle == 2:
            # il s'agit de l'axe des x
            x0 = self.points[0, 0]
            y0 = self.points[0, 1]
            x2 = pos[0]
            y2 = pos[1]
            self.v[0] = x2 - x0
            self.v[1] = y2 - y0
            self.set_space_group_constraints("v")
            self.define_points()

        for slave in self.slaves:
            slave.set_points()

    """
    def set_style(self, section, option):
        self.shapeparam.read_config(CONF, section, option)
        self.shapeparam.update_shape(self)
        self.axesparam.read_config(CONF, section, option)
        self.axesparam.update_axes(self)
    """

    def draw(self, painter, xMap, yMap, canvasRect):
        PointCloudShape.draw(self, painter, xMap, yMap, canvasRect)

        painter.setBrush(painter.pen().color())
        self.draw_arrow(painter, xMap, yMap, self.points[0], self.points[1])
        self.draw_arrow(painter, xMap, yMap, self.points[0], self.points[2])

    def draw_arrow(self, painter, xMap, yMap, p0, p1):

        sz = self.arrow_size
        angle = pi * self.arrow_angle / 180.0
        ca, sa = cos(angle), sin(angle)
        d1x = xMap.transform(p1[0]) - xMap.transform(p0[0])
        d1y = yMap.transform(p1[1]) - yMap.transform(p0[1])
        norm = sqrt(d1x ** 2 + d1y ** 2)
        if abs(norm) < 1e-6:
            return
        d1x *= sz / norm
        d1y *= sz / norm
        n1x = -d1y
        n1y = d1x
        # arrow : a0 - a1 == p1 - a2
        a1x = xMap.transform(p1[0])
        a1y = yMap.transform(p1[1])
        a0x = a1x - ca * d1x + sa * n1x
        a0y = a1y - ca * d1y + sa * n1y
        a2x = a1x - ca * d1x - sa * n1x
        a2y = a1y - ca * d1y - sa * n1y

        poly = QPolygonF()
        poly.append(QPointF(a0x, a0y))
        poly.append(QPointF(a1x, a1y))
        poly.append(QPointF(a2x, a2y))
        painter.drawPolygon(poly)
        d0x = xMap.transform(p0[0])
        d0y = yMap.transform(p0[1])
        painter.drawLine(QPointF(d0x, d0y), QPointF(a1x, a1y))

    def get_item_parameters(self, itemparams):
        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)
        self.gridshapeparam.update_param(self)
        itemparams.add("GridShapeParam", self, self.gridshapeparam)

    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"), visible_only=True)
        self.shapeparam.update_shape(self)
        # modification des parametres specifiques de la gridshape
        order0 = self.order
        self.gridshapeparam.update_shape(self)
        self.define_points()
        if order0 is not self.order:
            self.set_nodes()
            self.set_points()
            self.set_connects()

    """
    def set_style(self, section, option):
        PolygonShape.set_style(self, section, option+"/border")
        self.gridshapeparam.read_config(CONF, section, option)
        self.gridshapeparam.update_axes(self) #problematique...
    """


class ReconstructionShape(PointCloudShape, PolygonShape):
    # une classe particuliere de pointcloudshape
    _can_move = False

    def __init__(
        self,
        A=[2.0, 0.0],
        B=[0.0, 2.0],
        master=None,
        order=2,
        axesparam=None,
        shapeparam=None,
        reconstructionshapeparam=None,
    ):
        super(ReconstructionShape, self).__init__(shapeparam=shapeparam)

        self.arrow_angle = 15  # degrees
        self.arrow_size = 10  # pixels
        self.x_pen = self.pen
        self.x_brush = self.brush
        self.y_pen = self.pen
        self.y_brush = self.brush

        if axesparam is None:
            self.axesparam = AxesShapeParam(_("Axes"), icon="gtaxes.png")
        else:
            self.axesparam = axesparam
            self.axesparam.update_param(self)

        if shapeparam is None:
            self.shapeparam = ShapeParam(_("Shape"), icon="rectangle.png")
        else:
            self.shapeparam = shapeparam
            self.shapeparam.update_shape(self)

        if reconstructionshapeparam is None:
            self.reconstructionshapeparam = ReconstructionShapeParam(
                _("ReconstructionShape"), icon="rectangle.png"
            )
        else:
            self.reconstructionshapeparam = reconstructionshapeparam
            self.reconstructionshapeparam.update_shape(self)

        self.A = np.array(
            A, float
        )  # vecteur 1 de la matrice de la reconstruction dans l'espace direct
        self.B = np.array(
            B, float
        )  # vecteur 2 de la matrice de la reconstruction dans l'espace direct
        self.delta = float(A[0] * B[1] - A[1] * B[0])
        self.order = order
        self.set_nodes()
        # print "master=",master
        if master:
            self.set_master(master)

    def __reduce__(self):
        self.shapeparam.update_param(self)
        state = (self.shapeparam, self.A, self.B, self.order, self.z())
        return (ReconstructionShape, (), state)

    def __getstate__(self):
        return self.shapeparam, self.A, self.B, self.order, self.z()

    def __setstate__(self, state):
        param, A, B, order, z = state
        self.A = A
        self.B = B
        self.order = order
        self.set_nodes()
        self.setZ(z)
        self.shapeparam = param
        self.shapeparam.update_shape(self)

    def set_master(self, master):
        self.master = master
        self.O = self.master.O
        self.set_points()
        self.set_connects()
        self.define_points()

    def set_nodes(self):
        self.nodes = list()
        # les trois premiers noeuds servent a definir le reseau
        self.nodes.append([0, 0])
        self.nodes.append([1, 0])
        self.nodes.append([0, 1])
        for i in range(-self.order, self.order + 1):
            for j in range(-self.order, self.order + 1):
                node = [i, j]
                if node not in self.nodes:
                    self.nodes.append(node)
        self.npts = len(self.nodes)

    def set_points(self):
        # print self.A,self.B
        self.delta = float(self.A[0] * self.B[1] - self.A[1] * self.B[0])
        # relation dans l'espace reciproque
        self.Ar = [self.B[1] / self.delta, -self.B[0] / self.delta]
        self.Br = [-self.A[1] / self.delta, self.A[0] / self.delta]
        # print self.Ar,self.Br
        self.u = [
            self.master.u[0] * self.Ar[0] + self.master.v[0] * self.Ar[1],
            self.master.u[1] * self.Ar[0] + self.master.v[1] * self.Ar[1],
        ]
        self.v = [
            self.master.u[0] * self.Br[0] + self.master.v[0] * self.Br[1],
            self.master.u[1] * self.Br[0] + self.master.v[1] * self.Br[1],
        ]
        self.O = self.master.O
        # print self.u,self.v,self.O
        self.points = np.array(
            list(
                [
                    self.O[0] + self.u[0] * node[0] + self.v[0] * node[1],
                    self.O[1] + self.u[1] * node[0] + self.v[1] * node[1],
                ]
                for node in self.nodes
            ),
            float,
        )

    def set_connects(self):
        self.connects = np.array(
            list(
                [i, j]
                for i in range(self.npts)
                for j in range(self.npts)
                if self.dist(i, j) == 1
            )
        )

    def define_points(self):
        # redefini les positions des points de la grille en fonction des vecteurs ux et uy
        for i in range(self.npts):
            node = self.nodes[i]
            self.points[i, 0] = self.O[0] + self.u[0] * node[0] + self.v[0] * node[1]
            self.points[i, 1] = self.O[1] + self.u[1] * node[0] + self.v[1] * node[1]

    def dist(self, i, j):
        ni = self.nodes[i]
        nj = self.nodes[j]
        return (ni[0] - nj[0]) ** 2 + (ni[1] - nj[1]) ** 2

    def set_space_group_constraints(self, master="u"):
        if master == "u":
            if self.sg in ["pm", "p2mm", "pg", "p2mg", "p2gg"]:
                # les axes doivent etre perpendiculaires
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(v2 / u2)
                self.v[0] = -self.u[1] * ratio
                self.v[1] = self.u[0] * ratio

            elif self.sg in ["cm", "c2mm"]:
                # les axes doivent etre orthonormes
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(u2 / v2)
                self.v[0] = self.v[0] * ratio
                self.v[1] = self.v[1] * ratio

            elif self.sg in ["p4", "p4mm", "p4gm"]:
                # les axes doivent etre orthonormes
                self.v[0] = -self.u[1]
                self.v[1] = self.u[0]

            elif self.sg in ["p3", "p3m1", "p31m", "p6", "p6mm"]:
                # les axes doivent etre orthonormes
                self.v[0] = self.u[0] * cos_60 - self.u[1] * sin_60
                self.v[1] = self.u[0] * sin_60 + self.u[1] * cos_60
        else:
            if self.sg in ["pm", "p2mm", "pg", "p2mg", "p2gg"]:
                # les axes doivent etre perpendiculaires
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(u2 / v2)
                self.u[0] = self.v[1] * ratio
                self.u[1] = -self.v[0] * ratio

            if self.sg in ["cm", "c2mm"]:
                # les axes doivent etre orthonormes
                u2 = self.u[0] * self.u[0] + self.u[1] * self.u[1]
                v2 = self.v[0] * self.v[0] + self.v[1] * self.v[1]
                ratio = np.sqrt(v2 / u2)
                self.u[0] = self.u[0] * ratio
                self.u[1] = self.u[1] * ratio

            if self.sg in ["p4", "p4mm", "p4gm"]:
                # les axes doivent etre orthonormes
                self.u[0] = self.v[1]
                self.u[1] = -self.v[0]

            if self.sg in ["p3", "p3m1", "p31m", "p6", "p6mm"]:
                # les axes doivent etre orthonormes
                self.u[0] = self.v[0] * cos_60 + self.v[1] * sin_60
                self.u[1] = -self.v[0] * sin_60 + self.v[1] * cos_60

    def move_point_to(self, handle, pos, ctrl=None):
        # self.points[handle, :] = pos
        # ici il faut ecrire le comportement de la shape
        return

    def move_shape(self, old_pos, new_pos):
        return

    """
    def set_style(self, section, option):
        self.shapeparam.read_config(CONF, section, option)
        self.shapeparam.update_shape(self)
        self.axesparam.read_config(CONF, section, option)
        self.axesparam.update_axes(self)
    """

    def draw(self, painter, xMap, yMap, canvasRect):
        PointCloudShape.draw(self, painter, xMap, yMap, canvasRect)

        painter.setBrush(painter.pen().color())
        self.draw_arrow(painter, xMap, yMap, self.points[0], self.points[1])
        self.draw_arrow(painter, xMap, yMap, self.points[0], self.points[2])

    def draw_arrow(self, painter, xMap, yMap, p0, p1):
        sz = self.arrow_size
        angle = pi * self.arrow_angle / 180.0
        ca, sa = cos(angle), sin(angle)
        d1x = xMap.transform(p1[0]) - xMap.transform(p0[0])
        d1y = yMap.transform(p1[1]) - yMap.transform(p0[1])
        norm = sqrt(d1x ** 2 + d1y ** 2)
        if abs(norm) < 1e-6:
            return
        d1x *= sz / norm
        d1y *= sz / norm
        n1x = -d1y
        n1y = d1x
        # arrow : a0 - a1 == p1 - a2
        a1x = xMap.transform(p1[0])
        a1y = yMap.transform(p1[1])
        a0x = a1x - ca * d1x + sa * n1x
        a0y = a1y - ca * d1y + sa * n1y
        a2x = a1x - ca * d1x - sa * n1x
        a2y = a1y - ca * d1y - sa * n1y

        poly = QPolygonF()
        poly.append(QPointF(a0x, a0y))
        poly.append(QPointF(a1x, a1y))
        poly.append(QPointF(a2x, a2y))
        painter.drawPolygon(poly)
        d0x = xMap.transform(p0[0])
        d0y = yMap.transform(p0[1])
        painter.drawLine(QPointF(d0x, d0y), QPointF(a1x, a1y))

    def get_item_parameters(self, itemparams):
        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)
        self.reconstructionshapeparam.update_param(self)
        itemparams.add("ReconstructionShapeParam", self, self.reconstructionshapeparam)

    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"), visible_only=True)
        self.shapeparam.update_shape(self)
        # modification des parametres specifiques de la gridshape
        order0 = self.order
        self.reconstructionshapeparam.update_shape(self)
        self.set_points()
        if order0 is not self.order:
            self.set_nodes()
            self.set_points()
            self.set_connects()


class GridShapeTool(RectangularShapeTool):
    # redefinition de EllipseTool de guiqwt
    TITLE = _("Grid")
    ICON = gridpath
    SHAPE_STYLE_KEY = "shape/gridshape"

    def activate(self):
        """Activate tool"""
        for baseplot, start_state in self.start_state.items():
            baseplot.filter.set_state(start_state, None)
        self.action.setChecked(True)
        self.manager.set_active_tool(self)

    def create_shape(self):
        shape = GridShape()
        shape.select()
        self.set_shape_style(shape)
        return shape, 0, 1

    def add_shape_to_plot(self, plot, p0, p1):
        """
        Method called when shape's rectangular area
        has just been drawn on screen.
        Adding the final shape to plot and returning it.
        """
        shape = self.get_final_shape(plot, p0, p1)
        shape.unselect()
        # on donne a la shape le dernier numero  de la liste des EllipseStatShape

        items = list([item for item in plot.get_items() if isinstance(item, GridShape)])
        N = len(items)
        shape.setTitle("Reseau %d" % N)
        shape.set_color(QColor("#ffff00"))  # yellow
        plot.emit(SIG_ITEMS_CHANGED, plot)
        self.handle_final_shape(shape)
        plot.replot()

    def handle_final_shape(self, shape):
        super(GridShapeTool, self).handle_final_shape(shape)


class PeakIdentificationParameters:
    def __init__(self, plot, itemname="", item=None, h=None, k=None, itemindice=0):
        self.plot = plot
        self.itemname = itemname
        self.item = item
        self.itemindice = itemindice
        self.h = h
        self.k = k
        self.griditems = list(
            [item for item in self.plot.get_items() if isinstance(item, GridShape)]
        )
        self.reconstructionitems = list(
            [
                item
                for item in self.plot.get_items()
                if isinstance(item, ReconstructionShape)
            ]
        )

    def guess(self, xpic, ypic):
        # fonction qui va donner pour chacun des elements de griditems et reconstructionitems
        self.guesslist = list()
        self.distlist = list()  # liste des distances a la tache
        for item in self.griditems:
            x = xpic - item.O[0]
            y = ypic - item.O[1]
            delta = item.u[0] * item.v[1] - item.u[1] * item.v[0]
            h = (x * item.v[1] - y * item.v[0]) / delta
            k = (y * item.u[0] - x * item.u[1]) / delta
            self.guesslist.append([item.title().text(), h, k, item])

            xth = item.u[0] * round(h) + item.v[0] * round(k)
            yth = item.u[1] * round(h) + item.v[1] * round(k)
            dist = np.sqrt((xth - x) ** 2 + (yth - y) ** 2)
            self.distlist.append(dist)

        for item in self.reconstructionitems:
            x = xpic - item.O[0]
            y = ypic - item.O[1]
            delta = item.u[0] * item.v[1] - item.u[1] * item.v[0]
            h = (x * item.v[1] - y * item.v[0]) / delta
            k = (y * item.u[0] - x * item.u[1]) / delta
            self.guesslist.append([item.title().text(), h, k, item])

            xth = item.u[0] * round(h) + item.v[0] * round(k)
            yth = item.u[1] * round(h) + item.v[1] * round(k)
            dist = np.sqrt((xth - x) ** 2 + (yth - y) ** 2)
            self.distlist.append(dist)
        if len(self.distlist) > 0:
            self.itemindice = np.argmin(self.distlist)
            self.item = self.guesslist[self.itemindice][3]
            self.itemname = self.guesslist[self.itemindice][0]
            self.h = round(self.guesslist[self.itemindice][1])
            self.k = round(self.guesslist[self.itemindice][2])


class PeakIdentificationWindow(QDialog):
    # definit une fenetre pour rentrer la position de la tache reperee
    def __init__(self, pref):
        QDialog.__init__(self)
        self.pref = pref
        self.itemtexts = list([item[0] for item in self.pref.guesslist])
        self.Gridentry = QComboBox(self)
        self.Gridentry.setGeometry(QRect(5, 5, 200, 25))
        self.Gridentry.insertItems(0, self.itemtexts)

        self.hlabel = QLabel(self)
        self.hlabel.setGeometry(QRect(5, 35, 55, 25))

        self.klabel = QLabel(self)
        self.klabel.setGeometry(QRect(65, 35, 55, 25))

        self.hentry = QLineEdit(self)
        self.hentry.setGeometry(QRect(5, 65, 55, 25))

        self.kentry = QLineEdit(self)
        self.kentry.setGeometry(QRect(65, 65, 55, 25))

        self.changeitem(self.pref.itemindice)

        self.OK = QPushButton(self)
        self.OK.setGeometry(QRect(5, 185, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setGeometry(QRect(100, 185, 90, 25))
        self.Cancel.setText("Cancel")

        QObject.connect(self.Cancel, SIGNAL(_fromUtf8("clicked()")), self.closewin)
        QObject.connect(self.OK, SIGNAL(_fromUtf8("clicked()")), self.appl)
        QObject.connect(
            self.Gridentry,
            SIGNAL(_fromUtf8("currentIndexChanged (int)")),
            self.changeitem,
        )

    def changeitem(self, i):
        # quand on change x, on change les valeurs min et max de x par defaut
        self.Gridentry.setCurrentIndex(i)

        print("change")
        self.hlabel.setText("%f" % self.pref.guesslist[i][1])
        self.klabel.setText("%f" % self.pref.guesslist[i][2])
        h = int(round(self.pref.guesslist[i][1]))
        k = int(round(self.pref.guesslist[i][2]))
        self.hentry.setText("%d" % h)
        self.kentry.setText("%d" % k)

    def appl(self):
        h = self.hentry.text()
        k = self.kentry.text()

        try:
            itemindex = self.Gridentry.currentIndex()
            self.pref.itemname = self.pref.guesslist[itemindex][0]
            self.pref.item = self.pref.guesslist[itemindex][3]
            self.close()
            self.setResult(1)
            self.pref.h = int(h)
            self.pref.k = int(k)
        except Exception:
            QMessageBox.about(self, "Error", "Input not valid")

    def closewin(self):
        self.setResult(0)
        self.close()


class RectanglePeakShape(RectangleShape):
    def __init__(self, x1=0, y1=0, x2=0, y2=0, shapeparam=None):
        super(RectanglePeakShape, self).__init__(x1, y1, x2, y2, shapeparam=shapeparam)
        # self.hasmask=False

    """
    def itemChanged(self):
        super(EllipseStatShape,self).itemChanged()
        print "item Changed!"
    """


class RectanglePeakTool(RectangularShapeTool):
    TITLE = None
    ICON = peakfindpath

    def __init__(
        self,
        manager,
        setup_shape_cb=None,
        handle_final_shape_cb=None,
        shape_style=None,
        toolbar_id=DefaultToolbarID,
        title="find a peak",
        icon=ICON,
        tip=None,
        switch_to_default_tool=None,
    ):
        super(RectangularShapeTool, self).__init__(
            manager,
            self.add_shape_to_plot,
            shape_style,
            toolbar_id=toolbar_id,
            title=title,
            icon=icon,
            tip=tip,
            switch_to_default_tool=switch_to_default_tool,
        )
        self.setup_shape_cb = setup_shape_cb
        self.handle_final_shape_cb = handle_final_shape_cb

    def add_shape_to_plot(self, plot, p0, p1):
        """
        Method called when shape's rectangular area
        has just been drawn on screen.
        Adding the final shape to plot and returning it.
        """
        shape = self.get_final_shape(plot, p0, p1)
        plotpoints = shape.get_points()
        win = plot.window()
        data = (
            win.image.data
        )  # on recupere les donnes associees a l'image dans l'application maitresse
        # conversion des unites du plot en pixel de l'image
        points = np.empty(plotpoints.shape)
        for i in range(4):
            points[i, 0], points[i, 1] = win.image.get_pixel_coordinates(
                plotpoints[i, 0], plotpoints[i, 1]
            )

        # print points
        x1 = min(points[:, 0])
        x2 = max(points[:, 0])
        y1 = min(points[:, 1])
        y2 = max(points[:, 1])
        # on recupere les datas de l'image en cours de traitement

        dim = data.shape
        ix1 = max(0, round(x1))
        ix2 = min(dim[1] - 1, round(x2))
        if ix2 < ix1:
            ix2, ix1 = ix1, ix2
        iy1 = max(0, round(y1))
        iy2 = min(dim[0] - 1, round(y2))
        if iy2 < iy1:
            iy2, iy1 = iy1, iy2
        print(iy1, iy2, ix1, ix2)
        data2 = data[iy1:iy2, ix1:ix2]

        sx = ix2 - ix1
        sy = iy2 - iy1
        # Create x and y indices
        x = np.linspace(ix1, ix2 - 1, sx)
        y = np.linspace(iy1, iy2 - 1, sy)

        self.handle_final_shape(shape)

        try:

            # dans un premier temps on regarde les sommes 1D
            sumx = np.sum(data2, 0)
            fond = np.min(sumx)
            pic = np.max(sumx) - fond
            x0 = np.argmax(sumx) + ix1
            sigma = min(1.0, sx / 10.0)
            initial_guess = (pic, x0, sigma, fond)

            popt, pcov = opt.curve_fit(Gaussian, x, sumx, p0=initial_guess)
            picx = popt[0] / sy
            x0 = popt[1]
            sigmax = popt[2]
            fondx = popt[3] / sy

            sumy = np.sum(data2, 1)
            fond = np.min(sumy)
            pic = np.max(sumy) - fond
            y0 = np.argmax(sumy) + iy1
            sigma = min(1.0, sy / 10.0)
            initial_guess = (pic, y0, sigma, fond)
            popt, pcov = opt.curve_fit(Gaussian, y, sumy, p0=initial_guess)

            picy = popt[0] / sx
            y0 = popt[1]
            sigmay = popt[2]
            fondy = popt[3] / sx

            pic = (picx + picy) / 2.0
            sigma = (sigmax + sigmay) / 2.0
            fond = (fondx + fondy) / 2.0

            initial_guess = (pic, x0, y0, sigma, fond)
            # print initial_guess
            x, y = np.meshgrid(x, y)
            popt, pcov = opt.curve_fit(
                twoD_iso_Gaussian, (x, y), data2.ravel(), p0=initial_guess
            )

            initial_guess = (popt[0], popt[1], popt[2], popt[3], popt[3], 0.0, fond)
            # print popt
            popt, pcov = opt.curve_fit(
                twoD_Gaussian, (x, y), data2.ravel(), p0=initial_guess
            )
            # print popt

            # conversion en unite de plot
            x0, y0 = win.image.get_plot_coordinates(popt[1], popt[2])
            print("peak at", x0, y0)
            if (x1 < popt[1] < x2) and (y1 < popt[2] < y2):
                params = PeakIdentificationParameters(plot)
                params.guess(x0, y0)

                centre = SpotShape(
                    x0,
                    y0,
                    gridname=params.itemname,
                    grid=params.item,
                    h=params.h,
                    k=params.k,
                )
                centre.set_style("plot", "shape/spot")
                plot.add_item(centre)
                self.handle_final_shape(centre)
                plot.replot()

                centre.setTitle(params.itemname + "(%f,%f)" % (x0, y0))
                if params.h is not None:
                    Ok = PeakIdentificationWindow(params).exec_()
                    if Ok:
                        centre.setTitle(
                            params.itemname + "(%d,%d)" % (params.h, params.k)
                        )
            else:
                QMessageBox.about(plot, "Error", "Spot not found")

        except:
            QMessageBox.about(plot, "Error", "Spot not found")

        plot.del_item(shape)
        plot.replot()

    def setup_shape(self, shape):
        """To be reimplemented"""
        shape.setTitle(self.TITLE)
        if self.setup_shape_cb is not None:
            self.setup_shape_cb(shape)

    def handle_final_shape(self, shape):
        """To be reimplemented"""
        if self.handle_final_shape_cb is not None:
            self.handle_final_shape_cb(shape)


class ReconstructionInitParameters:
    # classe pour les parametres de fit
    def __init__(self, plot, Ax=1, Ay=0, Bx=0, By=1, order=4):
        self.Ax = Ax
        self.Ay = Ay
        self.Bx = Bx
        self.By = By
        self.plot = plot
        self.master = None
        self.sym = False
        self.order = order


class ReconstructionInitWindow(QDialog):
    # definit une fenetre pour rentrer les parametres d'affichage des preferences
    def __init__(self, pref):

        QDialog.__init__(self)

        self.pref = pref
        self.items = list(
            [item for item in pref.plot.get_items() if isinstance(item, GridShape)]
        )
        self.itemtexts = list([item.title().text() for item in self.items])

        self.setWindowTitle("Reconstruction Parameters")
        self.setFixedSize(QSize(330, 210))

        self.lab1 = QLabel(self)
        self.lab1.setGeometry(QRect(5, 35, 55, 25))
        self.lab1.setText("x")

        self.lab2 = QLabel(self)
        self.lab2.setGeometry(QRect(5, 65, 55, 25))
        self.lab2.setText("y")

        self.lab1 = QLabel(self)
        self.lab1.setGeometry(QRect(65, 5, 55, 25))
        self.lab1.setText("A")

        self.lab2 = QLabel(self)
        self.lab2.setGeometry(QRect(125, 5, 55, 25))
        self.lab2.setText("B")

        self.Axentry = QLineEdit(self)
        self.Axentry.setGeometry(QRect(65, 35, 55, 25))
        self.Axentry.setText("%d" % pref.Ax)

        self.Bxentry = QLineEdit(self)
        self.Bxentry.setGeometry(QRect(125, 35, 55, 25))
        self.Bxentry.setText("%d" % pref.Ay)

        self.Ayentry = QLineEdit(self)
        self.Ayentry.setGeometry(QRect(65, 65, 55, 25))
        self.Ayentry.setText("%d" % pref.Bx)

        self.Byentry = QLineEdit(self)
        self.Byentry.setGeometry(QRect(125, 65, 55, 25))
        self.Byentry.setText("%d" % pref.By)

        self.lab5 = QLabel(self)
        self.lab5.setGeometry(QRect(5, 95, 55, 25))
        self.lab5.setText("Master")

        self.Masterentry = QComboBox(self)
        self.Masterentry.setGeometry(QRect(65, 95, 120, 25))
        self.Masterentry.insertItems(0, self.itemtexts)

        self.Symentry = QCheckBox(self)
        self.Symentry.setGeometry(QRect(5, 125, 180, 25))
        self.Symentry.setText("Add symetrical domains")
        self.Symentry.setChecked(pref.sym)

        self.lab6 = QLabel(self)
        self.lab6.setGeometry(QRect(5, 155, 55, 25))
        self.lab6.setText("Order")

        self.Orderentry = QLineEdit(self)
        self.Orderentry.setGeometry(QRect(65, 155, 55, 25))
        self.Orderentry.setText("%d" % pref.order)

        self.OK = QPushButton(self)
        self.OK.setGeometry(QRect(5, 185, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setGeometry(QRect(100, 185, 90, 25))
        self.Cancel.setText("Cancel")

        QObject.connect(self.Cancel, SIGNAL(_fromUtf8("clicked()")), self.closewin)
        QObject.connect(self.OK, SIGNAL(_fromUtf8("clicked()")), self.appl)

        # self.exec_()

    def appl(self):
        Ax = self.Axentry.text()
        Ay = self.Ayentry.text()
        Bx = self.Bxentry.text()
        By = self.Byentry.text()
        order = self.Orderentry.text()

        try:
            Ax = float(Ax)
            Ay = float(Ay)
            Bx = float(Bx)
            By = float(By)
            order = int(order)
            self.pref.Ax = Ax
            self.pref.Ay = Ay
            self.pref.Bx = Bx
            self.pref.By = By
            self.pref.order = order
            self.pref.sym = self.Symentry.isChecked()
            itemindex = self.Masterentry.currentIndex()
            self.pref.master = self.items[itemindex]
            self.close()
            self.setResult(1)
        except Exception:
            QMessageBox.about(self, "Error", "Input can only be a float")

    def closewin(self):
        self.setResult(0)
        self.close()


def check_different(A1, B1, A2, B2):
    # fonction qui regarde si la matrice définie par (A1,B1) et celle par (A2,B2) représente
    # le même reseau
    delta2 = A2[0] * B2[1] - A2[1] * B2[0]
    nA1 = (A1[0] * B2[1] - A1[1] * B2[0]) / delta2
    pA1 = (A1[1] * A2[0] - A1[0] * A2[1]) / delta2
    nB1 = (B1[0] * B2[1] - B1[1] * B2[0]) / delta2
    pB1 = (B1[1] * A2[0] - B1[0] * A2[1]) / delta2
    """
    print A1,B1
    print A2,B2
    print delta2,A1[0]*B2[1]-A1[0]*B2[1]
    print nA1,pA1,nB1,pB1
    """

    if nA1 % 1 == 0 and pA1 % 1 == 0 and nB1 % 1 == 0 and pB1 % 1 == 0:
        return False
    else:
        return True


class ReconstructionShapeTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(
            self,
            manager,
            _("Reconstruction"),
            icon=reconstructionpath,
            tip=_("Add a recontruction"),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        # ouvre une fenetre pour les settings de reconstruction
        params = ReconstructionInitParameters(plot)
        Ok = ReconstructionInitWindow(params).exec_()
        if Ok:
            Ax = params.Ax
            Ay = params.Ay
            Bx = params.Bx
            By = params.By
            order = params.order
            master = params.master
            sym = params.sym
            self.addreconstruction(plot, Ax, Ay, Bx, By, order, master, sym)

    def addreconstruction(self, plot, Ax, Ay, Bx, By, order, master, sym):

        A1 = [Ax, Ay]
        B1 = [Bx, By]
        shape = ReconstructionShape(master=master, A=A1, B=B1, order=order)
        shape.set_style("plot", "shape/reconstructionshape")
        shape.set_color(QColor("#ff5500"))  # orange
        self.items = list(
            [item for item in plot.get_items() if isinstance(item, ReconstructionShape)]
        )
        n = len(self.items) + 1
        shape.setTitle("Reconstruction %d" % n)

        shape.set_movable("False")
        master.add_slave(shape)
        plot.add_item(shape)
        z1 = master.z()
        z2 = shape.z()
        shape.setZ(z1)
        master.setZ(z2)
        # print plot,shape

        if sym:
            # il faut regarder les reconstructions possibles
            if master.sg in ["pm", "pg", "cm", "pmm", "p2mm", "p2mg", "p2gg", "c2mm"]:
                A2 = [Ax, -Ay]
                B2 = [Bx, -By]
                if check_different(A1, B1, A2, B2):
                    shape = ReconstructionShape(master=master, A=A2, B=B2, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#0000ff"))  # blue
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

            if master.sg in ["p4"]:
                # rotation 90°
                A2 = [Ay, Ax]
                B2 = [By, Bx]
                if check_different(A1, B1, A2, B2):
                    shape = ReconstructionShape(master=master, A=A2, B=B2, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_color(QColor("#0000ff"))  # blue
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

            if master.sg in ["p4mm", "p4gm"]:
                # rotation 90°
                A2 = [Ay, Ax]
                B2 = [By, Bx]
                if check_different(A1, B1, A2, B2):
                    shape = ReconstructionShape(master=master, A=A2, B=B2, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#0000ff"))  # blue
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)
                # miroir par rapport a l'axe x
                A3 = [Ax, -Ay]
                B3 = [Bx, -By]
                if check_different(A1, B1, A3, B3) and check_different(A2, B2, A3, B3):
                    shape = ReconstructionShape(master=master, A=A3, B=B3, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#006400"))  # dark green
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)
                # miroir par rapport a l'axe x+rotation
                A4 = [Ay, -Ax]
                B4 = [By, -Bx]
                if check_different(A1, B1, A4, B4) and check_different(A2, B2, A4, B4):
                    shape = ReconstructionShape(master=master, A=A4, B=B4, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#ff0000"))  # red
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

            if master.sg in ["p3", "p6"]:
                # rotation 120°
                A2 = [-Ay, Ax - Ay]
                B2 = [-By, Bx - By]
                if check_different(A1, B1, A2, B2):
                    shape = ReconstructionShape(master=master, A=A2, B=B2, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#0000ff"))  # blue
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

                    # rotation -120°
                    A3 = [-Ax + Ay, -Ax]
                    B3 = [-Bx + By, -Bx]
                    shape = ReconstructionShape(master=master, A=A3, B=B3, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#006400"))  # dark green
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

            if master.sg in ["p3m1", "p31m", "p6mm"]:
                A2 = [-Ay, Ax - Ay]
                B2 = [-By, Bx - By]
                if check_different(A1, B1, A2, B2):
                    shape = ReconstructionShape(master=master, A=A2, B=B2, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#0000ff"))  # blue
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

                A3 = [-Ax + Ay, -Ax]
                B3 = [-Bx + By, -Bx]
                if check_different(A1, B1, A3, B3):
                    shape = ReconstructionShape(master=master, A=A3, B=B3, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#006400"))  # dark green
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

                A4 = [Ax - Ay, -Ay]
                B4 = [Bx - By, -By]
                if (
                    check_different(A1, B1, A4, B4)
                    and check_different(A2, B2, A4, B4)
                    and check_different(A3, B3, A4, B4)
                ):
                    shape = ReconstructionShape(master=master, A=A4, B=B4, order=order)
                    shape.set_style("plot", "shape/reconstructionshape")
                    shape.set_color(QColor("#ff0000"))  # red
                    n = n + 1
                    shape.setTitle("Reconstruction %d" % n)
                    shape.set_movable("False")
                    master.add_slave(shape)
                    plot.add_item(shape)
                    z1 = master.z()
                    z2 = shape.z()
                    shape.setZ(z1)
                    master.setZ(z2)

                    A5 = [Ay, Ax]
                    B5 = [By, Bx]
                    if check_different(A4, B4, A5, B5):

                        shape = ReconstructionShape(
                            master=master, A=A5, B=B5, order=order
                        )
                        shape.set_style("plot", "shape/reconstructionshape")
                        shape.set_color(QColor("#808080"))  # grey
                        n = n + 1
                        shape.setTitle("Reconstruction %d" % n)
                        shape.set_movable("False")
                        master.add_slave(shape)
                        plot.add_item(shape)
                        z1 = master.z()
                        z2 = shape.z()
                        shape.setZ(z1)
                        master.setZ(z2)

                        A6 = [-Ax, Ay - Ax]
                        B6 = [-Bx, By - Bx]
                        shape = ReconstructionShape(
                            master=master, A=A6, B=B6, order=order
                        )
                        shape.set_style("plot", "shape/reconstructionshape")
                        shape.set_color(QColor("#000000"))  # black
                        n = n + 1
                        shape.setTitle("Reconstruction %d" % n)
                        shape.set_movable("False")
                        master.add_slave(shape)
                        plot.add_item(shape)
                        z1 = master.z()
                        z2 = shape.z()
                        shape.setZ(z1)
                        master.setZ(z2)
        plot.replot()


class DistorsionCorrectionTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(
            self,
            manager,
            _("DistorsionCorrection"),
            icon=distorsionpath,
            tip=_("Distorsion Correction"),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        self.items = list(
            [item for item in plot.get_items() if isinstance(item, SpotShape)]
        )
        xexp = list()
        yexp = list()
        xth = list()
        yth = list()
        # on recupere l'image pour transformer des coordonnees plot en coordonnees image
        win = plot.window()
        for item in self.items:
            x, y = win.image.get_pixel_coordinates(item.x0, item.y0)
            h = item.h
            k = item.k
            u = item.grid.u
            v = item.grid.v
            O = item.grid.O
            xexp.append(x)
            yexp.append(y)
            x1, y1 = win.image.get_pixel_coordinates(
                O[0] + h * u[0] + k * v[0], O[1] + h * u[1] + k * v[1]
            )
            xth.append(x1)
            yth.append(y1)
        p3x, p3y = self.compute_distorsion(xexp, yexp, xth, yth)
        self.correct_distorsion(plot, p3x, p3y)

    def compute_distorsion(self, xexp, yexp, xth, yth):
        # calcule la transformation quadratique pour passer de
        nref = len(xexp)
        xexp = np.array(xexp)
        yexp = np.array(yexp)
        xth = np.array(xth)
        yth = np.array(yth)
        p1x = [0.0, 1.0, 0.0]
        p1y = [0.0, 0.0, 1.0]

        if nref > 2:
            p1x, success = opt.leastsq(
                erreur_lin, p1x, args=(xth, yth, xexp), maxfev=10000
            )
            p1y, success = opt.leastsq(
                erreur_lin, p1y, args=(xth, yth, yexp), maxfev=10000
            )

            p2x = [p1x[0], p1x[1], p1x[2], 0.0, 0.0, 0.0]
            p2y = [p1y[0], p1y[1], p1y[2], 0.0, 0.0, 0.0]

            if nref > 5:  # on ajoute des termes quadratiques

                p2x, success = opt.leastsq(
                    erreur_quad, p2x, args=(xth, yth, xexp), maxfev=10000
                )
                p2y, success = opt.leastsq(
                    erreur_quad, p2y, args=(xth, yth, yexp), maxfev=10000
                )

                p3x = [
                    p2x[0],
                    p2x[1],
                    p2x[2],
                    p2x[3],
                    p2x[4],
                    p2x[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                p3y = [
                    p2y[0],
                    p2y[1],
                    p2y[2],
                    p2y[3],
                    p2y[4],
                    p2y[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

                if nref > 9:  # on ajoute des termes cubiques
                    p3x, success = opt.leastsq(
                        erreur_cub, p3x, args=(xth, yth, xexp), maxfev=10000
                    )
                    p3y, success = opt.leastsq(
                        erreur_cub, p3y, args=(xth, yth, yexp), maxfev=10000
                    )

            else:
                p3x = [
                    p2x[0],
                    p2x[1],
                    p2x[2],
                    p2x[3],
                    p2x[4],
                    p2x[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                p3y = [
                    p2y[0],
                    p2y[1],
                    p2y[2],
                    p2y[3],
                    p2y[4],
                    p2y[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

        else:
            p3x = [p1x[0], p1x[1], p1x[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            p3y = [p1y[0], p1y[1], p1y[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return p3x, p3y

    def correct_distorsion(self, plot, p3x, p3y):
        # corrige la distorsion pour cela on mappe l'image d'arrivee avec des rectangles qui correspondent
        # a des quadrilateres sur l'image non deformee. Ceux-ci sont obtenus par la transformation
        # xth->xexp, yth->yexp
        # dans un premier temps, on regarde la valeur de p3x[1] et p3xy[2] pour savoir sur quelle taille d'image
        # faire la correction
        win = plot.window()
        w, h = win.pilim.size
        meshdata = []

        # dx=(x2-x1)/10.
        # dy=(y2-y1)/10.

        ninter = 10
        nsteps = ninter + 1
        xdest = np.linspace(0, w, nsteps)
        ydest = np.linspace(0, h, nsteps)
        xgrid, ygrid = np.meshgrid(
            xdest, ydest
        )  # ensemble des valeurs de x et y de destination
        xgrid = xgrid.ravel()
        ygrid = ygrid.ravel()
        xstart = corr_cub(p3x, xgrid, ygrid)  # ensemble des valeurs de x et y de depart
        ystart = corr_cub(p3y, xgrid, ygrid)
        fic = open("sortie.txt", "w")
        for i in range(len(xgrid)):
            fic.write("%f %f %f %f" % (xgrid[i], ygrid[i], xstart[i], ystart[i]))
            fic.write("\n")
        fic.close()

        xstart = xstart.reshape(nsteps, nsteps)
        ystart = ystart.reshape(nsteps, nsteps)

        xdest = np.array(np.around(xdest), int)
        ydest = np.array(np.around(ydest), int)
        xstart = np.array(np.around(xstart), int)
        ystart = np.array(np.around(ystart), int)

        meshdata = []
        for j in range(ninter):
            for i in range(ninter):
                xl = xdest[i]
                xr = xdest[i + 1]
                yu = ydest[j]
                yl = ydest[j + 1]
                xul = xstart[j, i]
                xur = xstart[j, i + 1]
                yul = ystart[j, i]
                yur = ystart[j, i + 1]
                xll = xstart[j + 1, i]
                xlr = xstart[j + 1, i + 1]
                yll = ystart[j + 1, i]
                ylr = ystart[j + 1, i + 1]
                meshdata.append(
                    ((xl, yu, xr, yl), (xul, yul, xll, yll, xlr, ylr, xur, yur))
                )
        img = win.pilim.transform((w, h), Image.MESH, meshdata, Image.BICUBIC)
        data = np.array(img).reshape((w, h))

        win2 = win.duplicate()
        win2.image.set_data(data)

        # on effectue la transformation sur les taches identifiees
        spotitems = list([item for item in plot.items if (isinstance(item, SpotShape))])
        spotitems2 = list(
            [item for item in win2.plot.items if (isinstance(item, SpotShape))]
        )

        for i in range(len(spotitems)):
            item = spotitems[i]
            item2 = spotitems2[i]
            # coordonnees pixels
            x0, y0 = win2.image.get_pixel_coordinates(item.x0, item.y0)
            # on recherche le polygone de depart
            jquad = -1
            for j in range(
                len(meshdata)
            ):  # on pourrait faire plus rapide en affinant a partir de la position supposee....
                quad = mplPath(np.array(meshdata[j][1]).reshape(4, 2))
                if quad.contains_point([x0, y0]):
                    jquad = j
                    break
            if jquad == -1:
                # le point n'est pas sur l'image d'arrivee, on le supprime
                win2.plot.del_item(item2)
            else:
                # on resoud le systeme d'equation lineaire pour obtenir les coefficients
                # de la transformation quad->rectangle
                xl, yu, xr, yl = meshdata[j][0]
                xul, yul, xll, yll, xlr, ylr, xur, yur = meshdata[j][1]
                A = [
                    [xul, yul, 1.0, 0.0, 0.0, 0.0, -xul * xl, -yul * xl],
                    [xur, yur, 1.0, 0.0, 0.0, 0.0, -xur * xr, -yur * xr],
                    [xll, yll, 1.0, 0.0, 0.0, 0.0, -xll * xl, -yll * xl],
                    [xlr, ylr, 1.0, 0.0, 0.0, 0.0, -xlr * xr, -ylr * xr],
                    [0.0, 0.0, 0.0, xul, yul, 1.0, -xul * yu, -yul * yu],
                    [0.0, 0.0, 0.0, xll, yll, 1.0, -xll * yl, -yll * yl],
                    [0.0, 0.0, 0.0, xur, yur, 1.0, -xur * yu, -yur * yu],
                    [0.0, 0.0, 0.0, xlr, ylr, 1.0, -xlr * yl, -ylr * yl],
                ]
                B = [xl, xr, xl, xr, yu, yl, yu, yl]

                C = np.linalg.solve(A, B)
                # print B,np.dot(A,C)
                a, b, c, d, e, f, g, h = C
                # transformation quad->rectangle
                x1 = (a * x0 + b * y0 + c) / (g * x0 + h * y0 + 1.0)
                y1 = (d * x0 + e * y0 + f) / (g * x0 + h * y0 + 1.0)
                # coordonnees plot
                x2, y2 = win2.image.get_plot_coordinates(x1, y1)
                item2.set_pos(x2, y2)

        win2.show()


class D2DDialog(ImageDialog):
    # une fenetre qui permet d'afficher les images D2D et gere les datas associees
    def __init__(
        self,
        edit=False,
        toolbar=True,
        options={"show_contrast": True},
        parent=None,
        data=None,
    ):
        ImageDialog.__init__(
            self, edit=edit, toolbar=toolbar, wintitle="D2D window", options=options
        )
        self.setGeometry(QRect(440, 470, 500, 500))
        self.plot = self.get_plot()
        # self.plot.window()=self
        self.add_tool(GridShapeTool)
        self.ReconstructionShapeTool = self.add_tool(ReconstructionShapeTool)
        self.add_tool(RectanglePeakTool)
        self.add_tool(DistorsionCorrectionTool)
        self.data = data
        self.connect(self.plot, SIG_ITEM_REMOVED, self.refreshlist)
        self.show()

    def set_data(
        self,
        data,
        xdata=[None, None],
        ydata=[None, None],
        colormap="bone",
        transformable=False,
    ):
        # data est un tableau 2D qui represente les donnees
        # xdata sont les bornes en x, ydata les bornes en y
        self.data = np.array(data, np.float32)
        # on regarde s'il y a quelque chose de dessine
        listimage = self.plot.get_items(item_type=IColormapImageItemType)
        if len(listimage) > 0:
            self.defaultcolormap = listimage[0].get_color_map_name()
        else:
            self.defaultcolormap = "bone"
        if transformable:
            self.image = make.trimage(self.data, colormap=colormap)
        else:
            self.image = make.image(
                self.data, xdata=xdata, ydata=ydata, colormap=colormap
            )
        self.pilim = Image.frombuffer(
            "F", (self.data.shape[1], self.data.shape[0]), self.data, "raw", "F", 0, 1
        )

    def show_image(self, remove=True):
        if remove:
            # on enleve tout ce qu'il y a
            self.plot.del_all_items()

        self.plot.add_item(self.image)
        self.plot.show_items()

    def refreshlist(self, item):
        # quand on efface un item
        if isinstance(item, GridShape):
            # on supprime referencement aux reconstructions
            if len(item.slaves) > 0:
                for slave in item.slave:
                    slave.master = None
        if isinstance(item, ReconstructionShape):
            if item.master is not None:
                item.master.remove_slave(item)

    def duplicate(self):
        # retourne une copie de la fenetre
        win2 = D2DDialog()
        win2.set_data(
            self.data, xdata=self.image.get_xdata(), ydata=self.image.get_ydata()
        )

        win2.show_image()
        plot = self.plot
        plot2 = win2.plot
        plot2.set_title(plot.get_title())

        allitems = plot.items
        griditems = list([item for item in allitems if (isinstance(item, GridShape))])
        spotitems = list([item for item in allitems if (isinstance(item, SpotShape))])

        # on rajoute les griditems
        for item in griditems:
            item.shapeparam.update_param(
                item
            )  # certains parametres ne sont pas forcement actualises
            item2 = GridShape()
            # item.shapeparam.update_shape(item)
            param, O, u, v, order, sg, slaves, z = item.__getstate__()
            # print param
            slaves2 = list()  # on oublie les slaves pour l'instant
            item2.__setstate__((param, O, u, v, order, sg, slaves2, z))
            plot2.add_item(item2)
            # on rajoute les reconstructionitems avec les nouvelles relations master-slave
            # item2.slaves=list()  #on repart de zero
            for slave in slaves:
                slave.shapeparam.update_param(
                    slave
                )  # certains parametres ne sont pas forcement actualises
                slave2 = ReconstructionShape()
                params = slave.__getstate__()
                slave2.__setstate__(params)
                item2.add_slave(slave2)
                slave2.set_master(item2)  # on ajoute la relation master-slave
                plot2.add_item(slave2)

        # on rajoute les spotitems
        for item in spotitems:
            item.shapeparam.update_param(
                item
            )  # certains parametres ne sont pas forcement actualises
            item2 = SpotShape()
            params = item.__getstate__()
            item2.__setstate__(params)
            plot2.add_item(item2)
            item2.set_grid_from_gridname(item2.gridname)
        return win2


def test():
    """Test"""
    # -- Create QApplication
    # --
    filename = osp.join(osp.dirname(__file__), "test.jpg")

    FNAME = "test.pickle"
    win = D2DDialog()

    if access(FNAME, R_OK):
        print("Restoring data...")
        iofile = open(FNAME, "rb")
        image = pickle.load(iofile)

        x0, x1 = image.get_xdata()
        y0, y1 = image.get_ydata()
        data = image.get_data(x0, y0, x1, y1)[2]
        image.data = np.array(data, float)

        print("taille de l'image", image.data.shape)

        win.set_data(data, xdata=[x0, x1], ydata=[y0, y1])
        win.show_image(remove=True)

        ngriditem = pickle.load(iofile)
        plot = win.get_plot()
        # print ngriditem
        for i in range(ngriditem):
            griditem = pickle.load(iofile)
            plot.add_item(griditem)
            for slave in griditem.slaves:
                # slave deja charge
                #               slave.set_master(griditem)
                plot.add_item(slave)

        nspotitem = pickle.load(iofile)
        # print nspotitem
        for i in range(nspotitem):
            spotitem = pickle.load(iofile)
            plot.add_item(spotitem)
            spotitem.set_grid_from_gridname(spotitem.gridname)
            if spotitem.grid is None:
                message = "Spot %d not indexed" % (i)
                QMessageBox.about(plot, "Error", message)
            # print i,spotitem.gridname,spotitem.grid

        iofile.close()
        print("OK")
    else:

        image = make.image(
            filename=filename,
            xdata=[-200.5, 200.5],
            ydata=[-200.5, 200.5],
            colormap="bone",
        )
        x0, x1 = image.get_xdata()
        y0, y1 = image.get_ydata()
        data = image.get_data(x0, y0, x1, y1)[2]
        image.data = np.array(data, float)

        print("taille de l'image", image.data.shape)
        win.set_data(data, xdata=[x0, x1], ydata=[y0, y1])
        win.show_image(remove=True)

        plot = win.get_plot()

        shape = GridShape(O=[00, 000], u=[185, 0], v=[92.5, 160], sg="p3m1", order=2)
        shape.set_style("plot", "shape/gridshape")
        shape.setTitle("Reseau 1")
        plot.add_item(shape)

        win.ReconstructionShapeTool.addreconstruction(
            plot, 4.0, 1.0, -1.0, 3.0, 4, shape, True
        )

    win.exec_()

    iofile = open(FNAME, "wb")
    pickle.dump(image, iofile)
    allitems = plot.get_items()
    # print allitems
    griditems = list([item for item in allitems if (isinstance(item, GridShape))])
    # reconstructionitems=list([item for item in allitems if (isinstance(item, ReconstructionShape))])
    spotitems = list([item for item in allitems if (isinstance(item, SpotShape))])
    pickle.dump(len(griditems), iofile)
    for item in griditems:
        # print item
        pickle.dump(item, iofile)
    pickle.dump(len(spotitems), iofile)
    for item in spotitems:
        # print item
        pickle.dump(item, iofile)


if __name__ == "__main__":
    import guidata

    _app = guidata.qapplication()
    # test()
    win = D2DDialog()
    win.exec_()
