# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Tue Oct 08 10:39:12 2013

@author: Prevot
"""


import numpy as np
from scipy.interpolate import griddata

from guidata.qt import QtCore, QtGui

import os
import tables


# from guidata.qt.QtGui import QFont
# from guidata.qt.QtCore import Qt
from guiqwt.plot import CurveDialog
from guiqwt.builder import make
from guiqwt.signals import (
    SIG_ACTIVE_ITEM_CHANGED,
    SIG_START_TRACKING,
    SIG_MOVE,
    SIG_STOP_NOT_MOVING,
    SIG_STOP_MOVING,
)
from guiqwt.events import QtDragHandler, setup_standard_tool_filter
from guiqwt.tools import InteractiveTool, DefaultToolbarID, CommandTool
from guiqwt.interfaces import ICurveItemType
from guiqwt.config import _
from guiqwt.shapes import Marker
from guiqwt.interfaces import ITrackableItemType

# from guidata.qthelpers import get_std_icon
from guidata.configtools import get_icon
from reciprocal2D import D2Dview

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

# import guidata
# _app = guidata.qapplication()
degtorad = np.pi / 180.0

"""
paramsdict = {a[0]:a[1] for a in params}
paramslist = list(a[0] for a in params)
"""


class MeshParameters:
    # classe pour les parametres de construction d'une map 3D
    def __init__(
        self,
        Nx=1,
        Ny=1,
        xmin=0.0,
        ymin=0.0,
        xmax=0.0,
        ymax=0.0,
        ix=0,
        iy=0,
        iz=0,
        ilog=0,
        io=0,
        angle=0.0,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.ilog = ilog  # intensite en echelle log
        self.io = io  # utiliser la matrice d'orientation
        self.angle = angle  # rajouter cet angle
        self.bc = False  # correction de fond sur chaque scan
        self.b1 = 0.0  # percentile min pour estimation du background
        self.b2 = 10.0  # percentile max pour estimation du background


class DataFilter:
    # classe pour les parametres de filtre
    def __init__(self, value=0.0, role="none"):
        self.ifil = -1
        self.value = value
        self.role = role
        self.roles = list(("none", "high-pass", "low-pass", "cut"))
        self.irole = self.roles.index(self.role)


class PrefParameters:
    # classe pour les parametres de preference
    # -> affichage du nom des moteurs
    # -> mise a jour automatique du fit
    def __init__(self):
        self.namedisplays = list(
            (
                "my suggestion",
                "short name",
                "short and family name",
                "full name",
                "nxs name",
            )
        )
        self.inamedisplay = 0
        self.autoupdate = True
        self.followscans = True


class DataSet:
    # definit une classe pour rentrer les donnees associees a un fichier nxs
    def __init__(self, longname, shortname, datafilter=None, pref=None):
        if pref is None:
            pref = PrefParameters()
        self.shortname = shortname
        self.longname = longname

        try:
            fichier = tables.openFile(longname)
            self.nodedatasizes = list()  # liste des longueurs des  tableaux de donnees
            # print shortname,fichier.listNodes('/')[0].start_time[0],fichier.listNodes('/')[0].end_time[0]
            # for leaf in fichier.__iter__():
            #
            #    if leaf._v_name=="temperature":
            #        print leaf._v_name,leaf.read()[0]

            for leaf in fichier.listNodes("/")[0].scan_data:
                self.nodedatasizes.append(leaf.shape[0])

            self.npts = max(self.nodedatasizes)

            # on ne selectionne que les noeuds qui ont la meme longueur, il peut y avoir des donnes avec des tableaux de taille plus petite par exemple 1, on laisse tomber
            self.nodenames = list()  # nom des noeuds (ex data_01)
            self.nodelongnames = list()  # nom comprehensible des noeuds (complet)
            self.nodenicknames = list()  # diminutif pour affichage
            self.data = np.empty(0)  # creation d'un tableau vide de depart

            # print pref.inamedisplay
            for leaf in fichier.listNodes("/")[0].scan_data:
                if leaf.shape[0] == self.npts:
                    self.nodenames.append(leaf.name)
                    try:
                        nodelongname = leaf.attrs.long_name
                    except:
                        nodelongname = ""
                    # nodelongname=leaf.attrs.long_name[leaf.attrs.long_name.rfind('/')+1:]
                    if len(nodelongname) == 0:
                        nodelongname = (
                            leaf.name
                        )  # si pas de nom long on garde le nom nxs
                    self.nodelongnames.append(nodelongname)
                    self.data = np.concatenate(
                        (self.data, leaf.read()[1:])
                    )  # on ajoute les donnees au tableau np, on enleve le premier point

                    if pref.inamedisplay <= 1:
                        nodenickname = nodelongname.split("/")[
                            -1
                        ]  # on prend le dernier
                        self.nodenicknames.append(nodenickname)

                    elif pref.inamedisplay == 2:
                        try:
                            namesplit = nodelongname.split("/")
                            nodenickname = (
                                namesplit[-2] + "/" + namesplit[-1]
                            )  # on prend les deux derniers si possible
                            self.nodenicknames.append(nodenickname)
                        except:
                            self.nodenicknames.append(nodelongname)

                    elif pref.inamedisplay == 3:
                        self.nodenicknames.append(
                            nodelongname
                        )  # on prend le nom long complet

                    elif pref.inamedisplay == 4:
                        self.nodenicknames.append(leaf.name)  # on prend le nom nxs

        except ValueError:
            print("probleme le fichier ", longname, "est corrompu")
            self.npts = 0
            self.nmotors = 0
            self.mins = np.empty(0)
            self.maxs = np.empty(0)
            self.data = np.empty(0)
            fichier.close()
            return
        except tables.exceptions.NoSuchNodeError:
            print("probleme le fichier ", longname, "est corrompu")
            self.npts = 0
            self.nmotors = 0
            self.mins = np.empty(0)
            self.maxs = np.empty(0)
            self.data = np.empty(0)
            fichier.close()
            return
        else:
            fichier.close()

        self.npts = self.npts - 1  # on a enleve le premier point
        self.nmotors = len(self.nodenames)  # nombre de colonnes retenues

        # si les preferences de display sont "my suggestion" on va regarder si un nom figure plusieurs fois
        # dans ce cas, on choisit de prendre le nom plus long

        if pref.inamedisplay == 0:
            for i in range(self.nmotors - 1):  # pas la peine de faire le dernier point!

                nickname = self.nodenicknames[i]

                if nickname in self.nodenicknames[i + 1 :]:  # alors item en double
                    nodelongname = self.nodelongnames[i]
                    namesplit = nodelongname.split("/")
                    try:
                        nodenickname = (
                            namesplit[-2] + "/" + namesplit[-1]
                        )  # on prend les deux derniers
                    except:
                        nodenickname = nodelongname  # on prend les deux derniers

                    self.nodenicknames[i] = nodenickname

                    j = i
                    try:
                        while 1:
                            j = self.nodenicknames.index(j + 1)
                            self.nodenicknames[
                                j
                            ] = nodenickname  # attention, on ne garantit pas que nodenickname!=nickname

                    except ValueError:
                        pass

        self.data = self.data.reshape((self.nmotors, self.npts))
        # print self.data[0]
        test = np.any(
            self.data != 0, axis=0
        )  # si une valeur non nulle, la condition est verifiee
        # print test
        self.data = np.compress(test, self.data, axis=1)

        if datafilter is not None:
            # on filtre les valeurs en regardant la condition sur le filtre
            # print datafilter.role,datafilter.ifil
            if datafilter.role != "none" and datafilter.ifil > -1:
                # print "irole=",datafilter.irole
                if datafilter.irole == 1:
                    # print "on laisse si col",datafilter.ifil,">",datafilter.value
                    # print self.data[datafilter.ifil]>datafilter.value
                    self.data = np.compress(
                        self.data[datafilter.ifil] > datafilter.value, self.data, axis=1
                    )
                elif datafilter.irole == 2:
                    # print "on laisse si col",datafilter.ifil,"<",datafilter.value
                    self.data = np.compress(
                        self.data[datafilter.ifil] < datafilter.value, self.data, axis=1
                    )
                elif datafilter.irole == 3:
                    # print "on laisse si col",datafilter.ifil,"!=",datafilter.value
                    self.data = np.compress(
                        self.data[datafilter.ifil] != datafilter.value,
                        self.data,
                        axis=1,
                    )

        self.npts = self.data.shape[1]  # nombre de points non totalement nuls

        if self.npts == 0:  # filtrage tel qu'il n'y a plus de points
            QtGui.QMessageBox.about(None, "Error", "All points are filtered")

            self.mins = np.zeros(self.nmotors)
            self.maxs = np.ones(self.nmotors)
        else:
            self.mins = np.amin(self.data, axis=1)  # bornes pour chaque parametre
            self.maxs = np.amax(self.data, axis=1)


class Set3DparametersWindow(QtGui.QDialog):
    # definit une fenetre pour rentrer les parametres de dialogue de construction de map 3D
    def __init__(self, dataset, params, NX=1, NY=1, mins=None, maxs=None):
        self.params = params
        xdef = params.ix
        ydef = params.iy
        zdef = params.iz
        ilog = params.ilog
        io = params.io
        angle = params.angle
        bc = params.bc
        b1 = params.b1
        b2 = params.b2
        self.dataset = dataset
        self.NY = NY  # a priori le nombre de scans selectionne
        super(
            Set3DparametersWindow, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.setWindowTitle("Parameters for 3D mesh computation")
        self.setFixedSize(QtCore.QSize(330, 230))
        self.mins = mins
        self.maxs = maxs

        """
        if mins==None:
            self.mins=np.zeros(dataset.nmotors)
        else:
            self.mins=mins

        if maxs==None:
            self.maxs=np.zeros(dataset.nmotors)
        else:
            self.maxs=maxs
        """
        # print self.mins,self.maxs
        self.lab1 = QtGui.QLabel(self)
        self.lab1.setGeometry(QtCore.QRect(35, 5, 100, 25))
        self.lab1.setText("component")

        self.lab2 = QtGui.QLabel(self)
        self.lab2.setGeometry(QtCore.QRect(140, 5, 55, 25))
        self.lab2.setText("bins")

        self.lab3 = QtGui.QLabel(self)
        self.lab3.setGeometry(QtCore.QRect(200, 5, 55, 25))
        self.lab3.setText("min")

        self.lab4 = QtGui.QLabel(self)
        self.lab4.setGeometry(QtCore.QRect(260, 5, 55, 25))
        self.lab4.setText("max")

        self.labx = QtGui.QLabel(self)
        self.labx.setGeometry(QtCore.QRect(5, 35, 25, 25))
        self.labx.setText("x")

        self.laby = QtGui.QLabel(self)
        self.laby.setGeometry(QtCore.QRect(5, 65, 25, 25))
        self.laby.setText("y")

        self.labz = QtGui.QLabel(self)
        self.labz.setGeometry(QtCore.QRect(5, 95, 25, 25))
        self.labz.setText("z")

        self.paramx = QtGui.QComboBox(self)
        self.paramx.setGeometry(QtCore.QRect(35, 35, 100, 25))
        self.paramx.addItems(dataset.nodenicknames)
        self.paramx.setCurrentIndex(xdef)

        self.paramy = QtGui.QComboBox(self)
        self.paramy.setGeometry(QtCore.QRect(35, 65, 100, 25))
        self.paramy.addItems(dataset.nodenicknames)
        self.paramy.addItem("None")
        self.paramy.setCurrentIndex(ydef)

        self.paramz = QtGui.QComboBox(self)
        self.paramz.setGeometry(QtCore.QRect(35, 95, 100, 25))
        self.paramz.addItems(dataset.nodenicknames)
        self.paramz.setCurrentIndex(zdef)

        self.binx = QtGui.QLineEdit(self)
        self.binx.setGeometry(QtCore.QRect(140, 35, 45, 25))
        self.binx.setText("%d" % (NX))

        self.biny = QtGui.QLineEdit(self)
        self.biny.setGeometry(QtCore.QRect(140, 65, 45, 25))
        self.biny.setText("%d" % (NY))

        self.zscale = QtGui.QComboBox(self)
        self.zscale.setGeometry(QtCore.QRect(140, 95, 100, 25))
        self.zscale.addItems(("z linear", "z log"))
        self.zscale.setCurrentIndex(ilog)

        self.xmin = QtGui.QLineEdit(self)
        self.xmin.setGeometry(QtCore.QRect(190, 35, 65, 25))

        self.ymin = QtGui.QLineEdit(self)
        self.ymin.setGeometry(QtCore.QRect(190, 65, 65, 25))

        self.xmax = QtGui.QLineEdit(self)
        self.xmax.setGeometry(QtCore.QRect(260, 35, 65, 25))

        self.ymax = QtGui.QLineEdit(self)
        self.ymax.setGeometry(QtCore.QRect(260, 65, 65, 25))

        self.changex(xdef)
        self.changey(ydef)

        self.io = QtGui.QRadioButton(self)
        self.io.setGeometry(QtCore.QRect(35, 125, 155, 25))
        self.io.setText("use orientation matrix")
        self.io.setChecked(io)

        self.angle = QtGui.QLineEdit(self)
        self.angle.setGeometry(QtCore.QRect(200, 125, 65, 25))
        self.angle.setText("%f" % (angle))

        self.bc = QtGui.QRadioButton(self)
        self.bc.setGeometry(QtCore.QRect(35, 155, 155, 25))
        self.bc.setText("correct background")
        self.bc.setChecked(bc)

        self.b1 = QtGui.QLineEdit(self)
        self.b1.setGeometry(QtCore.QRect(200, 155, 45, 25))
        self.b1.setText("%d" % (b1))

        self.b2 = QtGui.QLineEdit(self)
        self.b2.setGeometry(QtCore.QRect(250, 155, 45, 25))
        self.b2.setText("%d" % (b2))

        self.OK = QtGui.QPushButton(self)
        self.OK.setGeometry(QtCore.QRect(5, 185, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QtGui.QPushButton(self)
        self.Cancel.setGeometry(QtCore.QRect(100, 185, 90, 25))
        self.Cancel.setText("Cancel")

        self.bgroup = QtGui.QButtonGroup(self)
        self.bgroup.setExclusive(False)
        self.bgroup.addButton(self.io)
        self.bgroup.addButton(self.bc)

        QtCore.QObject.connect(
            self.Cancel, QtCore.SIGNAL(_fromUtf8("clicked()")), self.closewin
        )
        QtCore.QObject.connect(
            self.OK, QtCore.SIGNAL(_fromUtf8("clicked()")), self.appl
        )
        QtCore.QObject.connect(
            self.paramx,
            QtCore.SIGNAL(_fromUtf8("currentIndexChanged (int)")),
            self.changex,
        )
        QtCore.QObject.connect(
            self.paramy,
            QtCore.SIGNAL(_fromUtf8("currentIndexChanged (int)")),
            self.changey,
        )
        self.exec_()

    def changex(self, i):
        # quand on change x, on change les valeurs min et max de x par defaut
        self.xmin.setText("%f" % (self.mins[i]))
        self.xmax.setText("%f" % (self.maxs[i]))

    def changey(self, i):
        # quand on change x, on change les valeurs min et max de x par defaut
        # sauf quand on est sur le dernier de la liste
        if i < len(self.mins):
            self.ymin.setText("%f" % (self.mins[i]))
            self.ymax.setText("%f" % (self.maxs[i]))
        else:
            self.ymin.setText("%d" % (1))
            self.ymax.setText("%d" % (self.NY))

    def appl(self):

        if self.parent:

            try:
                self.params.NX = int(self.binx.text())
                self.params.NY = int(self.biny.text())
                self.params.xmin = float(self.xmin.text())
                self.params.xmax = float(self.xmax.text())
                self.params.ymin = float(self.ymin.text())
                self.params.ymax = float(self.ymax.text())
                self.params.angle = float(self.angle.text())
                self.params.b1 = float(self.b1.text())
                self.params.b2 = float(self.b2.text())

            except Exception:
                QtGui.QMessageBox.about(self, "Error", "Input can only be a number")
                return

            ix = self.paramx.currentIndex()
            iy = self.paramy.currentIndex()
            iz = self.paramz.currentIndex()
            if ix == iy or iy == iz or ix == iy:
                QtGui.QMessageBox.about(self, "Error", "Select 3 different parameters")
                return

            self.params.ix = ix
            self.params.iy = iy
            self.params.iz = iz
            self.params.ilog = self.zscale.currentIndex()
            self.params.io = self.io.isChecked()
            self.params.bc = self.bc.isChecked()
        self.close()

    def closewin(self):
        self.close()


class SetFilterWindow(QtGui.QDialog):
    # definit une fenetre pour rentrer les parametres de dialogue de construction de map 3D
    def __init__(self, datafilter):

        super(
            SetFilterWindow, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.datafilter = datafilter
        self.setWindowTitle("Filter Policy")
        self.setFixedSize(QtCore.QSize(200, 100))

        # print self.mins,self.maxs
        self.lab1 = QtGui.QLabel(self)
        self.lab1.setGeometry(QtCore.QRect(5, 5, 100, 25))
        self.lab1.setText("Threshold value")

        self.lab2 = QtGui.QLabel(self)
        self.lab2.setGeometry(QtCore.QRect(5, 35, 100, 25))
        self.lab2.setText("Filter policy")

        self.value = QtGui.QLineEdit(self)
        self.value.setGeometry(QtCore.QRect(105, 5, 90, 25))
        self.value.setText("%f" % (datafilter.value))

        self.role = QtGui.QComboBox(self)
        self.role.setGeometry(QtCore.QRect(105, 35, 90, 25))
        self.role.addItems(datafilter.roles)
        self.role.setCurrentIndex(datafilter.irole)

        self.OK = QtGui.QPushButton(self)
        self.OK.setGeometry(QtCore.QRect(5, 65, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QtGui.QPushButton(self)
        self.Cancel.setGeometry(QtCore.QRect(100, 65, 90, 25))
        self.Cancel.setText("Cancel")

        QtCore.QObject.connect(
            self.Cancel, QtCore.SIGNAL(_fromUtf8("clicked()")), self.closewin
        )
        QtCore.QObject.connect(
            self.OK, QtCore.SIGNAL(_fromUtf8("clicked()")), self.appl
        )
        self.exec_()

    def appl(self):

        self.datafilter.irole = self.role.currentIndex()
        self.datafilter.role = self.datafilter.roles[self.datafilter.irole]

        try:
            self.datafilter.value = float(self.value.text())

        except Exception:
            QtGui.QMessageBox.about(self, "Error", "Input can only be a number")
            return

        self.close()

    def closewin(self):
        self.close()


class SetPrefWindow(QtGui.QDialog):
    # definit une fenetre pour rentrer les parametres d'affichage des preferences
    def __init__(self, pref):

        QtGui.QDialog.__init__(self)

        self.pref = pref
        self.setWindowTitle("Preferences")
        self.setFixedSize(QtCore.QSize(250, 160))

        self.lab1 = QtGui.QLabel(self)
        self.lab1.setGeometry(QtCore.QRect(5, 5, 90, 25))
        self.lab1.setText("Name display")

        self.nameDisplay = QtGui.QComboBox(self)
        self.nameDisplay.setGeometry(QtCore.QRect(95, 5, 150, 25))
        self.nameDisplay.addItems(pref.namedisplays)
        self.nameDisplay.setCurrentIndex(pref.inamedisplay)

        self.autoupdate = QtGui.QCheckBox(self)
        self.autoupdate.setGeometry(QtCore.QRect(5, 35, 150, 25))
        self.autoupdate.setText("auto-update fit")
        self.autoupdate.setChecked(pref.autoupdate)

        self.followscans = QtGui.QCheckBox(self)
        self.followscans.setGeometry(QtCore.QRect(5, 65, 150, 25))
        self.followscans.setText("follow fitted scans")
        self.followscans.setChecked(pref.followscans)

        self.OK = QtGui.QPushButton(self)
        self.OK.setGeometry(QtCore.QRect(5, 125, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QtGui.QPushButton(self)
        self.Cancel.setGeometry(QtCore.QRect(100, 125, 90, 25))
        self.Cancel.setText("Cancel")

        QtCore.QObject.connect(
            self.Cancel, QtCore.SIGNAL(_fromUtf8("clicked()")), self.closewin
        )
        QtCore.QObject.connect(
            self.OK, QtCore.SIGNAL(_fromUtf8("clicked()")), self.appl
        )
        self.exec_()

    def appl(self):

        self.pref.inamedisplay = self.nameDisplay.currentIndex()
        self.pref.autoupdate = self.autoupdate.isChecked()
        self.pref.followscans = self.followscans.isChecked()

        self.close()

    def closewin(self):
        self.close()


class SelectMultiPointTool(InteractiveTool):
    TITLE = _("Point selection")
    ICON = "point_selection.png"
    MARKER_STYLE_SECT = "plot"
    MARKER_STYLE_KEY = "marker/curve"
    CURSOR = QtCore.Qt.PointingHandCursor

    def __init__(
        self,
        manager,
        mode="reuse",
        on_active_item=False,
        title=None,
        icon=None,
        tip=None,
        end_callback=None,
        toolbar_id=DefaultToolbarID,
        marker_style=None,
        switch_to_default_tool=None,
    ):
        super(SelectMultiPointTool, self).__init__(
            manager, toolbar_id, title=title, icon=icon, tip=tip
        )
        #                              switch_to_default_tool=switch_to_default_tool)
        assert mode in ("reuse", "create")
        self.mode = mode
        self.end_callback = end_callback
        self.marker = None
        self.last_pos = None
        self.on_active_item = on_active_item
        if marker_style is not None:
            self.marker_style_sect = marker_style[0]
            self.marker_style_key = marker_style[1]
        else:
            self.marker_style_sect = self.MARKER_STYLE_SECT
            self.marker_style_key = self.MARKER_STYLE_KEY
        self.impact = 0

    def set_marker_style(self, marker):
        marker.set_style(self.marker_style_sect, self.marker_style_key)

    def setup_filter(self, baseplot):
        filter = baseplot.filter
        # Initialisation du filtre
        start_state = filter.new_state()
        # Bouton gauche :
        handler = QtDragHandler(filter, QtCore.Qt.LeftButton, start_state=start_state)
        self.connect(handler, SIG_START_TRACKING, self.start)
        self.connect(handler, SIG_MOVE, self.move)
        self.connect(handler, SIG_STOP_NOT_MOVING, self.stop)
        self.connect(handler, SIG_STOP_MOVING, self.stop)
        return setup_standard_tool_filter(filter, start_state)

    def start(self, filter, event):
        if self.marker is None:
            title = ""
            if self.TITLE:
                title = "<b>%s</b><br>" % self.TITLE
            if self.on_active_item:
                # constraint_cb = filter.plot.on_active_curve
                # constraint_cb = self.on_active_curve
                constraint_cb = lambda x, y: self.on_active_curve(x, y, filter.plot)

                label_cb = lambda x, y: self.on_active_curve_label(x, y, filter.plot)
                """
                label_cb = lambda x, y: title + \
                           filter.plot.get_coordinates_str(x, y)
                """
            else:
                label_cb = lambda x, y: "%sx = %g<br>y = %g" % (title, x, y)
            self.marker = Marker(label_cb=label_cb, constraint_cb=constraint_cb)
            # print self.marker.xValue()   : 0
            self.set_marker_style(self.marker)
        self.marker.attach(filter.plot)
        self.marker.setZ(filter.plot.get_max_z() + 1)
        self.marker.setVisible(True)

    def stop(self, filter, event):
        self.move(filter, event)
        if self.mode != "reuse":
            self.marker.detach()
            self.marker = None
        if self.end_callback:
            self.end_callback(self)

    def move(self, filter, event):
        if self.marker is None:
            return  # something is wrong ...
        self.marker.move_local_point_to(0, event.pos())
        filter.plot.replot()
        self.last_pos = self.marker.xValue(), self.marker.yValue()

    def get_coordinates(self):
        return self.last_pos

    def on_active_curve(self, x, y, plot):
        curve = plot.get_last_active_item(ITrackableItemType)
        if curve:
            # x, y = get_closest_coordinates(x, y,curve)
            ax = curve.xAxis()
            ay = curve.yAxis()
            xc = plot.transform(ax, x)
            yc = plot.transform(ay, y)
            _distance, i, _inside, _other = curve.hit_test(QtCore.QPoint(xc, yc))
            x = curve.x(i)
            y = curve.y(i)
            self.impact = i
        return x, y

    def on_active_curve_label(self, x, y, plot):
        curve = plot.get_last_active_item(ITrackableItemType)
        if curve:
            label = "valeurs des parametres\n"
            for i in range(curve.dataset.nmotors):
                label = (
                    label
                    + curve.dataset.nodenicknames[i]
                    + ":%f\n" % (curve.dataset.data[i, self.impact])
                )
            return label


class Draw3DTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        super(Draw3DTool, self).__init__(
            manager,
            _("Draw a 3D map"),
            get_icon("histogram2D.png"),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        """Activate tool"""
        itemselected = plot.get_selected_items()
        itemselection = list(
            item for item in itemselected if ICurveItemType in item.types()
        )  # on ne prend que les courbes
        trueselection = list(item.dataset for item in itemselection)

        if len(trueselection) == 0:
            print("aucun curveitem selectionne")
            return

        Nitem = len(trueselection)
        item0 = itemselection[0]
        dataset0 = item0.dataset
        Nmotors = dataset0.nmotors

        mins = np.amin(
            np.concatenate(list(item.mins for item in trueselection)).reshape(
                Nitem, Nmotors
            ),
            axis=0,
        )
        maxs = np.amax(
            np.concatenate(list(item.maxs for item in trueselection)).reshape(
                Nitem, Nmotors
            ),
            axis=0,
        )

        xlabel = item0.xlabel
        zlabel = item0.ylabel
        names = dataset0.nodenicknames
        ix = names.index(xlabel)  # on propose par defaut ce qui est trace
        iz = names.index(zlabel)

        meshpar = MeshParameters(ix=ix, iy=0, iz=iz)
        Set3DparametersWindow(
            dataset0, meshpar, NX=item0.dataset.npts, NY=Nitem, mins=mins, maxs=maxs
        )  # ouvre une fenetre de dialogue
        prepare3Dmesh(trueselection, meshpar)


class ImageFileList(QtGui.QListWidget):
    """ A specialized QListWidget that displays the list
        of all nxs files in a given directory and its subdirectories. """

    def __init__(self, parent=None):
        QtGui.QListWidget.__init__(self, parent)
        self.setSelectionMode(3)
        # print "init"
        # item = QtGui.QListWidgetItem(self)
        # item.setText("vide")
        # self.setVisible(1)

    def setDirpath(self, dirpath):
        """ Set the current image directory and refresh the list. """
        self.dirpath = dirpath
        self.populate()

    def getnxs(self):
        """ Return a list of filenames of all
            supported images in self._dirpath. """

        self.nxs = [
            (name, os.path.join(root, name))
            for root, dirs, files in os.walk(self.dirpath)
            for name in files
            if name.endswith((".nxs"))
        ]
        self.nxs.sort()
        """
        self.nxs = [os.path.join(root, name)
          for root, dirs, files in os.walk(self._dirpath)
          for name in files
          if name.endswith((".nxs"))]
        """
        print(len(self.nxs), " files found")
        self.nxsdict = {a[0]: a[1] for a in self.nxs}
        # print self.nxsdict

    def populate(self):
        """ Fill the list with images from the
            current directory in self._dirpath. """

        # In case we're repopulating, clear the list
        self.clear()
        self.getnxs()

        # Create a list item for each image file,
        # setting the text and icon appropriately
        for names in self.nxs:
            shortname = names[0]
            item = QtGui.QListWidgetItem(self)
            item.setText(shortname)
            # item.setIcon(QtGui.QIcon(image))


def prepare3Dmesh(trueselection, meshpar):
    # prepare la mesh en fonction de la liste des scans et des parametres coches
    dataset = trueselection[0]

    xx = np.concatenate(list(item.data[meshpar.ix] for item in trueselection))
    xtitle = dataset.nodenicknames[meshpar.ix]

    if meshpar.iy < len(dataset.data):  # on prend la valeur y
        yy = np.concatenate(list(item.data[meshpar.iy] for item in trueselection))
        ytitle = dataset.nodenicknames[meshpar.iy]
    else:  # on prend le numero dans la liste, le premier scan commence a 1
        yy = np.zeros_like(dataset.data[meshpar.ix]) + 1
        # print len(yy)
        for i in range(1, len(trueselection)):
            ypp = np.zeros_like(trueselection[i].data[meshpar.ix]) + i + 1
            yy = np.concatenate((yy, ypp))
            # print len(yy)
        ytitle = "numero"

    if meshpar.bc:
        # sur chaque scan en z, on fait une correction d'intensite.
        b1 = int(meshpar.b1)
        b2 = int(meshpar.b1)
        if b1 > b2:
            b1, b2 = b2, b1
        b1 = max(0, b1)
        b2 = max(1, b2)
        b1 = min(b1, 99)
        b2 = min(b2, 100)
        fic = open("corr.txt", "w")
        for i in range(len(dataset.nodenicknames)):
            fic.write(dataset.nodenicknames[i])
            fic.write(" ")
        fic.write("background")
        fic.write("\n")
        vm = dict()
        for item in trueselection:
            z1 = np.percentile(item.data[meshpar.iz], b1)
            z2 = np.percentile(item.data[meshpar.iz], b2)
            vm[item] = np.ma.mean(np.ma.masked_outside(item.data[meshpar.iz], z1, z2))
            for i in range(len(dataset.nodenicknames)):
                fic.write("%f " % (item.data[i, 0]))
            fic.write("%f " % (vm[item]))
            fic.write("\n")
            vm[item] = 623 * np.arctan((item.data[15, 0] - 11.73) / 1.24)
            # print vm[item],item.data[:,0]
        zz = np.concatenate(
            list((item.data[meshpar.iz] - vm[item]) for item in trueselection)
        )

    else:
        zz = np.concatenate(list(item.data[meshpar.iz] for item in trueselection))

    bins = (meshpar.NY, meshpar.NX)
    bornes = [[meshpar.ymin, meshpar.ymax], [meshpar.xmin, meshpar.xmax]]

    scanlist = list(item.shortname for item in trueselection)
    scanlist.sort()  # on met en ordre la liste des scans et on prend le premier et le dernier
    ztitle = scanlist[0] + "->" + scanlist[-1]
    ztitle = ztitle + ":" + dataset.nodenicknames[meshpar.iz]

    if meshpar.io:
        # on utilise la matrice d'orientation pour tracer la map
        x2 = xx
        y2 = yy
        try:
            fichier = tables.openFile(dataset.longname)
            for node in fichier.listNodes("/")[0].SIXS:
                if "alpha_star" in node:
                    A_star = node.A_star.read()[0]
                    B_star = node.B_star.read()[0]
                    C_star = node.C_star.read()[0]
                    alpha_star = node.alpha_star.read()[0]
                    beta_star = node.beta_star.read()[0]
                    gamma_star = node.gamma_star.read()[0]
                    # print A_star,B_star,C_star,alpha_star,beta_star,gamma_star

        except Exception:
            print("la matrice d'orientation n'est pas definie")
        xtitle2 = xtitle
        ytitle2 = ytitle
        angle = 90.0  # valeurs par defaut
        unorm = 1.0
        vnorm = 1.0

        if xtitle == "h":
            if ytitle == "k":
                angle = gamma_star
                unorm = A_star
                vnorm = B_star
                xtitle2 = "kx"
                ytitle2 = "ky"
            elif ytitle == "l":
                angle = beta_star
                unorm = A_star
                vnorm = C_star
                xtitle2 = "kx"
                ytitle2 = "kz"
        elif xtitle == "k":
            if ytitle == "h":
                angle = -gamma_star
                unorm = B_star
                vnorm = A_star
                xtitle2 = "ky"
                ytitle2 = "kx"
            elif ytitle == "l":
                angle = alpha_star
                unorm = B_star
                vnorm = C_star
                xtitle2 = "ky"
                ytitle2 = "kz"
        elif xtitle == "l":
            if ytitle == "h":
                angle = -beta_star
                unorm = C_star
                vnorm = A_star
                xtitle2 = "kz"
                ytitle2 = "kx"
            elif ytitle == "k":
                angle = -alpha_star
                unorm = C_star
                vnorm = B_star
                xtitle2 = "kz"
                ytitle2 = "ky"
        angle = angle
        u = unorm * np.array([1, 0])
        v = vnorm * np.array([np.cos(angle * degtorad), np.sin(angle * degtorad)])
        x1 = xx * u[0] + yy * v[0]
        y1 = yy * u[1] + yy * v[1]
        angle2 = meshpar.angle

        cc = np.cos(angle2 * degtorad)
        ss = np.sin(angle2 * degtorad)
        x2 = x1 * cc - y1 * ss
        y2 = y1 * cc + x1 * ss
        u1 = [u[0] * cc - u[1] * ss, u[1] * cc + u[1] * ss]
        v1 = [v[0] * cc - v[1] * ss, v[1] * cc + v[1] * ss]

        bornes = [[np.amin(y2), np.amax(y2)], [np.amin(x2), np.amax(x2)]]
        # print bornes
        win = make3Dmesh(
            x2,
            y2,
            zz,
            zlog=meshpar.ilog,
            bins=bins,
            bornes=bornes,
            xtitle=xtitle2,
            ytitle=ytitle2,
            ztitle=ztitle,
        )
        plot = win.get_plot()
        eps = 1.0e-10
        if unorm == vnorm:
            if abs(angle - 90.0) < eps:
                sg = "p4"
            elif abs(angle - 60.0) < eps:
                sg = "p3"
            else:
                sg = "cm"
        else:
            if abs(angle - 90.0) < eps:
                sg = "p2"
            else:
                sg = "p1"
        shape = D2Dview.GridShape(O=[00, 000], u=u1, v=v1, sg=sg, order=2)
        shape.set_style("plot", "shape/gridshape")
        shape.setTitle("Reseau 1")
        shape.set_selectable(False)
        plot.add_item(shape)

    else:
        make3Dmesh(
            xx,
            yy,
            zz,
            zlog=meshpar.ilog,
            bins=bins,
            bornes=bornes,
            xtitle=xtitle,
            ytitle=ytitle,
            ztitle=ztitle,
        )


def make3Dmesh(
    x, y, z, zlog=0, bins=None, bornes=None, xtitle="x", ytitle="y", ztitle="3D mesh"
):
    # retourne un tableau 2D
    # print x.shape
    # print y.shape
    # print z.shape
    ones = np.ones(z.shape)

    if bins[0] > 1:  # s'il y a plus d'une ligne en y on fait une map
        # on change les bornes pour faire l'histogramme correctement:
        # par exemple si on a pour y 4 valeurs en 1 2 3 4 il faut prendre bornes=(0.5,4.5)
        bornesy = bornes[0]
        deltay = (bornesy[1] - bornesy[0]) / (bins[0] - 1)
        bornesy[0] = bornesy[0] - deltay / 2.0
        bornesy[1] = bornesy[1] + deltay / 2.0

        if zlog:
            H, yedges, xedges = np.histogram2d(
                y, x, bins=bins, range=bornes, weights=np.log(z + 1)
            )
        else:
            H, yedges, xedges = np.histogram2d(y, x, bins=bins, range=bornes, weights=z)

        C, yedges, xedges = np.histogram2d(y, x, bins=bins, range=bornes, weights=ones)
        # fic=open('hist.txt',"w")
        # for i in range(bins[0]):
        #    for j in range (bins[1]):
        #        fic.write("%f "%(H[i,j]))
        #    fic.write("\n")
        # Hc=np.where(C=00,0,H)   # inutile a priori si C=0 H=0

        ## Find indecies of zeros values
        # index = np.where(C==0)

        Cc = np.where(C == 0, 1, C)  # on met 1 pour eviter division par 0
        hist = H / Cc

        mask0 = np.isfinite(hist)

        hist[C == 0] = np.nan
        # histvalues=H[::-1] / Cc[::-1] #image inversee en hauteur
        # hist=make.image(histvalues,xdata=[xedges[0],xedges[-1]],ydata=[yedges[0],yedges[-1]])

        ## Create Boolean array of missing values
        mask = np.isfinite(hist)

        # interpolation des donnees manquantes
        values = hist[mask].flatten()

        ## Find indecies of finite values
        index = np.where(mask == True)

        """
      x0,y0 = index[0],index[1]
      print "valeurs grille non masque"
      print x
      print y
      """
        grid = np.where(mask0 == True)

        ## Grid irregular points to regular grid using delaunay triangulation
        values = griddata(index, values, grid, method="linear").reshape(hist.shape)
        # a la fin, il reste des NAN, si on veut pouvoir ajuster le contraste avec guiqwt, il
        # faut changer True en False dans le module guiqwt.image, get_histogram ligne 672

        win = D2Dview.D2DDialog()
        win.setWindowTitle("3D mesh")
        win.set_data(
            values,
            xdata=[xedges[0], xedges[-1]],
            ydata=[yedges[0], yedges[-1]],
            colormap="jet",
        )
        win.show_image()
        plot = win.get_plot()
        plot.set_axis_title("left", ytitle)
        plot.set_axis_title("bottom", xtitle)
        plot.set_title(ztitle)
        plot.set_axis_direction("left", reverse=False)

    else:  # on trace un scan moyen
        if zlog:
            H, xedges = np.histogram(
                x, bins=bins[1], range=bornes[1], weights=np.log(z + 1)
            )
        else:
            H, xedges = np.histogram(x, bins=bins[1], range=bornes[1], weights=z)

        C, xedges = np.histogram(x, bins=bins[1], range=bornes[1], weights=ones)

        win = CurveDialog(wintitle="Scan moyen")
        plot = win.get_plot()
        plot.set_axis_title("left", ztitle)
        plot.set_axis_title("bottom", xtitle)
        Cc = np.where(C == 0, 1, C)  # on met 1 pour eviter division par 0
        hist = H / Cc
        hist[C == 0] = np.nan  # on met nan la ou il n'y a pas de valeur
        xedges = xedges - (xedges[1] - xedges[0]) / 2.0  # on decale d'un demi pas
        xedges = xedges[1:]
        plot.add_item(make.curve(xedges, hist))
        win.show()
        """

    win.image=hist
    win.setGeometry(QtCore.QRect(420,50,800, 600))
    plot=win.get_plot()
    plot.add_item(hist)
    plot.set_axis_direction("left", reverse=False)
    plot.set_axis_title("left",ytitle)
    plot.set_axis_title("bottom",xtitle)
    plot.font_title.setPointSize (9)
    plot.set_title(ztitle)
    plot.set_aspect_ratio(lock=False)
    win.show()

    """
    """
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #pylab.imshow(H[::-1] / C[::-1], extent=extent, interpolation='nearest', aspect=aspect)
    pylab.imshow(H[::-1] / Cc[::-1], extent=extent, interpolation='nearest')
    pylab.xlim(xedges[0], xedges[-1])
    pylab.ylim(yedges[0], yedges[-1])
    pylab.clim()
    pylab.show()
    """
    return win


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.setGeometry(QtCore.QRect(10, 50, 420, 600))
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "nxsViewer", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.maxpar = 30  # probleme de maxpar a gerer...

        self.initlist()
        defaultfont = QtGui.QFont()
        defaultfont.setPointSize(10)
        QtGui.QApplication.setFont(defaultfont)
        self.listfiles = ImageFileList(parent=self.centralwidget)
        self.listfiles.setGeometry(QtCore.QRect(10, 10, 250, 550))
        self.listfiles.setObjectName(_fromUtf8("listfiles"))

        self.autoadd = QtGui.QRadioButton(self.centralwidget)
        self.autoadd.setGeometry(QtCore.QRect(270, 10, 50, 30))
        self.autoadd.setText("add")
        self.autoadd.setFont(defaultfont)

        self.autoreplace = QtGui.QRadioButton(self.centralwidget)
        self.autoreplace.setGeometry(QtCore.QRect(320, 10, 80, 30))
        self.autoreplace.setText("replace")

        self.Xboxlabel = QtGui.QLabel(self.centralwidget)
        self.Xboxlabel.setGeometry(QtCore.QRect(270, 30, 30, 30))
        self.Xboxlabel.setObjectName(_fromUtf8("Xboxlabel"))
        self.Xboxlabel.setText("X")

        self.Yboxlabel = QtGui.QLabel(self.centralwidget)
        self.Yboxlabel.setGeometry(QtCore.QRect(290, 30, 30, 30))
        self.Yboxlabel.setObjectName(_fromUtf8("Xboxlabel"))
        self.Yboxlabel.setText("Y")

        self.Filterboxlabel = QtGui.QLabel(self.centralwidget)
        self.Filterboxlabel.setGeometry(QtCore.QRect(310, 30, 30, 30))
        self.Filterboxlabel.setObjectName(_fromUtf8("Filterboxlabel"))
        self.Filterboxlabel.setText("Filter")
        """
        self.Zboxlabel = QtGui.QLabel(self.centralwidget)
        self.Zboxlabel.setGeometry(QtCore.QRect(330, 10, 30, 30))
        self.Zboxlabel.setObjectName(_fromUtf8("Xboxlabel"))
        self.Zboxlabel.setText("Z")
        """

        self.XGroup = QtGui.QButtonGroup(self.centralwidget)
        self.XCheckBoxes = list()
        self.YCheckBoxes = list()
        self.FilterGroup = QtGui.QButtonGroup(self.centralwidget)
        self.FilterCheckBoxes = list()

        v = 60

        for param in range(self.maxpar):
            cbox = QtGui.QCheckBox(self.centralwidget)
            cbox.hide()
            cbox.setGeometry(QtCore.QRect(270, v, 20, 20))
            self.XCheckBoxes.append(cbox)
            self.XGroup.addButton(cbox)
            QtCore.QObject.connect(
                cbox,
                QtCore.SIGNAL(_fromUtf8("stateChanged (int)")),
                self.boxselection_changed,
            )

            cbox = QtGui.QCheckBox(self.centralwidget)
            cbox.hide()
            cbox.setGeometry(QtCore.QRect(290, v, 20, 20))
            self.YCheckBoxes.append(cbox)
            QtCore.QObject.connect(
                cbox,
                QtCore.SIGNAL(_fromUtf8("stateChanged (int)")),
                self.boxselection_changed,
            )

            cbox = QtGui.QCheckBox(self.centralwidget)
            cbox.hide()
            cbox.setGeometry(QtCore.QRect(310, v, 300, 20))
            self.FilterCheckBoxes.append(cbox)
            self.FilterGroup.addButton(cbox)
            QtCore.QObject.connect(
                cbox,
                QtCore.SIGNAL(_fromUtf8("stateChanged (int)")),
                self.boxselection_changed,
            )

            v = v + 20
        self.nCheckBoxes = 0  # au debut, pas de cases affichees
        """
        self.XCheckBoxes[self.maxpar-1].setChecked(1)
        self.XCheckBoxes[7].setChecked(1)
        self.YCheckBoxes[0].setChecked(1)
        """

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 20))
        self.menubar.setObjectName(_fromUtf8("menubar"))

        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuFile.setObjectName(_fromUtf8("menuFile"))

        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Edit", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))

        self.menuData = QtGui.QMenu(self.menubar)
        self.menuData.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Data", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuData.setObjectName(_fromUtf8("menuData"))

        self.menuGraph = QtGui.QMenu(self.menubar)
        self.menuGraph.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Graph", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuGraph.setObjectName(_fromUtf8("menuEdit"))

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Open", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionOpen.setObjectName(_fromUtf8("actionImage"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Save", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionSave.setObjectName(_fromUtf8("actionImage"))
        self.actionSave.setDisabled(1)
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionQuit)

        self.actionCopy = QtGui.QAction(MainWindow)
        self.actionCopy.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Copy", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionCopy.setObjectName(_fromUtf8("actionCopy"))
        self.actionClear = QtGui.QAction(MainWindow)
        self.actionClear.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Clear", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionClear.setObjectName(_fromUtf8("actionClear"))

        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addAction(self.actionClear)

        self.actionNewWindow = QtGui.QAction(MainWindow)
        self.actionNewWindow.setText(
            QtGui.QApplication.translate(
                "MainWindow", "New window", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionNewWindow.setObjectName(_fromUtf8("NewWindow "))

        self.actionXLog = QtGui.QAction(MainWindow)
        self.actionXLog.setText(
            QtGui.QApplication.translate(
                "MainWindow", "X log", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionXLog.setObjectName(_fromUtf8("XLog"))
        self.actionXLog.setCheckable(1)

        self.actionYLog = QtGui.QAction(MainWindow)
        self.actionYLog.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Y log", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionYLog.setObjectName(_fromUtf8("YLog"))
        self.actionYLog.setCheckable(1)

        self.actionAutoscale = QtGui.QAction(MainWindow)
        self.actionAutoscale.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Autoscale", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionAutoscale.setObjectName(_fromUtf8("Autoscale"))
        self.actionAutoscale.setCheckable(1)

        self.menuGraph.addAction(self.actionNewWindow)
        self.menuGraph.addAction(self.actionXLog)
        self.menuGraph.addAction(self.actionYLog)
        self.menuGraph.addAction(self.actionAutoscale)

        self.actionFilter = QtGui.QAction(MainWindow)
        self.actionFilter.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Filter policy", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionFilter.setObjectName(_fromUtf8("InitFilter "))

        self.actionInitFit = QtGui.QAction(MainWindow)
        self.actionInitFit.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Fit", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionInitFit.setObjectName(_fromUtf8("InitFit "))

        self.action3DMesh = QtGui.QAction(MainWindow)
        self.action3DMesh.setText(
            QtGui.QApplication.translate(
                "MainWindow", "3DMesh", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.action3DMesh.setObjectName(_fromUtf8("Draw a 3D Mesh "))

        self.actionPreferences = QtGui.QAction(MainWindow)
        self.actionPreferences.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Preferences", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionPreferences.setObjectName(_fromUtf8("Preferences "))

        self.menuData.addAction(self.actionFilter)
        self.menuData.addAction(self.actionInitFit)
        self.menuData.addAction(self.action3DMesh)
        self.menuData.addAction(self.actionPreferences)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuData.menuAction())
        self.menubar.addAction(self.menuGraph.menuAction())
        self.makewindow()
        self.retranslateUi(MainWindow)

        QtCore.QObject.connect(
            self.actionOpen,
            QtCore.SIGNAL(_fromUtf8("triggered()")),
            self.ouvrirFichiers,
        )
        QtCore.QObject.connect(
            self.actionSave, QtCore.SIGNAL(_fromUtf8("triggered()")), self.enregistrer
        )
        QtCore.QObject.connect(
            self.actionQuit, QtCore.SIGNAL(_fromUtf8("triggered()")), self.fermer
        )

        QtCore.QObject.connect(
            self.actionXLog, QtCore.SIGNAL(_fromUtf8("triggered()")), self.setXLog
        )
        QtCore.QObject.connect(
            self.actionYLog, QtCore.SIGNAL(_fromUtf8("triggered()")), self.setYLog
        )
        QtCore.QObject.connect(
            self.actionNewWindow,
            QtCore.SIGNAL(_fromUtf8("triggered()")),
            self.makewindow,
        )

        QtCore.QObject.connect(
            self.actionFilter, QtCore.SIGNAL(_fromUtf8("triggered()")), self.setfilter
        )
        QtCore.QObject.connect(
            self.actionInitFit, QtCore.SIGNAL(_fromUtf8("triggered()")), self.initfit
        )
        QtCore.QObject.connect(
            self.action3DMesh, QtCore.SIGNAL(_fromUtf8("triggered()")), self.draw3Dmesh
        )
        QtCore.QObject.connect(
            self.actionPreferences,
            QtCore.SIGNAL(_fromUtf8("triggered()")),
            self.setpreferences,
        )

        QtCore.QObject.connect(
            self.listfiles,
            QtCore.SIGNAL(_fromUtf8("itemSelectionChanged ()")),
            self.item_selectionchanged,
        )
        QtCore.QObject.connect(
            self.listfiles,
            QtCore.SIGNAL(_fromUtf8("itemClicked (QListWidgetItem *)")),
            self.item_clicked,
        )

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.clipboard = QtGui.QApplication.clipboard()

        # self.Image.setFocus()
        self.csuite = (
            "#00bfff",
            "#ff6a6a",
            "#98fb98",
            "#ffd700",
            "#ee82e2",
            "#e9967a",
            "#4169e1",
            "#ff0000",
            "#00ff00",
            "#ffa500",
        )
        self.icolor = 0
        self.listtitles = list()
        direct = "E:\Geoffroy-Boulot\Au-Cu\Sixs-decembre2013"
        self.listfiles.setDirpath(direct)
        self.autoreplace.setChecked(1)
        self.isafit = 0
        self.lastitem = None
        self.nXBoxesChecked = 0
        self.nYBoxesChecked = 0
        self.datafilter = DataFilter()
        self.pref = PrefParameters()
        self.meshpar = MeshParameters()

    def refreshboxes(self, dataset):
        # on met à jour la liste des cases a cocher en fonction du dataset selectionne
        if dataset.nmotors > self.nCheckBoxes:  # on rajoute des cases
            for i in range(self.nCheckBoxes, dataset.nmotors):
                self.XCheckBoxes[i].show()
                self.YCheckBoxes[i].show()
                self.FilterCheckBoxes[i].show()

        elif dataset.nmotors < self.nCheckBoxes:  # on rajoute des cases
            for i in range(dataset.nmotors, self.nCheckBoxes):
                self.XCheckBoxes[i].hide()
                self.YCheckBoxes[i].hide()
                self.FilterCheckBoxes[i].hide()

                if self.XCheckBoxes[i].isChecked():
                    self.XCheckBoxes[i].setChecked(0)
                if self.FilterCheckBoxes[i].isChecked():
                    self.FilterCheckBoxes[i].setChecked(0)
                """
                a priori pas necessaire en Y, comme ca on garde en memoire les cases cochees
                if self.YCheckBoxes[i].isChecked():
                    self.YCheckBoxes[i].setChecked(0)
                """
        self.nXBoxesChecked = 0
        self.nYBoxesChecked = 0

        for i in range(dataset.nmotors):
            self.FilterCheckBoxes[i].setText(dataset.nodenicknames[i])
            if self.XCheckBoxes[i].isChecked():
                self.nXBoxesChecked = self.nXBoxesChecked + 1
            if self.YCheckBoxes[i].isChecked():
                self.nYBoxesChecked = self.nYBoxesChecked + 1

        self.nCheckBoxes = dataset.nmotors

    def initfit(self):
        # ici on inclue une figure de fit provenant de grafit
        if self.isafit == 0:
            from grafit2.grafit2 import Ui_FitWindow

            #           import grafit2 as grafit
            self.figfit = Ui_FitWindow()
            #           self.figfit=grafit.Ui_FitWindow()
            self.figfit.setupUi()
            self.figfit.setupdatestart(self.pref.autoupdate)
            # tags est une liste de couples [non,valeur]
            self.figfit.settags([], saveint=True)

            self.isafit = 1
            # ici on redirige le signal de fermeture de la fenetre de fit
            self.figfit.closeEvent = self.closefit

        self.figfit.move(10, 500)
        self.figfit.show()
        self.updatefit()

    def closefit(self, closeevent):
        self.isafit = 0

    def showfit(self, itemselected):
        x, y = itemselected.get_data()
        xlabel = itemselected.xlabel
        ylabel = itemselected.ylabel
        xmin, xmax = self.plot.get_axis_limits("bottom")

        y2 = np.compress((x >= xmin) & (x <= xmax), y)
        x2 = np.compress((x >= xmin) & (x <= xmax), x)

        # on recupere les donnees du scan (angles, H,K,L) pour les envoyer a grafit
        try:
            fichier = tables.openFile(itemselected.dataset.longname)
            group = fichier.listNodes("/")[0]
            leafnames = [
                "I14-C-CX2__EX__DIFF-UHV-H__#1/raw_value",
                "I14-C-CX2__EX__DIFF-UHV-K__#1/raw_value",
                "I14-C-CX2__EX__DIFF-UHV-L__#1/raw_value",
                "SIXS/I14-C-CX2__EX__mu-uhv__#1/raw_value",
                "SIXS/I14-C-CX2__EX__omega-uhv__#1/raw_value",
                "SIXS/I14-C-CX2__EX__delta-uhv__#1/raw_value",
                "SIXS/I14-C-CX2__EX__gamma-uhv__#1/raw_value",
            ]
            leafshortnames = ["H", "K", "L", "mu", "omega", "delta", "gamma"]
            leafs = [group._f_getChild(leafname) for leafname in leafnames]
            leafvalues = list(leaf.read()[0] for leaf in leafs)
            tags = zip(leafshortnames, leafvalues)

        except Exception:
            tags = []

        self.figfit.setvalexp(
            x2,
            y2,
            xlabel=xlabel,
            ylabel=ylabel,
            title=itemselected.curveparam.label,
            xlog=self.actionXLog.isChecked(),
            ylog=self.actionYLog.isChecked(),
            tags=tags,
            followscans=self.pref.followscans,
        )
        self.figfit.show()

    def updatefit(self):
        # print "update fit"

        itemselected = self.plot.get_active_item(force=False)

        if itemselected == None:
            itemselected = self.lastitem

        if itemselected == None:
            pass

        elif ICurveItemType in itemselected.types():
            # on a selectionne une curve
            # print "curve"
            if self.isafit:
                self.showfit(itemselected)

    def initlist(self):
        self.paramlist = [
            "y",
            "ybrut",
            "filters",
            "mu",
            " ",
            "delta",
            "gamma",
            "h",
            "k",
            "l",
            "q",
        ]

    def fermer(self):
        if self.isafit:
            self.figfit.close()
        self.win.close()
        MainWindow.close()

    def enregistrer(self):
        pass

    def setXLog(self):
        if self.actionXLog.isChecked():
            self.plot.set_axis_scale("bottom", "log")
        else:
            self.plot.set_axis_scale("bottom", "lin")
        self.plot.show_items()

    def setYLog(self):
        if self.actionYLog.isChecked():
            self.plot.set_axis_scale("left", "log")
        else:
            self.plot.set_axis_scale("left", "lin")
        self.plot.show_items()

    def changeZ(self):
        pass

    def retranslateUi(self, MainWindow):
        pass

    def drawcurve(self, dataset):
        kid = self.XGroup.checkedId()  # a priori il y  a toujours un X de selectionne
        if kid == -1:
            print("no X selected")
        else:
            ix = -kid - 2
            absi = dataset.data[ix]
        for iy in range(dataset.nmotors):
            if self.YCheckBoxes[iy].isChecked():
                # print i,params[i]
                xlabel = dataset.nodenicknames[ix]
                ylabel = dataset.nodenicknames[iy]

                title = dataset.shortname + "_" + ylabel + "(" + xlabel + ")"

                if (
                    title not in self.listtitles
                ):  # on ne retrace pas des courbes deja faites

                    self.lastitem = self.drawgraph(
                        absi,
                        dataset.data[iy],
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )
                    self.lastitem.dataset = dataset
                    self.plot.set_titles(xlabel=xlabel, ylabel=ylabel)

                    self.addplot(self.lastitem)
                    self.icolor = (self.icolor + 1) % 10
                    self.listtitles.append(title)

    def draw3Dmesh(self):
        trueselection = list()
        scanlist = list()
        for item in self.listfiles.selectedItems():
            shortname = str(item.text())
            longname = self.listfiles.nxsdict[shortname]
            dataset = DataSet(
                longname, shortname, datafilter=self.datafilter, pref=self.pref
            )
            trueselection.append(dataset)
            scanlist.append(shortname)

        if len(trueselection) == 0:
            print("aucun item selectionne")
            return

        Nitem = len(trueselection)
        Nmotors = dataset.nmotors

        mins = np.amin(
            np.concatenate(list(dataset.mins for dataset in trueselection)).reshape(
                Nitem, Nmotors
            ),
            axis=0,
        )
        maxs = np.amax(
            np.concatenate(list(dataset.maxs for dataset in trueselection)).reshape(
                Nitem, Nmotors
            ),
            axis=0,
        )
        Set3DparametersWindow(
            dataset, self.meshpar, NX=dataset.npts, NY=Nitem, mins=mins, maxs=maxs
        )  # ouvre une fenetre de dialogue
        prepare3Dmesh(trueselection, self.meshpar)

    def makewindow(self):
        self.win = CurveDialog(
            edit=False,
            toolbar=True,
            wintitle="Scan window",
            options=dict(xlabel="xlabel", ylabel="ylabel"),
        )
        self.win.setGeometry(QtCore.QRect(440, 50, 800, 600))
        self.plot = self.win.get_plot()
        self.win.get_itemlist_panel().show()
        self.plot.set_items_readonly(False)
        self.win.add_tool(
            SelectMultiPointTool, title=None, on_active_item=True, mode="create"
        )
        self.win.add_tool(Draw3DTool)
        self.win.show()
        self.addplot(make.legend())
        self.plot.connect(self.plot, SIG_ACTIVE_ITEM_CHANGED, self.updatefit)
        # self.win.exec_()

    def addplot(self, *items):

        for item in items:
            self.plot.add_item(item)
        # self.plot.activateWindow() met la fenêtre self.win au premier plan
        if self.actionAutoscale.isChecked():
            self.plot.do_autoscale(replot=True)
        self.plot.show_items()

    def drawgraph(self, absi, ordo, title=None, xlabel=None, ylabel=None):

        """i
        #inutile on a mis des filtres
        if self.actionYLog.isChecked():
            ymin=min(ordo)
            #print ymin
            if ymin<=0.0001:
                ymax=max(ordo)+1
                ycorr=np.where(ordo <= 0.0001, ymax+1, ordo)
                ymin=min(ycorr)
                ordo=np.where(ycorr > ymax, ymin, ycorr)
        """
        item = make.curve(absi, ordo, title=title, color=self.csuite[self.icolor])
        item.xlabel = xlabel
        item.ylabel = ylabel

        return item

    def ouvrirFichiers(self):

        self.dirfiles = QtGui.QFileDialog.getExistingDirectory(None, "Browse Directory")
        print(str(self.dirfiles))
        self.listfiles.setDirpath(str(self.dirfiles))

    def item_selectionchanged(self):
        # self.item_clicked(self.listfiles.selectedItems()[0])

        if self.autoreplace.isChecked():  # on efface
            self.icolor = 0
            self.listtitles = list()
            self.plot.del_all_items()
            self.addplot(make.legend())

        if (
            self.nXBoxesChecked > 0 and self.nYBoxesChecked > 0
        ):  # pas la peine de perdre du temps si rien n'est coche
            for item in self.listfiles.selectedItems():  # on ajoute les items
                self.init_item(item)
            if self.pref.autoupdate:
                self.updatefit()  # on update le fit avec le dernier item si self.pref.autoupdate est True

    def item_clicked(self, item):
        # le fichier a lire est obtenu par concatenation du directory et du filename

        self.shortname = str(item.text())
        longname = self.listfiles.nxsdict[self.shortname]

        self.currentdataset = DataSet(
            longname, self.shortname, datafilter=self.datafilter, pref=self.pref
        )
        self.refreshboxes(self.currentdataset)

    def init_item(self, item):
        self.shortname = str(item.text())
        longname = self.listfiles.nxsdict[self.shortname]

        self.currentdataset = DataSet(
            longname, self.shortname, datafilter=self.datafilter, pref=self.pref
        )
        self.refreshboxes(self.currentdataset)
        self.isacurve = 1
        self.drawcurve(self.currentdataset)

    def boxselection_changed(self, state):

        self.nXBoxesChecked = 0
        self.nYBoxesChecked = 0

        kid = self.FilterGroup.checkedId()
        self.datafilter.ifil = -kid - 2

        for i in range(self.nCheckBoxes):
            if self.XCheckBoxes[i].isChecked():
                self.nXBoxesChecked = self.nXBoxesChecked + 1
            if self.YCheckBoxes[i].isChecked():
                self.nYBoxesChecked = self.nYBoxesChecked + 1
        self.item_selectionchanged()

    def setfilter(self):
        # ouvre une fenetre pour le reglage des filtres
        SetFilterWindow(self.datafilter)

    def setpreferences(self):
        # ouvre une fenetre pour le reglage des filtres
        SetPrefWindow(self.pref)
        if self.isafit:
            self.figfit.setupdatestart(self.pref.autoupdate)


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.listfiles.setDirpath(os.getcwd())
    MainWindow.show()
    sys.exit(app.exec_())
