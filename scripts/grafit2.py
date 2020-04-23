# -*- coding: utf-8 -*-
"""
Created on Fri Dec 06 12:15:46 2013

@author: prevot

Objectif: remplacer les courbes par des instances de classe


FitCurve.nom : nom de la courbe
FitCurve.function : fonction associee
FitCurve.parameters : parametres standards de la courbe
FitCurve.letter : lettre symbolique de la fonction
FitCurve.getIntegral() : calcule l'integrale pour les fonctions le permettant
FitCurve.RectangularActionTool : le type d'outil de tracage
FitCurve.action : le comportement de la souris au traçage

"""

import copy
import numpy as np

from os import path, getcwd

from guiqwt.plot import CurveWidget, PlotManager
from guiqwt.builder import make

from guidata.qt.QtCore import Qt, SIGNAL, QRegExp, QRect, QSize, QPoint, QString
from guidata.qt.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QMenu,
    QAction,
    QCursor,
    QColor,
    QIcon,
    QMessageBox,
    QInputDialog,
    QFileDialog,
    QDialog,
    QMainWindow,
    QSplitter,
    QApplication,
    QPushButton,
    QButtonGroup,
    QSpinBox,
    QRadioButton,
    QCheckBox,
)
from guidata.qthelpers import add_actions, get_std_icon

import sys, weakref
from guiqwt.events import RectangularSelectionHandler, setup_standard_tool_filter
from guiqwt.tools import (
    OpenFileTool,
    CommandTool,
    DefaultToolbarID,
    RectangularActionTool,
    BaseCursorTool,
)
from guiqwt.shapes import XRangeSelection
from guiqwt.signals import SIG_END_RECT, SIG_RANGE_CHANGED, SIG_ITEM_REMOVED
from guiqwt.config import _
from guiqwt.interfaces import ICurveItemType, IShapeItemType
from lmfit import minimize, Parameters, Parameter
from scipy.special import erf
from scipy.integrate import simps

SHAPE_Z_OFFSET = 1000

try:
    _fromUtf8 = QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

abspath = path.abspath(__file__)
dirpath = path.dirname(abspath)

abspath = getcwd()
runpath = path.join(dirpath, "Run.png")
gausspath = path.join(dirpath, "Gauss.png")
eraserpath = path.join(dirpath, "Eraser.png")
steppath = path.join(dirpath, "Step.png")
doorpath = path.join(dirpath, "Door.png")
gaussderpath = path.join(dirpath, "GaussDer.png")
gausslorpath = path.join(dirpath, "Skew.png")
skewpath = path.join(dirpath, "Skew.png")
yfullrangepath = path.join(dirpath, "y_full_range.png")


class Highlighter(QSyntaxHighlighter):
    PATTERNS = [" ", "\t", ","]

    def __init__(self, parent, patternid=0):
        super(Highlighter, self).__init__(parent)
        self.highlightFormat = QTextCharFormat()
        self.highlightFormat.setForeground(Qt.blue)
        self.highlightFormat.setBackground(Qt.green)
        self.setpattern(patternid)

    def setpattern(self, patternid):
        self.expression = QRegExp(self.PATTERNS[patternid])

    def highlightBlock(self, text):
        index = self.expression.indexIn(text)
        while index >= 0:
            length = self.expression.matchedLength()
            self.setFormat(index, length, self.highlightFormat)
            index = text.indexOf(self.expression, index + length)


# fonctions de fit
def lorentz(x, p):
    # p[0] max d'intensite, p[1] centre, p[2],FWHM
    return p[0] / (1.0 + ((x - p[1]) / 0.5 / p[2]) ** 2)


def lorentzder(x, p):
    # p[0] max d'intensite, p[1] centre, p[2],FWHM
    return p[0] * (p[1] - x) / p[2] / (1.0 + ((x - p[1]) / 0.5 / p[2]) ** 2) ** 2


def gauss(x, p):
    w = 0.600561204393225
    return p[0] * np.exp(-(((x - p[1]) / w / p[2]) ** 2))


def gausslor(x, p):
    # gaussian switching to lorentzian for x-p[1]=p[2]*p[3]
    if p[3] > 0:
        w = 0.600561204393225
        p32 = p[3] * p[2]
        pp1 = p32 + p[1] - ((w * p[2]) ** 2) / p32
        pp0 = p[0] * np.exp(-(((p32) / w / p[2]) ** 2)) * (p[1] + p32 - pp1) ** 2
        b = np.greater(x, p[1] + p32)
        return (p[0] * np.exp(-(((x - p[1]) / w / p[2]) ** 2))) * (1 - b) + (
            pp0 / (x - pp1) ** 2
        ) * b
    elif p[3] < 0:
        w = 0.600561204393225
        p32 = p[3] * p[2]
        pp1 = p32 + p[1] - ((w * p[2]) ** 2) / p32
        pp0 = p[0] * np.exp(-(((p32) / w / p[2]) ** 2)) * (p[1] + p32 - pp1) ** 2
        b = np.less(x, p[1] + p32)
        return (p[0] * np.exp(-(((x - p[1]) / w / p[2]) ** 2))) * (1 - b) + (
            pp0 / (x - pp1) ** 2
        ) * b
    else:
        return gauss(x, p)


def gaussder(x, p):
    # derivee d'une gaussienne
    w = 0.600561204393225
    return p[0] * (p[1] - x) / p[2] * np.exp(-(((x - p[1]) / w / p[2]) ** 2))


def skew(x, p):
    w = 0.600561204393225
    return (
        p[0]
        * np.exp(-(((x - p[1]) / w / p[2]) ** 2))
        * (1.0 + erf((x - p[1]) * p[3] / w / p[2]))
    )


def voigt(x, p):
    w = 0.600561204393225
    return p[0] * (
        (1.0 - p[3]) * np.exp(-(((x - p[1]) / w / p[2]) ** 2))
        + p[3] / (1.0 + ((x - p[1]) / 0.5 / p[2]) ** 2)
    )


def step(x, p):
    w = 0.600561204393225
    return 0.5 * p[0] * (1.0 - erf((x - p[1]) / w / p[2]))


def door(x, p):
    return (
        p[0]
        / 2.0
        * (erf((x - p[1] + p[2] / 2.0) / p[3]) - erf((x - p[1] - p[2] / 2.0) / p[3]))
    )


# integrales des fonctions
# integrales des fonctions
def Ilorentz(p):
    return p[0] * p[2] * np.pi / 2.0


def Igauss(p):
    wpi = 1.064467019431226
    return p[0] * p[2] * wpi


def Ivoigt(p):
    wpi = 1.064467019431226
    return p[0] * p[2] * ((1.0 - p[3]) * wpi + p[3] * np.pi / 2.0)


def Igausslor(p):
    w = 0.600561204393225
    wpis2 = 0.532233509715613

    p32 = np.abs(p[3] * p[2])
    pp = ((w * p[2]) ** 2) / p32
    pp1 = p32 + p[1] - pp

    pp0 = p[0] * np.exp(-(((p32) / w / p[2]) ** 2)) * (p[1] + p32 - pp1) ** 2
    return p[0] * p[2] * wpis2 * (1.0 + erf(p[3] / p[2])) + pp0 / pp


# dictionnaire reliant la fonction au nom
curvetypes = dict(
    gaussian=gauss,
    lorentzian=lorentz,
    voigt=voigt,
    door=door,
    step=step,
    gaussian_derivative=gaussder,
    lorentz_derivative=gaussder,
    gaussian_lorentzian=gausslor,
    skew=skew,
)
# dictionnaire reliant l'integrale au nom
inttypes = dict(
    gaussian=Igauss,
    lorentzian=Ilorentz,
    voigt=Ivoigt,
    door=None,
    step=None,
    gaussian_derivative=None,
    lorentz_derivative=None,
    gaussian_lorentzian=Igausslor,
    skew=Igauss,
)

curvenames = list(
    (
        "gaussian",
        "lorentzian",
        "voigt",
        "door",
        "step",
        "gaussian_derivative",
        "lorentz_derivative",
        "gaussian_lorentzian",
        "skew",
    )
)
curveparams = dict(
    gaussian=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
    ),
    lorentzian=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
    ),
    voigt=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0, None),
        ("eta", 0.5, 0.0, 1.0),
    ),
    door=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0, None),
        ("steep", 0.25, 0.0, None),
    ),
    step=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("steep", 1.0, 0.0, None),
    ),
    gaussian_derivative=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
    ),
    lorentzian_derivative=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
    ),
    gaussian_lorentzian=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
        ("eps", 0.1, None, None),
    ),
    skew=(
        ("int", 1.0, None, None),
        ("pos", 0.0, None, None),
        ("fwhm", 1.0, 0.0, None),
        ("alpha", 0.0, None, None),
    ),
)

# parametres de reference communs aux differentes courbes
int0 = Parameter("int", 1.0, min=0.0)
pos0 = Parameter("pos", 0.0)
fwhm0 = Parameter("fwhm", 1.0, min=0.0)
eta0 = Parameter("eta", 0.5, min=0.0, max=1.0)
steep0 = Parameter("steep", 0.25, min=0.0)
alpha0 = Parameter("alpha", 0.0)
eps0 = Parameter("alpha", 0.0)


# Emitted when a fit is performed
SIG_FIT_DONE = SIGNAL("fit_done")

# Emitted when a fit is performed
SIG_FIT_SAVED = SIGNAL("fit_saved")


class FitCurve:
    # definit une classe pour les fonction de fit
    def __init__(self, name, curvetype, list_of_parameters, integral=None):
        self.name = name
        self.curvetype = curvetype
        self.Parameters = Parameters()
        for parameter in list_of_parameters:
            self.Parameters.__setitem__(parameter.name, parameter)


gaussparams = FitCurve("gaussian", gauss, (int0, pos0, fwhm0), Igauss)
lorentzparams = FitCurve("lorentzian", lorentz, (int0, pos0, fwhm0), Ilorentz)
voigtparams = FitCurve("voigt", voigt, (int0, pos0, fwhm0, eta0), Ivoigt)
doorparams = FitCurve("door", door, (int0, pos0, fwhm0, steep0))
stepparams = FitCurve("step", step, (int0, pos0, steep0))
gaussderparams = FitCurve("gaussian_derivative", gaussder, (int0, pos0, fwhm0))
lorentzderparams = FitCurve("lorentzian_derivative", gaussder, (int0, pos0, fwhm0))
gausslorparams = FitCurve("gaussian_lorentzian", gausslor, (int0, pos0, fwhm0, eps0))
skewparams = FitCurve("skew", skew, (int0, pos0, fwhm0, alpha0))


def model(params, x, fixparams):
    # fixparams : parametres fixes
    # nombre de courbes utilisees
    degpol = fixparams[0]
    curveslist = fixparams[1]

    ncurv = len(curveslist)
    # degre du polynome : degpol

    p = np.empty(degpol + 1)
    for i in range(degpol + 1):
        name = "pol%d" % (i)
        p[i] = params[name].value

    yth = poly(x, p, degpol)

    for i in range(ncurv):
        i1 = i + 1
        kstr = "%d" % (i1)
        curvename = curveslist[i]
        p = list()
        for cp in curveparams[curvename]:
            entry = cp[0] + kstr
            p.append(params[entry].value)
        yth = yth + curvetypes[curvename](x, p)
    return yth


def integral(params, curveslist):
    # retourne la liste des valeurs des integrales
    # le premier point est l'integrale totale

    ncurv = len(curveslist)
    intlist = list()
    inttot = 0.0
    for i in range(ncurv):
        i1 = i + 1
        kstr = "%d" % (i1)
        curvename = curveslist[i]
        p = list()
        for cp in curveparams[curvename]:
            ent = cp[0]
            entry = ent + kstr
            p.append(params[entry].value)
        intfunct = inttypes[curvename]
        if intfunct is not None:
            intpart = intfunct(p)  # on recupere la fonction qui calcule l'integrale
            intlist.append(intpart)
            inttot = inttot + intpart
        else:
            intlist.append(None)
    intlist.insert(0, inttot)
    return intlist


def residual(params, x, data, fixparams):
    # fparam : parametres fixes
    return model(params, x, fixparams) - data


## Parametric function: 'p' is the parameter vector, 'x' the independent varibles
"""
def poly(x,p,degpol):
    y=p[0]
    xp=1
    for i in range(degpol):
        xp=xp*x
        y=y+xp*p[i+1]
    return y
"""


def poly(x, p, degpol):
    y = 0
    # p[0]
    xp = 1
    for i in range(degpol + 1):
        y = y + xp * p[i]
        xp = xp * x
    return y


class TagsWindow(QDialog):
    # definit une fenetre pour rentrer l'affichage des tags
    def __init__(self, tags):

        QDialog.__init__(self)
        self.setWindowTitle(_fromUtf8("List of tags"))
        self.list = QListWidget(self)

        self.list = QListWidget(self)
        self.list.setGeometry(QRect(5, 5, 200, 300))
        for tag in tags:

            tagname = tag[0]
            tagvalue = ": %f" % tag[1]
            text = tagname + tagvalue
            item = QListWidgetItem(self.list)
            item.setText(text)

        self.OK = QPushButton(self)
        self.OK.setGeometry(QRect(5, 310, 100, 25))
        self.OK.setText("OK")

        self.setGeometry(QRect(5, 30, 210, 350))

        self.connect(self.OK, SIGNAL(_fromUtf8("clicked()")), self.appl)

        self.exec_()

    def appl(self):
        self.close()


class readfileparams:
    # definit une classe pour la lecture des parametres d'un fichier a lire
    def __init__(self):
        self.heading = 0  # taille de l'entete
        self.title = False  # ligne de titre
        self.delimiter = 0  # separateur de nombre
        self.x = 1  # indice colonne x
        self.y = 2  # indice colonne y


class readfilewindow(QDialog):
    def __init__(self, params, ficlines):
        self.params = params
        super(
            readfilewindow, self
        ).__init__()  # permet l'initialisation de la fenetre sans perdre les fonctions associees
        self.setWindowTitle("Parameters for reading file")
        self.setFixedSize(QSize(410, 320))
        self.ficlines = ficlines

        self.display = QTextEdit(self)
        self.display.setReadOnly(True)
        self.display.setGeometry(QRect(5, 5, 400, 100))
        self.display.setLineWrapMode(0)
        self.highlighter = Highlighter(self.display.document(), self.params.delimiter)

        self.lab1 = QLabel(self)
        self.lab1.setGeometry(QRect(5, 110, 50, 25))
        self.lab1.setText("heading")

        self.lab2 = QLabel(self)
        self.lab2.setGeometry(QRect(5, 140, 50, 25))
        self.lab2.setText("title")

        self.lab3 = QLabel(self)
        self.lab3.setGeometry(QRect(5, 170, 50, 25))
        self.lab3.setText("delimiter")

        self.lab4 = QLabel(self)
        self.lab4.setGeometry(QRect(5, 200, 50, 25))
        self.lab4.setText("x column")

        self.lab4 = QLabel(self)
        self.lab4.setGeometry(QRect(5, 230, 50, 25))
        self.lab4.setText("y column")

        self.heading = QSpinBox(self)
        self.heading.setGeometry(QRect(60, 110, 45, 25))
        self.heading.setValue(params.heading)
        self.heading.setMaximum(len(ficlines))

        self.title = QRadioButton(self)
        self.title.setGeometry(QRect(60, 140, 45, 25))
        self.title.setChecked(params.title)

        self.delimiter0 = QCheckBox(self)
        self.delimiter0.setGeometry(QRect(60, 170, 65, 25))
        self.delimiter0.setText("space")

        self.delimiter1 = QCheckBox(self)
        self.delimiter1.setGeometry(QRect(130, 170, 65, 25))
        self.delimiter1.setText("tab")

        self.delimiter2 = QCheckBox(self)
        self.delimiter2.setGeometry(QRect(200, 170, 45, 25))
        self.delimiter2.setText("comma")

        self.delimiters = QButtonGroup(self)

        self.delimiters.addButton(self.delimiter0, 0)
        self.delimiters.addButton(self.delimiter1, 1)
        self.delimiters.addButton(self.delimiter2, 2)
        self.delimiters.button(params.delimiter).setChecked(True)

        self.x = QSpinBox(self)
        self.x.setGeometry(QRect(60, 200, 45, 25))
        self.x.setValue(params.x)

        self.y = QSpinBox(self)
        self.y.setGeometry(QRect(60, 230, 45, 25))
        self.y.setValue(params.y)

        self.OK = QPushButton(self)
        self.OK.setGeometry(QRect(5, 260, 90, 25))
        self.OK.setText("OK")

        self.Cancel = QPushButton(self)
        self.Cancel.setGeometry(QRect(100, 260, 90, 25))
        self.Cancel.setText("Cancel")

        self.connect(
            self.heading, SIGNAL(_fromUtf8("valueChanged(int)")), self.changeheading
        )
        self.QObject.connect(self.Cancel, SIGNAL(_fromUtf8("clicked()")), self.closewin)
        self.QObject.connect(self.OK, SIGNAL(_fromUtf8("clicked()")), self.appl)
        self.QObject.connect(
            self.delimiters, SIGNAL(_fromUtf8("buttonClicked(int)")), self.setpattern
        )
        self.changeheading(self.params.heading)

        self.exec_()

    def setpattern(self, patternid):
        self.highlighter.setpattern(patternid)
        self.display.setText(self.text)

    def changeheading(self, value):
        self.params.heading = value
        # first line
        n1 = min(value, len(self.ficlines) - 1)
        n2 = min(value + 20, len(self.ficlines))
        self.text = ""
        for i in range(n1, n2):
            self.text = self.text + self.ficlines[i]

        self.display.setText(self.text)

    def appl(self):
        # try:
        self.params.heading = int(self.heading.text())
        self.params.title = self.title.isChecked()
        self.params.x = int(self.x.text())
        self.params.y = int(self.y.text())
        self.params.delimiter = self.delimiters.checkedId()

        # except Exception:
        #    QMessageBox.about(self, 'Error','Input can only be a number')
        #    return

        self.close()

    def closewin(self):
        self.close()


class FitTable(QTableWidget):
    # objet qui gere les entree de parametres, l'affichage des courbes, le lancement des fits, l'enregistrement du resultat
    def __init__(self, parent=None, extendedparams=None):
        QTableWidget.__init__(self, parent=parent)

        self.setColumnCount(8)
        self.setObjectName(_fromUtf8("tableValeurs"))
        self.setHorizontalHeaderLabels(
            (
                "Parameters",
                "Estimation",
                "Fit result",
                "Sigma",
                "Restrains",
                "Min",
                "Max",
                "Expression",
            )
        )

        # on remet le tableau a zero : deux rangees pour un polynome du 1er degre
        self.reset(extendedparams)

        # on definit les differents menus contextuels
        self.initcontexts()

        # au cas ou
        self.int1 = 0

        # utilise pour contourner le signal de cellChanged
        self.trackchange = True

        # On connecte les signaux
        self.connect(self, SIGNAL(_fromUtf8("cellChanged(int,int)")), self.cellChanged)
        self.connect(self, SIGNAL(_fromUtf8("cellPressed(int,int)")), self.cellPressed)
        self.connect(
            self.horizontalHeader(),
            SIGNAL(_fromUtf8("sectionPressed(int)")),
            self.sectionClicked,
        )

        # pas de fit sauve
        self.isfitted = False
        self.issaved = False
        self.tags = list()
        self.saveint = False
        self.ints = []
        self.updatestart = (
            False  # si coche, on active setwholeastry apres enregistrement
        )
        self.cwd = abspath

    def reset(self, extendedparams=None):
        if extendedparams is None:
            # on met deux items vide
            extendedparams = ExtendedParams()
            extendedparams.addbackground()
            extendedparams.addbackground()
            # on remplit le tableau a l'aide du dictionnaire extendedparams
        self.initparams(extendedparams, settry=True)
        self.savename = None
        self.isfitted = False

    def inititems(self, listrow):
        # met pour les lignes i de listrow un item vide non editable pour colonnes 0,2,3,4
        for i in listrow:
            for j in range(8):
                # self.setItem(i,j,QTableWidgetItem(_fromUtf8("")))
                self.setItem(i, j, QTableWidgetItem(0))
            self.item(i, 0).setFlags(Qt.ItemFlags(33))
            self.item(i, 2).setFlags(Qt.ItemFlags(33))
            self.item(i, 3).setFlags(Qt.ItemFlags(33))
            self.item(i, 4).setFlags(Qt.ItemFlags(33))

    def initparams(self, extendedparams=None, settry=False, setopt=False):
        self.trackchange = False

        # on redessine le tableau a l'aide du dictionnaire extendedparams
        # si settry, on remplit aussi les valeurs d'essai, les bornes, les expressions
        if extendedparams is not None:
            # on utilise les valeurs donnees en input, sinon on prend celles de self.extendedparams
            self.extendedparams = extendedparams

        n = 0
        oldrc = self.rowCount()
        newrc = len(self.extendedparams.partry) - len(
            self.extendedparams.tags
        )  # on n'affiche pas les tags
        self.setRowCount(newrc)

        if newrc > oldrc:
            # on rajoute des rangees vides
            self.inititems(range(oldrc, newrc))

        for k in range(self.extendedparams.degpol + 1):
            kstr = "%d" % (k)
            entry = "pol" + kstr
            self.setrow(entry, n, settry, setopt)
            n = n + 1

        for k in range(self.extendedparams.ncurves):
            curvename = self.extendedparams.curveslist[k]
            for cp in curveparams[curvename]:
                ent = cp[0]
                entry = ent + "%d" % (k + 1)
                self.setrow(entry, n, settry, setopt)
                n = n + 1
        self.trackchange = True

    def setrow(self, nom, n, settry=True, setopt=False):
        # appele uniquement dans initparams. donc on ne change pas self.trackchange
        # on ne change pas expr car on veut pouvoir la garder en memoire meme si on ne l'utilise pas
        ptry = self.extendedparams.partry
        popt = self.extendedparams.paropt
        self.item(n, 0).setText(_fromUtf8(nom))
        """
        self.setItem(n,0,QTableWidgetItem(_fromUtf8(nom)))
        self.setItem(n,2,QTableWidgetItem(_fromUtf8("")))
        self.setItem(n,3,QTableWidgetItem(_fromUtf8("")))
        self.setItem(n,4,QTableWidgetItem(_fromUtf8("")))
        self.item(n,0).setFlags(Qt.ItemFlags(33))
        self.item(n,2).setFlags(Qt.ItemFlags(33))
        self.item(n,3).setFlags(Qt.ItemFlags(33))
        self.item(n,4).setFlags(Qt.ItemFlags(33))
        """

        if settry:
            vtry = ptry[nom].value

            mintry = ptry[nom].min
            maxtry = ptry[nom].max
            print(nom, mintry, maxtry)
            varytry = ptry[nom].vary
            exprtry = ptry[nom].expr
            self.item(n, 1).setText(_fromUtf8("%f" % (vtry)))
            # self.setItem(n,1,QTableWidgetItem(_fromUtf8("%f" % (vtry))))

            if exprtry is not None:
                # self.setItem(n,4,QTableWidgetItem(_fromUtf8(exprtry)))
                self.item(n, 4).setText(_fromUtf8("Expr"))
            elif varytry is True:
                # self.setItem(n,4,QTableWidgetItem(_fromUtf8("Free")))
                self.item(n, 4).setText(_fromUtf8("Free"))
            else:
                # self.setItem(n,4,QTableWidgetItem(_fromUtf8("Fixed")))
                self.item(n, 4).setText(_fromUtf8("Fixed"))

            if mintry is not None:
                # self.setItem(n,5,QTableWidgetItem(_fromUtf8("%f" % (mintry))))
                self.item(n, 5).setText(_fromUtf8("%f" % (mintry)))

            if maxtry is not None:
                # self.setItem(n,6,QTableWidgetItem(_fromUtf8("%f" % (maxtry))))
                self.item(n, 6).setText(_fromUtf8("%f" % (maxtry)))

        # here we rewrite the expression even if it is not used
        self.item(n, 7).setText(_fromUtf8(ptry[nom].express))

        if setopt and self.isfitted:
            vopt = popt[nom].value
            # self.setItem(n,2,QTableWidgetItem(_fromUtf8("%f" % (vopt))))
            self.item(n, 2).setText(_fromUtf8("%f" % (vopt)))
            verr = popt[nom].stderr
            if verr is not None:
                self.item(n, 3).setText(_fromUtf8("%f" % (verr)))

        else:
            self.item(n, 2).setText(_fromUtf8(""))
            self.item(n, 3).setText(_fromUtf8(""))
        n = n + 1

        """
        exemple:
        self["degpol"]=1
        self["ncurves"]=2
        self["curveslist"]=({courbe:gaussienne,parametre:(int,pos,fwhm)},{courbe:voigt,parametre:(int,pos,fwhm,eta)})
        self["partry"]=orderer dict
        """

    def setactivecurve(self, val):
        # defini la courbe active qui va etre modifie a la souris
        self.activecurve = val

    def initcontexts(self):
        # initialise une liste de menus contextuels pour les cases, associes a chaque colonne
        self.contexts = list(None for i in range(self.columnCount()))
        # initialise une liste de menus contextuels pour les sectionheaders, associes a chaque colonne
        self.Contexts = list(None for i in range(self.columnCount()))

        # menu pour la colonne 0
        contexta = QMenu(self)
        self.addbg = QAction("Increase background order", self)
        contexta.addAction(self.addbg)
        self.connect(self.addbg, SIGNAL(_fromUtf8("triggered()")), self.add_bg)

        self.rmbg = QAction("Decrease background order", self)
        contexta.addAction(self.rmbg)
        self.connect(self.rmbg, SIGNAL(_fromUtf8("triggered()")), self.rm_bg)

        self.rmbg = QAction("Decrease background order", self)
        contexta.addAction(self.rmbg)
        self.connect(self.rmbg, SIGNAL(_fromUtf8("triggered()")), self.rm_bg)

        self.freezebg = QAction("Fixed background", self)
        contexta.addAction(self.freezebg)
        self.connect(self.freezebg, SIGNAL(_fromUtf8("triggered()")), self.freeze_bg)

        self.releasebg = QAction("Free background", self)
        contexta.addAction(self.releasebg)
        self.connect(self.releasebg, SIGNAL(_fromUtf8("triggered()")), self.release_bg)

        contextb = QMenu(self)
        self.addcv = QAction("Add a curve", self)
        contextb.addAction(self.addcv)
        self.connect(self.addcv, SIGNAL(_fromUtf8("triggered()")), self.set_cv)

        self.rmcv = QAction("Remove this curve", self)
        contextb.addAction(self.rmcv)
        self.connect(self.rmcv, SIGNAL(_fromUtf8("triggered()")), self.rm_cv)

        self.chcv = QAction("Reconfigure curve", self)
        contextb.addAction(self.chcv)
        self.connect(self.chcv, SIGNAL(_fromUtf8("triggered()")), self.ch_cv)

        self.freezecv = QAction("Fixed curve", self)
        contextb.addAction(self.freezecv)
        self.connect(self.freezecv, SIGNAL(_fromUtf8("triggered()")), self.freeze_cv)

        self.releasecv = QAction("Free curve", self)
        contextb.addAction(self.releasecv)
        self.connect(self.releasecv, SIGNAL(_fromUtf8("triggered()")), self.release_cv)

        self.contexts[0] = (contexta, contextb)

        # menu pour la colonne 2

        context = QMenu(self)

        self.setastry = QAction("Set as try", self)
        context.addAction(self.setastry)
        self.connect(self.setastry, SIGNAL(_fromUtf8("triggered()")), self.set_as_try)

        self.showfit = QAction("Display Fit", self)
        context.addAction(self.showfit)
        self.connect(self.showfit, SIGNAL(_fromUtf8("triggered()")), self.showfitpart)

        self.showtry = QAction("Display Try", self)
        context.addAction(self.showtry)
        self.connect(self.showtry, SIGNAL(_fromUtf8("triggered()")), self.drawtrypart)

        self.contexts[2] = context

        # menu pour la colonne 4
        context = QMenu(self)

        self.setfixed = QAction("Fixed", self)
        context.addAction(self.setfixed)
        self.connect(self.setfixed, SIGNAL(_fromUtf8("triggered()")), self.set_fixed)

        self.setfree = QAction("Free", self)
        context.addAction(self.setfree)
        self.connect(self.setfree, SIGNAL(_fromUtf8("triggered()")), self.set_free)

        self.useexpr = QAction("Expr", self)
        context.addAction(self.useexpr)
        self.connect(self.useexpr, SIGNAL(_fromUtf8("triggered()")), self.use_expr)

        self.contexts[4] = context

        # menu pour le header 2
        context = QMenu(self)
        self.setwholeastry = QAction("Set whole as try", self)
        context.addAction(self.setwholeastry)
        self.connect(
            self.setwholeastry, SIGNAL(_fromUtf8("triggered()")), self.set_wholeastry
        )

        self.Contexts[2] = context

        # menu pour le header 4
        context = QMenu(self)
        self.inverserestrains = QAction("Inverse restrains", self)
        context.addAction(self.inverserestrains)
        self.connect(
            self.inverserestrains,
            SIGNAL(_fromUtf8("triggered()")),
            self.inverse_restrains,
        )

        self.fixall = QAction("All fixed", self)
        context.addAction(self.fixall)
        self.connect(self.fixall, SIGNAL(_fromUtf8("triggered()")), self.fix_all)

        self.releaseall = QAction("All free", self)
        context.addAction(self.releaseall)
        self.connect(
            self.releaseall, SIGNAL(_fromUtf8("triggered()")), self.release_all
        )

        self.Contexts[4] = context

    def cellPressed(self, int1, int2):
        self.int1 = int1

        context = self.contexts[int2]

        if context is not None:

            if int2 == 0:
                if int1 <= self.extendedparams.degpol:
                    context[0].popup(QCursor.pos())
                else:
                    context[1].popup(QCursor.pos())
            else:
                context.popup(QCursor.pos())
        # self.extendedparams.printparams()

    def cellChanged(self, int1, int2):

        if self.trackchange:
            # print  "Changed at",int1,int2
            if int2 == 1:  # on change la valeur du parametre
                txt = str(self.item(int1, 1).text())
                entry = str(self.item(int1, 0).text())
                self.extendedparams.partry[entry].value = float(txt)
                self.drawtry()

            if int2 == 5:  # on change la borne inferieure
                txt = str(self.item(int1, 5).text())
                entry = str(self.item(int1, 0).text())
                self.extendedparams.partry[entry].min = float(txt)

            if int2 == 6:  # on change la borne superieure
                txt = str(self.item(int1, 6).text())
                entry = str(self.item(int1, 0).text())
                self.extendedparams.partry[entry].max = float(txt)

            if int2 == 7:  # on change l'expression
                txt = str(self.item(int1, 7).text())
                entry = str(self.item(int1, 0).text())
                self.extendedparams.partry[entry].express = txt  # a copy of expression
                if len(txt) == 0:
                    self.extendedparams.partry[entry].expr = None
                    self.extendedparams.partry[entry].vary = True
                else:
                    self.extendedparams.partry[entry].expr = txt

                    self.trackchange = False
                    self.item(int1, 4).setText(_fromUtf8("Expr"))
                    self.trackchange = True

    def sectionClicked(self, int2):
        # quand une colonne entiere est selectionnee valable pour les colonnes 2 et 4
        context = self.Contexts[int2]
        if context is not None:
            context.popup(QCursor.pos())
        # self.extendedparams.printparams()

    def add_bg(self):
        # on ajoute un ordre au polynome de fond
        self.extendedparams.addbackground()
        self.isfitted = False
        self.initparams(settry=True)
        self.drawtry()

    def rm_bg(self):
        # on enleve un ordre au polynome de fond
        self.extendedparams.removebackground()
        self.isfitted = False
        self.initparams(settry=True)
        self.drawtry()

    def freeze_bg(self):
        # on met fixe tous les parametres du fond
        self.trackchange = False
        for int1 in range(self.extendedparams.degpol + 1):
            item = self.item(int1, 4)
            item.setText(_fromUtf8("Fixed"))
            entry = str(self.item(int1, 0).text())
            self.extendedparams.partry[entry].vary = False
        self.trackchange = True

    def release_bg(self):
        # on met fixe tous les parametres du fond
        self.trackchange = False
        for int1 in range(self.extendedparams.degpol + 1):
            item = self.item(int1, 4)
            item.setText(_fromUtf8("Free"))
            entry = str(self.item(int1, 0).text())
            self.extendedparams.partry[entry].vary = True
        self.trackchange = True

    def freeze_cv(self):
        # on met fixe tous les parametres d'une courbe
        self.trackchange = False
        entry = str(self.item(self.int1, 0).text())
        icurve = self.extendedparams.partry[entry].number
        self.extendedparams.freezecurve(icurve)
        self.initparams(settry=True, setopt=True)
        self.trackchange = True

    def release_cv(self):
        # on met fixe tous les parametres d'une courbe
        self.trackchange = False
        entry = str(self.item(self.int1, 0).text())
        icurve = self.extendedparams.partry[entry].number
        self.extendedparams.releasecurve(icurve)
        self.initparams(settry=True, setopt=True)
        self.trackchange = True

    def release_all(self):
        # on met fixe tous les parametres du fond
        self.trackchange = False
        for int1 in range(self.rowCount()):
            item = self.item(int1, 4)
            item.setText(_fromUtf8("Free"))
            entry = str(self.item(int1, 0).text())
            self.extendedparams.partry[entry].vary = True
        self.trackchange = True

    def fix_all(self):
        # on met fixe tous les parametres du fond
        self.trackchange = False
        for int1 in range(self.rowCount()):
            item = self.item(int1, 4)
            item.setText(_fromUtf8("Fixed"))
            entry = str(self.item(int1, 0).text())
            self.extendedparams.partry[entry].vary = False
        self.trackchange = True

    def set_cv(self, curvetype=None, params=None, bg=0.0):
        # icurve numero de la courbe a changer le cas echeant -1 c'est la derniere
        print("bg", bg)
        if curvetype is None:
            # on ouvre une fenetre pour demander le choix de courbe
            self.isfitted = False
            cvtyp, ok = QInputDialog.getItem(
                self, "curve type", "curve choice", curvenames, editable=False
            )
            if ok:
                self.extendedparams.addcurve(str(cvtyp))

                self.initparams(settry=True)
                self.drawtry()
        else:
            # on prend les donnees en parametre
            if self.activecurve == "new":
                self.isfitted = False
                # plot.lastcurve=make.curve(xtry, ytry, color="k",title='gaussienne')
                # plot.add_item(plot.lastcurve)
                self.extendedparams.addcurve(curvetype, params)
                # si la courbe est la premiere courbe, on met le fond a la valeur bg
                icurve = len(self.extendedparams.curveslist) - 1
                if icurve == 0:
                    self.extendedparams.partry["pol0"].value = bg
                self.initparams(settry=True)
                self.plot.newcurve = False
                self.activecurve = -1
            else:
                # plot.lastcurve.set_data(xtry, ytry)
                self.extendedparams.changecurve(
                    paramvalues=params, icurve=self.activecurve
                )
                print(self.activecurve)
                # si la courbe est la premiere courbe, on met le fond a la valeur bg
                if self.activecurve == 0:
                    self.extendedparams.partry["pol0"].value = bg
                elif self.activecurve == -1:
                    icurve = len(self.extendedparams.curveslist) - 1
                    if icurve == 0:
                        self.extendedparams.partry["pol0"].value = bg
                self.initparams(settry=True)
            self.drawtry()

    def rm_cv(self):
        # on supprime la courbe selectionnee
        entry = str(self.item(self.int1, 0).text())
        icurve = self.extendedparams.partry[entry].number
        self.isfitted = False
        if icurve > 0:  # normalement c'est toujours le cas
            self.extendedparams.removecurve(icurve)
            self.initparams(settry=True)
            self.drawtry()
        self.parent().parent().manager.activate_default_tool()

    def ch_cv(self):
        # a partir de la souris, on retrace la courbe selectionnee
        entry = str(self.item(self.int1, 0).text())
        i = self.extendedparams.partry[entry].number
        if i > 0:  # normalement c'est toujours le cas
            curvename = self.extendedparams.curveslist[i - 1]
            manager = self.parent().parent().manager
            self.activecurve = i - 1
            fittool = manager.get_tool(FitTool)
            if curvename == "gaussian":
                fittool.gausstool.activate()
            elif curvename == "lorentzian":
                fittool.lorentztool.activate()
            elif curvename == "voigt":
                fittool.voigttool.activate()
            elif curvename == "step":
                fittool.steptool.activate()
            elif curvename == "door":
                fittool.doortool.activate()
            elif curvename == "gaussian_derivative":
                fittool.gaussdertool.activate()
            elif curvename == "lorentzian_derivative":
                fittool.lorentzdertool.activate()
            elif curvename == "gaussian_lorentzian":
                fittool.gausslortool.activate()
            elif curvename == "skew":
                fittool.skewtool.activate()

            self.initparams(settry=True)
            self.drawtry()

        # self.gausscurve.action,self.lorentzcurve.action,self.voigtcurve.action,self.stepcurve.action,self.doorcurve.action

    def set_fixed(self):
        self.trackchange = False
        item = self.item(self.int1, 4)
        item.setText(_fromUtf8("Fixed"))
        entry = str(self.item(self.int1, 0).text())
        self.extendedparams.partry[entry].vary = False
        self.extendedparams.partry[entry].expr = None
        # nothing to do with self.extendedparams.partry[entry].express, we keep it in memory
        self.trackchange = True

    def set_free(self):
        self.trackchange = False
        item = self.item(self.int1, 4)
        item.setText(_fromUtf8("Free"))
        entry = str(self.item(self.int1, 0).text())
        self.extendedparams.partry[entry].vary = True
        self.extendedparams.partry[entry].expr = None
        # nothing to do with self.extendedparams.partry[entry].express, we keep it in memory
        self.trackchange = True

    def use_expr(self):
        self.trackchange = False
        item = self.item(self.int1, 4)
        item.setText(_fromUtf8("Expr"))
        entry = str(self.item(self.int1, 0).text())

        txt = str(self.item(self.int1, 7).text())
        self.extendedparams.partry[
            entry
        ].express = txt  # in principle, should have been already set

        if len(txt) != 0:  # on met l'expression indiquee
            self.extendedparams.partry[entry].expr = txt
            self.extendedparams.partry[entry].vary = False

        self.trackchange = True

    def inverse_restrains(self):
        # inverse les contraintes: les parametres libres deviennent fixes et vice-versa
        self.trackchange = False
        for int1 in range(self.rowCount()):
            item = self.item(int1, 4)
            entry = str(self.item(int1, 0).text())
            vary = self.extendedparams.partry[entry].vary
            if vary:
                self.extendedparams.partry[entry].vary = False
                item.setText(_fromUtf8("Fixed"))
            else:
                self.extendedparams.partry[entry].vary = True
                item.setText(_fromUtf8("Free"))
        self.trackchange = True

    def set_wholeastry(self):
        self.trackchange = False
        self.extendedparams.setwholefitastry()
        self.initparams(settry=True, setopt=True)
        self.trackchange = True

    def set_as_try(self):
        self.trackchange = False
        item = self.item(self.int1, 1)
        entry = str(self.item(self.int1, 0).text())
        if entry in self.extendedparams.paropt:
            self.extendedparams.partry[entry].value = self.extendedparams.paropt[
                entry
            ].value
            item.setText("%f" % (self.extendedparams.paropt[entry].value))
            self.drawtry()
        self.trackchange = True

    def drawtrypart(self):
        pass

    def showfitpart(self):
        pass

    def startfit(self):
        self.extendedparams.startfit()
        self.isfitted = True
        self.issaved = False
        self.drawopt()
        self.initparams(settry=False, setopt=True)
        self.emit(SIG_FIT_DONE)

    def drawtry(self):
        xmin, xmax = self.plot.get_axis_limits("bottom")
        xtry = np.linspace(xmin, xmax, num=10000)
        ytry = model(
            self.extendedparams.partry,
            xtry,
            (self.extendedparams.degpol, self.extendedparams.curveslist),
        )
        self.plot.curvetry.set_data(xtry, ytry)
        self.plot.show_items((self.plot.curveexp, self.plot.curvetry))
        self.plot.hide_items((self.plot.curveopt,))

    def drawopt(self):
        xmin, xmax = self.plot.get_axis_limits("bottom")
        xopt = np.linspace(xmin, xmax, num=10000)
        yopt = model(
            self.extendedparams.paropt,
            xopt,
            (self.extendedparams.degpol, self.extendedparams.curveslist),
        )
        self.plot.curveopt.set_data(xopt, yopt)
        self.plot.show_items((self.plot.curveexp, self.plot.curveopt))
        self.plot.hide_items((self.plot.curvetry,))

    def save_fit(self):
        # sauvegarde des donnees
        if self.savename is None:
            self.setfilename()  # definit le nom de fichier

        if self.savename is not None:
            # dans le cas contraire, c'est qu'on a annule a l'etape precedente
            self.fic = open(self.savename, "a")

            # on ecrit le nom des variables
            lig = "Scan "
            # on ecrit le nom des variables passes en tags
            for kk in self.tags:
                lig = lig + kk[0] + " "

            # on ecrit la valeur des integrales brutes
            if self.saveint:
                lig = lig + "int_tot "

                for i in range(len(self.extendedparams.curveslist)):
                    lig = lig + "int_%d " % (i + 1)

            # on ecrit le nom des variables de fit
            for kk in self.extendedparams.partry:
                lig = lig + kk + " "
            lig = lig + " type \n"
            # print lig
            self.fic.write(lig)

            # on ecrit la valeur des variables
            title = self.scantitle

            lig = title + " "
            # on ecrit la valeur des variables passes en tags
            for kk in self.tags:
                lig = lig + kk[1] + " "

            # on ecrit la valeur des integrales brutes
            if self.saveint:
                self.ints = integral(
                    self.extendedparams.paropt, self.extendedparams.curveslist
                )
                lig = lig + "%g " % (self.ints[0])
                for i in range(len(self.extendedparams.curveslist)):
                    if self.ints[i + 1] is None:
                        lig = lig + "None "
                    else:
                        lig = lig + "%g " % (self.ints[i + 1])

            for kk in self.extendedparams.partry:
                lig = lig + "%g " % (self.extendedparams.paropt[kk].value)
            typ = ""
            for func in self.extendedparams.curveslist:
                typ = typ + func[0]  # on garde en abreviation le 1er caractere "
            lig = lig + typ + "\n"
            # print lig
            self.fic.write(lig)

            # on ecrit une ligne avec les erreurs
            lig = "sigma "
            if self.saveint:
                # ici on n'a pas integre l'evaluation des integrale car elle necessite
                # la diagonalisation de la matrice des covariances.
                lig = lig + "0. "
                for i in range(len(self.extendedparams.curveslist)):
                    lig = lig + "0. "
            for kk in self.extendedparams.partry:
                if self.extendedparams.paropt[kk].stderr is None:
                    lig = lig + "None "
                else:
                    lig = lig + "%g " % (self.extendedparams.paropt[kk].stderr)
            lig = lig + typ + "\n"
            self.fic.write(lig)
            self.fic.close()

            # on sauve la figure de fit
            ific = 1

            figshortname = self.scantitle + "_%.3d.png" % (ific)
            figname = path.join(self.cwd, figshortname)
            while path.isfile(figname):
                ific = ific + 1
                figshortname = self.scantitle + "_%.3d.png" % (ific)
                figname = path.join(self.cwd, figshortname)
            print(figname, " sauve")
            self.plot.save_widget(figname)
            self.issaved = True
            if self.updatestart:
                self.set_wholeastry()

            self.emit(SIG_FIT_DONE)

    def setfilename(self):
        self.savename = str(
            QFileDialog.getSaveFileName(
                None, "Save fit parameters", self.cwd, filter="*.txt"
            )
        )
        if len(self.savename) == 0:
            self.savename = None
        else:
            self.cwd = path.dirname(path.realpath(self.savename))

    def rangemove(self, shape):
        # on recalcule le masque
        items = self.plot.get_items(z_sorted=False, item_type=IShapeItemType)
        ranges = list()
        for item in items:
            ranges.append(item.get_range())

        if shape not in items:
            ranges.append(shape.get_range())
            # alors il s'agit d'une nouvelle XRangeSelection
        self.extendedparams.mask(ranges)

    def rangeremove(self, shape):
        self.setrange()

    def setrange(self):
        # on recalcule le masque
        items = self.plot.get_items(z_sorted=False, item_type=IShapeItemType)
        ranges = list()
        for item in items:
            ranges.append(item.get_range())
        self.extendedparams.mask(ranges)

    def shiftguess(self, shift):
        # on shifte toutes les positions guess des courbes
        # attention, si les positions sont fixes, ca decale les courbes quand meme
        # si on ne veut pas decaler, alors il faut mettre la valeur dans Expression
        for int1 in range(self.rowCount()):
            txt = str(self.item(int1, 1).text())
            entry = str(self.item(int1, 0).text())

            if "pos" in entry[:3]:  # alors c'est un parametre de position a shifter
                val = float(txt) + shift
                self.item(int1, 1).setText("%f" % (val))


class ExtendedParams:
    # dictionnaire de la liste des courbes pour le fit
    # utilise pour comprendre les Parameters de lmfit
    # ne pilote jamais le tracage des courbes et l'ecriture dans le tableau
    def __init__(self):
        self.degpol = -1  # degre du polynome de fond
        self.ncurves = 0  # nombre de courbes
        self.curveslist = (
            list()
        )  # liste d'un dictionnaire comprenant type de courbe et nom des parametres
        self.partry = Parameters()  # parametres d'essai
        self.paropt = Parameters()  # parametres optimises
        self.xexp = np.zeros((1,))
        self.yexp = np.zeros((1,))
        self.tags = []

    def printparams(self):
        print("printparams")
        for key in self.partry.keys():
            print(key, self.partry[key].value, self.partry[key].vary)

    def addbackground(self):
        k = self.degpol + 1
        kstr = "%d" % (k)
        entry = "pol" + kstr
        self.degpol = k
        self.partry.add(entry, value=0.00)
        self.partry[
            entry
        ].number = (
            0  # on rajoute une reference au nombre de la courbe (0 pour le polynome)
        )
        self.partry[
            entry
        ].express = ""  # on ajoute a la classe une variable 'express' qui est la valeur de l'expression pas forcement appliquee (pour la garder en memoire)

    def removebackground(self):
        k = self.degpol
        kstr = "%d" % (k)
        entry = "pol" + kstr
        del self.partry[entry]
        self.degpol = k - 1

    def addcurve(self, curvename="gausian", paramvalues=None):
        # ajoute une courbe dans la liste parametres
        self.curveslist.append(curvename)
        k = self.ncurves + 1
        if paramvalues is None:
            # on prend les valeurs par defaut
            for ent, val, vmin, vmax in curveparams[curvename]:
                entry = ent + "%d" % (k)
                self.partry.add(entry, value=val)
                self.partry[entry].express = ""
                self.partry[entry].number = k
                if vmin is not None:
                    self.partry[entry].min = vmin
                if vmax is not None:
                    self.partry[entry].max = vmax

        else:
            # on prend les valeurs de paramvalues
            for cp, val in zip(curveparams[curvename], paramvalues):
                entry = cp[0] + "%d" % (k)
                self.partry.add(entry, value=val)
                self.partry[entry].express = ""
                self.partry[entry].number = k
                vmin = cp[2]
                vmax = cp[3]
                if vmin is not None:
                    self.partry[entry].min = vmin
                if vmax is not None:
                    self.partry[entry].max = vmax

        self.ncurves = k

    def freezecurve(self, k):
        # freeze tout les parametres associes a la courbe
        curvename = self.curveslist[k - 1]
        print(curvename)
        for cp in curveparams[curvename]:
            ent = cp[0]
            entry = ent + "%d" % (k)
            self.partry[entry].vary = False
            print(entry, "Fixed")

    def releasecurve(self, k):
        # freeze tout les parametres associes a la courbe
        curvename = self.curveslist[k - 1]
        print(curvename)
        for cp in curveparams[curvename]:
            ent = cp[0]
            entry = ent + "%d" % (k)
            self.partry[entry].vary = True
            print(entry, "Free")

    def removecurve(self, k):
        ncurve = len(self.curveslist)
        curvename = self.curveslist.pop(k - 1)

        for cp in curveparams[curvename]:
            ent = cp[0]
            entry = ent + "%d" % (k)
            del self.partry[entry]

        # reorganisation des courbes
        for k2 in range(k, ncurve):
            curvename = self.curveslist[k2 - 1]
            for cp in curveparams[curvename]:
                ent = cp[0]
                entry = ent + "%d" % (k2 + 1)
                entry2 = ent + "%d" % (k2)  # on redescend les courbes d'une unite
                self.partry.add(
                    entry2,
                    value=self.partry[entry].value,
                    vary=self.partry[entry].vary,
                    min=self.partry[entry].min,
                    max=self.partry[entry].max,
                    expr=self.partry[entry].expr,
                )
                self.partry[entry2].number = k2
                self.partry[entry2].express = ""
                del self.partry[entry]

        self.ncurves = self.ncurves - 1

    def changecurve(self, paramvalues=None, icurve=-1):
        # icurve est le numero de la liste des courbes, a partir de 0
        curvename = self.curveslist[icurve]
        k = icurve + 1
        if icurve == -1:
            k = len(self.curveslist)

        if paramvalues is None:
            # on prend les valeurs par defaut
            for cp in curveparams[curvename]:
                ent = cp[0]
                val = cp[1]
                entry = ent + "%d" % (k)
                self.partry[entry].value = val
                # print entry,self.partry[entry]
        else:
            # on prend les valeurs de paramvalues
            for cp, val in zip(curveparams[curvename], paramvalues):
                entry = cp[0] + "%d" % (k)
                self.partry[entry].value = val

    def setvalexp(self, xexp, yexp):
        self.xexp = xexp
        self.yexp = yexp
        self.xexp0 = xexp
        self.yexp0 = yexp

    def mask(self, ranges):
        xexp = np.ma.array(self.xexp0)
        for rang in ranges:
            xexp = np.ma.masked_inside(xexp, rang[0], rang[1])
        mask = np.ma.getmaskarray(xexp)

        yexp = np.ma.array(self.yexp0, mask=mask)
        self.xexp = xexp.compressed()
        self.yexp = yexp.compressed()

    def startfit(self):
        # self.paropt=copy.deepcopy(self.partry)
        """
        print "self.paropt:",self.paropt
        print "self.degpol:",self.degpol
        print "self.curveslist:",self.curveslist
        print "self.xexp",self.xexp
        print "self.yexp",self.yexp
        """
        # modification of lmfit

        result = minimize(
            residual,
            self.partry,
            args=(self.xexp, self.yexp, (self.degpol, self.curveslist)),
        )
        self.paropt = result.params
        result.params.pretty_print()
        """
        for kk in self.partry:
            pkt=self.partry[kk].value
            pkmi=self.partry[kk].min
            pkma=self.partry[kk].max
            pko=self.paropt[kk].value
            pkv=self.paropt[kk].vary
            pke=self.paropt[kk].expr
            print kk,pkt,pko,pkv,pkmi,pkma,pke
        """
        self.integrate()
        print(integral(self.paropt, self.curveslist))

    def integrate(self):
        # calcule l'integrale brute en enlevant le fond
        self.int1 = 0.0
        self.int2 = 0.0
        ncurv = len(self.curveslist)
        if ncurv > 0:
            integraltot = simps(self.yexp, self.xexp)
            # integraltot=np.sum(self.yexp)
            ybg = model(self.paropt, self.xexp, (self.degpol, list()))
            # self.int1=integraltot-np.sum(ybg)
            self.int1 = integraltot - simps(ybg, self.xexp)
            print("integrale brute fond soustrait:", self.int1)
            # calcule l'integrale brute en enlevant toutes les courbes sauf la premiere
            self.parpart = copy.deepcopy(self.paropt)
            self.parpart["int1"].value = 0.0
            ybg = model(self.parpart, self.xexp, (self.degpol, self.curveslist))
            # self.int2=integraltot-np.sum(ybg)
            self.int2 = integraltot - simps(ybg, self.xexp)
            print("integrale brute fond + courbes >1 soustraits:", self.int2)
            yth = model(self.paropt, self.xexp, (-1, self.curveslist))
            self.int3 = simps(yth, self.xexp)
            print("integrale numerique courbes simulees:", self.int3)

    def setwholefitastry(self):
        for kk in self.partry:
            if kk in self.paropt:
                self.partry[kk].value = self.paropt[kk].value

    def updatetags(self, tags):
        # on supprime les entrees precedentes:
        for tag in self.tags:
            if tag[0] in tags:  # le tag existe encore, on update sa valeur
                self.partry[tag[0]].value = tag[1]
            else:
                del self.partry[tag[0]]  # le tag a disparu, on le supprime

        for tag in tags:
            if tag[0] not in tags:  # le tag n'existe pas encore, on l'ajoute
                self.partry.add(tag[0], value=tag[1], vary=False)
        self.tags = tags


class MaskTool(BaseCursorTool):
    TITLE = _("Mask data")
    ICON = "xrange.png"
    SWITCH_TO_DEFAULT_TOOL = True

    def __init__(
        self, manager, toolbar_id=DefaultToolbarID, title=None, icon=None, tip=None
    ):
        super(MaskTool, self).__init__(
            manager, toolbar_id, title=title, icon=icon, tip=tip
        )
        self._last_item = None

    def get_last_item(self):
        if self._last_item is not None:
            return self._last_item()

    def create_shape(self):
        return XRangeSelection(0, 0)

    def move(self, filter, event):
        super(MaskTool, self).move(filter, event)

    def end_move(self, filter, event):
        super(MaskTool, self).end_move(filter, event)

    def get_associated_item(self, plot):
        items = plot.get_selected_items(item_type=ICurveItemType)
        if len(items) == 1:
            self._last_item = weakref.ref(items[0])
        return self.get_last_item()

    def update_status(self, plot):
        pass
        # item = self.get_associated_item(plot)
        # self.action.setEnabled(item is not None)


class FitTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(
            self,
            manager,
            _("Fit"),
            icon=gausspath,
            tip=_("Add curve for fit"),
            toolbar_id=toolbar_id,
        )
        self.manager = manager

    def create_action_menu(self, manager):
        # Create and return menu for the tool's action
        """
        self.gausscurve = manager.add_tool(RectangularActionToolCX,actiongauss, toolbar_id=None, icon="rectangle.png",title = "Gauss")
        self.lorentzcurve = manager.add_tool(RectangularActionToolCX,actionlorentz, toolbar_id=None, icon="rectangle.png",title = "Lorentz")
        self.voigtcurve = manager.add_tool(RectangularActionToolCX,actionvoigt, toolbar_id=None, icon="rectangle.png",title = "Voigt")
        self.stepcurve = manager.add_tool(RectangularActionToolCXY ,actionstep, toolbar_id=None, icon="rectangle.png",title = "Step")
        self.doorcurve = manager.add_tool(RectangularActionToolCX,actiondoor, toolbar_id=None, icon="rectangle.png",title = "Door")
        """
        gaussaction = QAction(QIcon(gausspath), "Gaussian", self)
        self.connect(gaussaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewgauss)

        lorentzaction = QAction(QIcon(gausspath), "Lorentzian", self)
        self.connect(
            lorentzaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewlorentz
        )

        voigtaction = QAction(QIcon(gausspath), "Voigt", self)
        self.connect(voigtaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewvoigt)

        stepaction = QAction(QIcon(steppath), "Step", self)
        self.connect(stepaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewstep)

        dooraction = QAction(QIcon(doorpath), "Door", self)
        self.connect(dooraction, SIGNAL(_fromUtf8("triggered()")), self.drawnewdoor)

        gaussderaction = QAction(QIcon(gaussderpath), "Gaussian_derivative", self)
        self.connect(
            gaussderaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewgaussder
        )

        lorentzderaction = QAction(QIcon(gaussderpath), "Lorentz_derivative", self)
        self.connect(
            lorentzderaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewlorentzder
        )

        gaussloraction = QAction(QIcon(gausslorpath), "Gaussian_lorentzian", self)
        self.connect(
            gaussloraction, SIGNAL(_fromUtf8("triggered()")), self.drawnewgausslor
        )

        skewaction = QAction(QIcon(skewpath), "Skew", self)
        self.connect(skewaction, SIGNAL(_fromUtf8("triggered()")), self.drawnewskew)

        resetaction = QAction(QIcon(eraserpath), "Reset", self)
        self.connect(resetaction, SIGNAL(_fromUtf8("triggered()")), self.reset)

        menu = QMenu()
        # add_actions(menu, (self.gausscurve.action,self.lorentzcurve.action,self.voigtcurve.action,self.stepcurve.action,self.doorcurve.action,resetaction))
        add_actions(
            menu,
            (
                gaussaction,
                lorentzaction,
                voigtaction,
                stepaction,
                dooraction,
                gaussderaction,
                lorentzderaction,
                gaussloraction,
                skewaction,
                resetaction,
            ),
        )

        self.gausstool = self.manager.add_tool(
            RectangularActionToolCX,
            actiongauss,
            toolbar_id=None,
            icon="rectangle.png",
            title="Gauss",
        )
        self.lorentztool = self.manager.add_tool(
            RectangularActionToolCX,
            actionlorentz,
            toolbar_id=None,
            icon="rectangle.png",
            title="Lorentz",
        )
        self.voigttool = self.manager.add_tool(
            RectangularActionToolCX,
            actionvoigt,
            toolbar_id=None,
            icon="rectangle.png",
            title="Voigt",
        )
        self.steptool = self.manager.add_tool(
            RectangularActionToolCXY,
            actionstep,
            toolbar_id=None,
            icon="rectangle.png",
            title="Step",
        )
        self.doortool = self.manager.add_tool(
            RectangularActionToolCX,
            actiondoor,
            toolbar_id=None,
            icon="rectangle.png",
            title="Door",
        )
        self.gaussdertool = self.manager.add_tool(
            RectangularActionToolCX,
            actiongaussder,
            toolbar_id=None,
            icon="rectangle.png",
            title="GaussDer",
        )
        self.lorentzdertool = self.manager.add_tool(
            RectangularActionToolCX,
            actionlorentzder,
            toolbar_id=None,
            icon="rectangle.png",
            title="LorentzDer",
        )
        self.gausslortool = self.manager.add_tool(
            RectangularActionToolCX,
            actiongausslor,
            toolbar_id=None,
            icon="rectangle.png",
            title="GaussLor",
        )
        self.skewtool = self.manager.add_tool(
            RectangularActionToolCX,
            actionskew,
            toolbar_id=None,
            icon="rectangle.png",
            title="Skew",
        )
        # print "g",self.gausstool
        # print "l",self.lorentztool
        # print "v",self.voigttool
        # print "s",self.steptool
        # print "d",self.doortool

        self.action.setMenu(menu)
        return menu

    def reset(self):
        plot = self.get_active_plot()
        self.manager.activate_default_tool()
        plot.tabval.reset()
        plot.tabval.extendedparams.setvalexp(plot.xexp, plot.yexp)
        plot.tabval.drawtry()

    def drawnewgauss(self):
        # print "drawnewgauss"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.gausstool.activate()

    def drawnewlorentz(self):
        # print "drawnewlorentz"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.lorentztool.activate()

    def drawnewvoigt(self):
        # print "drawnewvoigt"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.voigttool.activate()

    def drawnewstep(self):
        # print "drawnewstep"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.steptool.activate()

    def drawnewdoor(self):
        # print "drawnewdoor"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.doortool.activate()

    def drawnewgaussder(self):
        # print "drawnewgauss"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.gaussdertool.activate()

    def drawnewlorentzder(self):
        # print "drawnewgauss"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.lorentzdertool.activate()

    def drawnewgausslor(self):
        # print "drawnewgauss"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.gausslortool.activate()

    def drawnewskew(self):
        # print "drawnewgauss"
        plot = self.get_active_plot()
        plot.tabval.setactivecurve("new")
        self.skewtool.activate()

    def handle_shape(self, shape, inside):
        shape.set_style("plot", "shape/mask")
        shape.set_private(True)
        plot = self.get_active_plot()
        plot.set_active_item(shape)
        self._mask_shapes[plot] += [(shape, inside)]

    def deactivate(self):
        """Deactivate tools"""
        print("deactivate")
        self.gausscurve.deactivate()
        self.lorentzcurve.deactivate()
        self.voigtcurve.deactivate()
        self.stepcurve.deactivate()
        self.doorcurve.deactivate()
        self.gaussdercurve.deactivate()
        self.lorentzdercurve.deactivate()
        print("deactivate done")


class RunTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        """
        CommandTool.__init__(self, manager, _("Run"),icon=QIcon("Run.png"),
                                            tip=_("Perform fit"),
                                            toolbar_id=toolbar_id)
        """

        CommandTool.__init__(
            self,
            manager,
            _("Run"),
            icon=runpath,
            tip=_("Perform fit"),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        plot = self.get_active_plot()
        plot.tabval.startfit()
        # Activate tool
        pass


class YFullRangeTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(
            self,
            manager,
            _("Full Range"),
            icon=yfullrangepath,
            tip=_("Set Y Full Range"),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        plot = self.get_active_plot()
        vmin = np.amin(plot.tabval.extendedparams.yexp)
        vmax = np.amax(plot.tabval.extendedparams.yexp)
        plot.set_axis_limits("left", vmin, vmax)
        plot.replot()


class PrefTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(
            self,
            manager,
            _("Run"),
            icon="settings.png",
            tip=_("Preferences"),
            toolbar_id=toolbar_id,
        )
        self.manager = manager

    def create_action_menu(self, manager):
        # Create and return menu for the tool's action
        self.saveprefaction = QAction("Save as...", self)
        self.connect(
            self.saveprefaction, SIGNAL(_fromUtf8("triggered()")), self.savepref
        )

        self.updatestartaction = manager.create_action(
            _("Start with last fit"), toggled=self.updatestart
        )
        self.updatestartaction.setChecked(False)

        self.showtagsaction = QAction("Show tags", self)
        self.connect(
            self.showtagsaction, SIGNAL(_fromUtf8("triggered()")), self.showtags
        )

        self.scaleprefaction = QAction("Autoscale", self)
        self.scaleprefaction.setCheckable(True)
        self.scaleprefaction.setChecked(True)

        menu = QMenu()
        # add_actions(menu, (self.gausscurve.action,self.lorentzcurve.action,self.voigtcurve.action,self.stepcurve.action,self.doorcurve.action,resetaction))
        add_actions(
            menu,
            (
                self.saveprefaction,
                self.updatestartaction,
                self.showtagsaction,
                self.scaleprefaction,
            ),
        )

        self.action.setMenu(menu)
        return menu

    def savepref(self):
        # print "drawnewlorentz"
        plot = self.get_active_plot()
        plot.tabval.setfilename()

    def showtags(self):
        # print "drawnewlorentz"
        plot = self.get_active_plot()
        tags = plot.tabval.extendedparams.tags
        # print 'tags'
        TagsWindow(tags)

    def updatestart(self):
        # print "drawnewlorentz"
        plot = self.get_active_plot()
        if self.updatestartaction.isChecked():
            plot.tabval.updatestart = True
        else:
            plot.tabval.updatestart = False

    def setupdatestart(self, value):
        plot = self.get_active_plot()
        self.updatestartaction.setChecked(value)
        plot.tabval.updatestart = value


class SaveFitTool(CommandTool):
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        super(SaveFitTool, self).__init__(
            manager,
            _("Save fit result"),
            get_std_icon("DialogSaveButton", 16),
            toolbar_id=toolbar_id,
        )

    def activate_command(self, plot, checked):
        if plot.tabval.isfitted:
            plot.tabval.save_fit()


class RectangularSelectionHandlerCX(RectangularSelectionHandler):
    # on utilise la classe heritee dont on surcharge la methode move

    def move(self, filter, event):
        """methode surchargee par la classe """
        sympos = QPoint(2 * self.start.x() - event.pos().x(), self.start.y())
        self.shape.move_local_point_to(self.shape_h0, sympos)
        self.shape.move_local_point_to(self.shape_h1, event.pos())
        self.move_action(filter, event)
        filter.plot.replot()


class RectangularActionToolCX(RectangularActionTool):
    # outil de tracage de rectangles centre en x
    # on utilise la classe heritee dont on surcharge la methode setup_filter

    def setup_filter(
        self, baseplot
    ):  # partie utilisee pendant le mouvement a la souris
        # utilise a l'initialisation de la toolbar
        # print "setup_filter"
        filter = baseplot.filter
        start_state = filter.new_state()
        handler = RectangularSelectionHandlerCX(
            filter, Qt.LeftButton, start_state=start_state  # gestionnaire du filtre
        )
        shape, h0, h1 = self.get_shape()
        shape.pen.setColor(QColor("#00bfff"))

        handler.set_shape(
            shape, h0, h1, self.setup_shape, avoid_null_shape=self.AVOID_NULL_SHAPE
        )
        self.connect(handler, SIG_END_RECT, self.end_rect)
        # self.connect(handler, SIG_CLICK_EVENT, self.start)   #a definir aussi dans RectangularSelectionHandler2
        return setup_standard_tool_filter(filter, start_state)

    def activate(self):
        """Activate tool"""

        # print "commande active",self
        for baseplot, start_state in self.start_state.items():
            baseplot.filter.set_state(start_state, None)
        self.action.setChecked(True)
        self.manager.set_active_tool(self)
        # plot = self.get_active_plot()
        # plot.newcurve=True

    def deactivate(self):
        """Deactivate tool"""
        # print "commande desactivee",self
        self.action.setChecked(False)


class RectangularSelectionHandlerCXY(RectangularSelectionHandler):
    # on utilise la classe heritee dont on surcharge la methode move

    def move(self, filter, event):
        """methode surchargee par la classe """
        sympos = QPoint(
            2 * self.start.x() - event.pos().x(), 2 * self.start.y() - event.pos().y()
        )
        self.shape.move_local_point_to(self.shape_h0, sympos)
        self.shape.move_local_point_to(self.shape_h1, event.pos())
        self.move_action(filter, event)
        filter.plot.replot()


class RectangularActionToolCXY(RectangularActionTool):
    # outil de tracage de rectangles centre en x
    # on utilise la classe heritee dont on surcharge la methode setup_filter

    def setup_filter(
        self, baseplot
    ):  # partie utilisee pendant le mouvement a la souris
        # utilise a l'initialisation de la toolbar
        # print "setup_filter"
        filter = baseplot.filter
        start_state = filter.new_state()
        handler = RectangularSelectionHandlerCXY(
            filter, Qt.LeftButton, start_state=start_state  # gestionnaire du filtre
        )
        shape, h0, h1 = self.get_shape()
        shape.pen.setColor(QColor("#00bfff"))

        handler.set_shape(
            shape, h0, h1, self.setup_shape, avoid_null_shape=self.AVOID_NULL_SHAPE
        )
        self.connect(handler, SIG_END_RECT, self.end_rect)
        # self.connect(handler, SIG_CLICK_EVENT, self.start)   #a definir aussi dans RectangularSelectionHandler2
        return setup_standard_tool_filter(filter, start_state)

    def activate(self):
        """Activate tool"""

        # print "commande active",self
        for baseplot, start_state in self.start_state.items():
            baseplot.filter.set_state(start_state, None)
        self.action.setChecked(True)
        self.manager.set_active_tool(self)
        # plot = self.get_active_plot()
        # plot.newcurve=True

    def deactivate(self):
        """Deactivate tool"""
        # print "commande desactivee",self
        self.action.setChecked(False)


def get_xiwb_cx(plot, p0, p1):
    # return position, intensity,fwhm and background  from the rectangular tool cx
    ax, ay = plot.get_axis_id("bottom"), plot.get_axis_id("left")
    x1, y1 = plot.invTransform(ax, p0.x()), plot.invTransform(ay, p0.y())
    x2, y2 = plot.invTransform(ax, p1.x()), plot.invTransform(ay, p1.y())
    return x1, y1 - y2, 2.0 * (x2 - x1), y2


def get_xiwb_cxy(plot, p0, p1):
    # return position, intensity,width and background  from the rectangular tool cxy
    ax, ay = plot.get_axis_id("bottom"), plot.get_axis_id("left")
    x1, y1 = plot.invTransform(ax, p0.x()), plot.invTransform(ay, p0.y())
    x2, y2 = plot.invTransform(ax, p1.x()), plot.invTransform(ay, p1.y())
    return x1, 2 * (y1 - y2), 2.0 * (x2 - x1), y2


def actiongauss(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("gaussian", (itry, x0try, wtry), bg=btry)

    # lim=plot.get_axis_limits("bottom")
    # xtry=np.linspace(lim[0],lim[1], num=100)
    # ytry=gauss(xtry,[itry,x0try,wtry])+btry

    """
    #pour rajouter un polygone
    points=np.concatenate((xtry,ytry))
    points=points.reshape((2,100))
    points=np.transpose(points)
    testpolygon=PolygonShape(points=points)
    testpolygon.set_selectable(False)
    plot.add_item(testpolygon)
    """


def actionlorentz(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("lorentzian", (itry, x0try, wtry), bg=btry)


def actionvoigt(plot, p0, p1):
    # trace une pseudo-voigt avec poids egal sur les pics lorentz et gauss
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("voigt", (itry, x0try, wtry, 0.5), bg=btry)


def actionstep(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cxy(plot, p0, p1)
    plot.tabval.set_cv("step", (itry, x0try, wtry), bg=btry)


def actiondoor(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("door", (itry, x0try, wtry, wtry / 4.0), bg=btry)


def actiongaussder(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("gaussian_derivative", (itry, x0try, wtry), bg=btry + itry / 2.0)


def actiongausslor(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("gaussian_lorentzian", (itry, x0try, wtry, 0.1), bg=btry)


def actionlorentzder(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("gaussian_derivative", (itry, x0try, wtry), bg=btry + itry / 2.0)


def actionskew(plot, p0, p1):
    x0try, itry, wtry, btry = get_xiwb_cx(plot, p0, p1)
    plot.tabval.set_cv("skew", (itry, x0try, wtry, 0.0), bg=btry)


class Ui_FitWindow(QMainWindow):
    def setupUi(self):
        self.setObjectName(_fromUtf8("MainWindow"))

        self.setWindowTitle("Fit window")

        self.savename = None

        # on utilise un splitter pour separer la fenetre en deux, ce sera le central widget
        self.splitter = QSplitter(Qt.Vertical)

        # widget pour afficher les courbes
        self.cv = CurveWidget(parent=self, show_itemlist=True)

        # widget pour afficher les donnees du fit , gerer les entrees de parametres
        self.tabval = FitTable(parent=self)

        self.splitter.addWidget(self.cv)
        self.splitter.addWidget(self.tabval)

        self.setCentralWidget(self.splitter)

        # on utilise le plot manager de guiqwt
        self.manager = PlotManager(self)
        self.plot = self.cv.get_plot()

        # on associe les references croisees au tableau des donnees et plot
        self.plot.tabval = self.tabval
        self.tabval.plot = self.plot

        # attribut pour dire qu'on a demande une nouvelle courbe qui n'est pas encore ajustee a la souris
        self.plot.newcurve = False

        # default empty curves
        z = np.empty((0,))
        self.empty = True
        self.plot.curveexp = make.curve(
            z,
            z,
            marker="Diamond",
            markerfacecolor="r",
            markeredgecolor="r",
            markersize=4,
            linestyle="NoPen",
            title="experiment",
        )
        self.plot.curvetry = make.curve(z, z, color="g", title="estimate")
        self.plot.curveopt = make.curve(z, z, color="k", title="fit result")

        self.plot.curveexp.set_readonly(True)
        self.plot.curvetry.set_readonly(True)
        self.plot.curveopt.set_readonly(True)

        self.plot.add_item(self.plot.curveexp)
        self.plot.add_item(self.plot.curvetry)
        self.plot.add_item(self.plot.curveopt)

        self.plot.hide_items(
            (self.plot.curveexp, self.plot.curvetry, self.plot.curveopt)
        )

        self.manager.add_plot(self.plot)

        self.connect(self.plot, SIG_RANGE_CHANGED, self.tabval.rangemove)
        self.connect(self.plot, SIG_ITEM_REMOVED, self.tabval.rangeremove)
        # self.connect(self.tabval, SIG_FIT_DONE, self.coucou)

        # ---Add toolbar and register manager tools
        toolbar = self.addToolBar("tools")

        self.manager.add_toolbar(toolbar, id(toolbar))

        # self.manager.register_all_curve_tools()
        self.manager.register_standard_tools()
        self.manager.register_other_tools()

        self.oft = self.manager.add_tool(OpenFileTool)
        self.oft.formats = " *.txt *.dat"
        self.title = "Open data file"
        self.oft.connect(self.oft, SIGNAL("openfile(QString*)"), self.open_file)

        self.FitTool = self.manager.add_tool(FitTool)  # ok
        self.MaskTool = self.manager.add_tool(MaskTool)
        self.manager.add_tool(RunTool)  # ok
        self.manager.add_tool(SaveFitTool)
        self.manager.add_tool(YFullRangeTool)
        self.Pref_Tool = self.manager.add_tool(PrefTool)
        self.Pref_Tool.setupdatestart(self.tabval.updatestart)
        # redimensionnement de la fenetre principale, les widgets suivent
        # sinon, prennent leur SizeHint (300,400) pour cv par exemple
        self.resize(820, 600)
        self.readfileparams = readfileparams()
        # pour savoir si un fit a ete fait et sauve

    def coucou(self):
        print("coucou")

    def reset(self):
        self.FitTool.reset()

        # testpolygon=PolygonShape(points=[[1,0],[1.1,0.1],[1.2,0.5],[1.4,2]])
        # self.plot.add_item(testpolygon)

    def open_file(self, QString):

        fname = str(QString)
        shortname = path.split(fname)[1]
        fic = open(fname)
        ficlines = fic.readlines()
        readfilewindow(self.readfileparams, ficlines)
        seps = [" ", "\t", ","]
        sep = seps[self.readfileparams.delimiter]
        nf = len(ficlines)
        ix = self.readfileparams.x - 1
        iy = self.readfileparams.y - 1

        if self.readfileparams.title:
            try:
                ni = self.readfileparams.heading + 1
                title = ficlines[self.readfileparams.heading].split(sep)
                print(title)
                xlabel = title[ix]
                ylabel = title[iy]
            except:
                ni = self.readfileparams.heading
                xlabel = "x"
                ylabel = "y"
        else:
            ni = self.readfileparams.heading
            xlabel = "x"
            ylabel = "y"

        x = []
        y = []
        for i in range(ni, nf):
            try:
                ll = ficlines[i].split(sep)
                xx = float(ll[ix])
                yy = float(ll[iy])
                if np.isfinite(xx) and np.isfinite(yy):
                    x.append(xx)
                    y.append(yy)
            except:
                pass

        self.tabval.cwd = path.dirname(path.realpath(fname))
        self.setvalexp(x, y, xlabel=xlabel, ylabel=ylabel, title=shortname)

    def settags(self, tags, saveint=None):
        # tags est une liste de couples de valeurs transmis par l'application qui sollicite grafit pour
        # etre enregistre dans le fichier de sauvegarde

        self.tabval.extendedparams.updatetags(tags)

        # soit saveint n'est pas modifie et rien ne change, soit saveint l'est et on change
        if saveint is not None:
            self.saveint = saveint
            self.tabval.saveint = saveint

            if self.saveint:
                print("on sauve les integrales")
            else:
                print("on ne sauve pas les integrales")

    def setupdatestart(self, value):
        self.tabval.updatestart = value
        self.Pref_Tool.setupdatestart(self.tabval.updatestart)

    def getfitresult(self):
        return [
            self.tabval.extendedparams.paropt,
            integral(
                self.tabval.extendedparams.paropt, self.tabval.extendedparams.curveslist
            ),
        ]

    def setsaveint(self, value=True):
        self.tabval.saveint = value

    def setsavedir(self, cwd):
        self.tabval.cwd = cwd

    def setvalexp(
        self,
        xexp,
        yexp,
        xlabel="x",
        ylabel="y",
        title="scan",
        xlog=False,
        ylog=False,
        tags=[],
        followscans=False,
    ):
        # title est le titre du scan qui est fitte. xlog et ylog determinent si l'echelle est log ou pas
        # infos est une liste de deux strings qui sont enregistrees en plus des parametres du fit
        # lors de la sauvegarde des donnes. Gere par le programme qui appelle grafit.
        i = 1

        if self.tabval.isfitted and self.tabval.issaved is False:
            # avant d'ecraser le fit precedent on demande s'il faut le sauver
            # pose probleme dans le cas ou setvalexp est genere par le deplacement d'une shape
            i = QMessageBox.question(
                self, "save", "do you want to save fit?", "Yes", "No", "Cancel"
            )

        if i == 0:
            self.tabval.save_fit()

        # s'il existe deja un fit, et que followscans=True, alors on decale les valeurs de fit
        # de façon a ce que la position de guess des pics se retrouve au meme endroit par
        # rapport au centre du scan
        if followscans and not self.empty:

            cen0 = (max(self.plot.xexp) + min(self.plot.xexp)) / 2.0
            cen1 = (max(xexp) + min(xexp)) / 2.0
            shift = cen1 - cen0
            print("on shifte de ", shift)
            self.tabval.shiftguess(shift)

        if i < 2:  # si i=2, on ne fait rien

            if self.Pref_Tool.scaleprefaction.isChecked():
                if xlog:
                    self.plot.set_axis_scale("bottom", "log", autoscale=True)
                else:
                    self.plot.set_axis_scale("bottom", "lin", autoscale=True)
                if ylog:
                    self.plot.set_axis_scale("left", "log", autoscale=True)
                else:
                    self.plot.set_axis_scale("left", "lin", autoscale=True)

            self.plot.xexp = xexp
            self.plot.yexp = yexp
            self.tabval.scantitle = title
            self.tabval.extendedparams.setvalexp(xexp, yexp)
            self.settags(tags)

            self.plot.set_titles(title=title, xlabel=xlabel, ylabel=ylabel)
            self.plot.curveexp.set_data(xexp, yexp)

            self.plot.show_items((self.plot.curveexp,))
            self.plot.hide_items((self.plot.curvetry, self.plot.curveopt))

            self.tabval.isfitted = False  # on n'a pas fait de fit sur ce set de donnees

            self.tabval.setrange()
            self.plot.do_autoscale()
            self.empty = False

        # print xexp.shape,yexp.shape


if __name__ == "__main__":

    app = QApplication(sys.argv)

    ui = Ui_FitWindow()
    ui.setupUi()
    ui.xlabel = "x"
    ui.ylabel = "y"
    x = np.arange(0, 3.14, 0.001)
    y = np.sin(x)
    ui.setvalexp(x, y)
    ui.show()

    sys.exit(app.exec_())
