"""
binoculars gui for data processing
Created on 2015-06-04
author: Remy Nencib (remy.nencib@esrf.r)
"""

import sys
import os
import time

from PyQt5.Qt import (Qt)  # noqa
from PyQt5.QtGui import (QColor, QPalette)
from PyQt5.QtWidgets import (QAction, QApplication, QTabWidget,
                             QFileDialog, QMessageBox,
                             QPushButton, QHBoxLayout,
                             QVBoxLayout, QSplitter, QTableWidgetItem, QTableWidget,
                             QLabel, QLineEdit, QMainWindow, QWidget, QComboBox,
                             QProgressDialog, QDockWidget)

import binoculars.main
import binoculars.util


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        # add the close button for tabs
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

    #method for close tabs
    def close_tab(self, tab):
        self.tab_widget.removeTab(tab)

    def initUI(self):
        #we create the menu bar
        openfile = QAction('Open', self)
        openfile.setShortcut('Ctrl+O')
        openfile.setStatusTip('Open new File')
        openfile.triggered.connect(self.ShowFile)

        savefile = QAction('Save', self)
        savefile.setShortcut('Ctrl+S')
        savefile.setStatusTip('Save File')
        savefile.triggered.connect(self.Save)

        create = QAction('Create', self)
        create.setStatusTip('Create Configfile')
        create.triggered.connect(self.New_Config)

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(openfile)
        filemenu.addAction(savefile)
        filemenu = menubar.addMenu('&New Configfile')
        filemenu.addAction(create)

        #we configue the main windows
        palette = QPalette()
        palette.setColor(QPalette.Background, Qt.gray)
        self.setPalette(palette)
        self.setGeometry(50, 100, 700, 700)
        self.setWindowTitle('Binoculars processgui')
        self.show()

        self.ListCommand = QTableWidget(1, 2, self)
        self.ListCommand.verticalHeader().setVisible(True)
        self.ListCommand.horizontalHeader().setVisible(False)
        self.ListCommand.horizontalHeader().stretchSectionCount()
        self.ListCommand.setColumnWidth(0, 80)
        self.ListCommand.setColumnWidth(1, 80)
        self.ListCommand.setRowCount(0)
        self.buttonDelete = QPushButton('Delete', self)
        self.buttonDelete.clicked.connect(self.removeConf)
        self.process = QPushButton('run', self)
        self.process.setStyleSheet("background-color: darkred")
        self.process.clicked.connect(self.run)

        self.wid = QWidget()
        self.CommandLayout = QVBoxLayout()
        self.CommandLayout.addWidget(self.ListCommand)
        self.CommandLayout.addWidget(self.process)
        self.CommandLayout.addWidget(self.buttonDelete)
        self.wid.setLayout(self.CommandLayout)

        self.Dock = QDockWidget()
        self.Dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.Dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.Dock.setWidget(self.wid)
        self.Dock.setMaximumWidth(200)
        self.Dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea(1), self.Dock)

    def removeConf(self):
        self.ListCommand.removeRow(self.ListCommand.currentRow())

    def Add_To_Liste(self, xxx_todo_changeme):
        (command, cfg) = xxx_todo_changeme
        row = self.ListCommand.rowCount()
        index = self.tab_widget.currentIndex()
        filename = self.tab_widget.tabText(index)
        self.ListCommand.insertRow(self.ListCommand.rowCount())
        dic = {filename: cfg}
        self.item1 = QTableWidgetItem(str(command))
        self.item1.command = command
        self.item2 = QTableWidgetItem(str(filename))
        self.item2.cfg = dic[filename]
        self.ListCommand.setItem(row, 0, self.item1)
        self.ListCommand.setItem(row, 1, self.item2)

    #We run the script and create a hdf5 file
    def run(self):
        maximum = self.ListCommand.rowCount()
        pd = QProgressDialog('running', 'Cancel', 0, maximum, self)
        pd.setWindowModality(Qt.WindowModal)
        pd.show()

        def progress(cfg, command):
            if pd.wasCanceled():
                raise KeyboardInterrupt
            QApplication.processEvents()
            return binoculars.main.Main.from_object(cfg, command)
        try:
            for index in range(self.ListCommand.rowCount()):
                pd.setValue(index)
                cfg = self.ListCommand.item(index, 1).cfg
                command = self.ListCommand.item(index, 0).command
                print(cfg)
                progress(cfg, command)
            self.ListCommand.clear()
            self.ListCommand.setRowCount(0)
        except BaseException as e:
                #cfg = self.ListCommand.item(index,1).cfg
                #print cfg
                QMessageBox.about(self, "Error", f"There was an error processing one of the scans: {e}")

        finally:
                pd.close()

    #we call the load function
    def ShowFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '')
        confwidget = Conf_Tab(self)
        confwidget.read_data(str(filename))
        newIndex = self.tab_widget.addTab(confwidget, os.path.basename(str(filename)))
        confwidget.command.connect(self.Add_To_Liste)
        self.tab_widget.setCurrentIndex(newIndex)

    #we call the save function
    def Save(self):
        filename, _ = QFileDialog().getSaveFileName(self, 'Save', '', '*.txt')
        widget = self.tab_widget.currentWidget()
        widget.save(filename)

    #we call the new tab conf
    def New_Config(self):
        widget = Conf_Tab(self)
        self.tab_widget.addTab(widget, 'New configfile')
        widget.command.connect(self.Add_To_Liste)

#----------------------------------------------------------------------------------------------------
#-----------------------------------------CREATE TABLE-----------------------------------------------


class Table(QWidget):
    def __init__(self, label, parent=None):
        super().__init__()

        # create a QTableWidget
        self.table = QTableWidget(1, 2, self)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Comment'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setTextElideMode(Qt.ElideLeft)
        #create combobox
        self.combobox = QComboBox()
        #add items
        self.cell = QTableWidgetItem("type")
        self.table.setItem(0, 0, self.cell)
        self.table.setCellWidget(0, 1, self.combobox)
        #we create pushbuttons and we call the method when we clic on
        self.btn_add_row = QPushButton('+', self)
        self.btn_add_row.clicked.connect(self.add_row)
        self.buttonRemove = QPushButton('-', self)
        self.buttonRemove.clicked.connect(self.remove)
        #the dispositon of the table and the butttons

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        hbox.addWidget(self.btn_add_row)
        hbox.addWidget(self.buttonRemove)

        vbox.addWidget(label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.table)
        self.setLayout(vbox)

    def add_row(self):
        self.table.insertRow(self.table.rowCount())

    def remove(self):
        self.table.removeRow(self.table.currentRow())

    def get_keys(self):
        return list(str(self.table.item(index, 0).text()) for index in range(self.table.rowCount()))

    #Here we take all values from tables
    def getParam(self):
        for index in range(self.table.rowCount()):
            if self.table.item is not None:
                key = str(self.table.item(index, 0).text())
                comment = str(self.table.item(index, 0).toolTip())
                if index == 0:
                    yield key, str(self.table.cellWidget(index, 1).currentText()), comment
                elif self.table.item(index, 1):
                    if len(str(self.table.item(index, 1).text())) != 0 and self.table.item(index, 0).textColor() == QColor('black'):
                        yield key, str(self.table.item(index, 1).text()), comment

    #Here we put all values in tables
    def addData(self, cfg):
        for item in cfg:
            if item == 'type':
                box = self.table.cellWidget(0, 1)
                value = cfg[item].split(':')
                if len(value) > 1:
                    box.setCurrentIndex(box.findText(value[1], Qt.MatchFixedString))
                else:
                    box.setCurrentIndex(box.findText(cfg[item], Qt.MatchFixedString))
            elif item not in self.get_keys():
                self.add_row()
                row = self.table.rowCount()
                for col in range(self.table.columnCount()):
                    if col == 0:
                        newitem = QTableWidgetItem(item)
                        self.table.setItem(row - 1, col, newitem)
                    if col == 1:
                        newitem2 = QTableWidgetItem(cfg[item])
                        self.table.setItem(row - 1, col, newitem2)
            else:
                index = self.get_keys().index(item)
                self.table.item(index, 1).setText(cfg[item])

    def addDataConf(self, options):
        keys = self.get_keys()
        newconfigs = {option[0]: '' for option in options if option[0] not in keys}
        self.addData(newconfigs)

        names = list(option[0] for option in options)

        for index, key in enumerate(self.get_keys()):
            if str(key) in names:
                self.table.item(index, 0).setTextColor(QColor('black'))
                self.table.item(index, 0).setToolTip(options[names.index(key)][1])
            elif str(key) == 'type':
                self.table.item(index, 0).setTextColor(QColor('black'))
            else:
                self.table.item(index, 0).setTextColor(QColor('gray'))

    def add_to_combo(self, items):
        self.combobox.clear()
        self.combobox.addItems(items)

#----------------------------------------------------------------------------------------------------
#-----------------------------------------CREATE CONFIG----------------------------------------------


class Conf_Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #we create 3 tables
        self.Dis = Table(QLabel('<strong>Dispatcher :</strong>'))
        self.Inp = Table(QLabel('<strong>Input :</strong>'))
        self.Pro = Table(QLabel('<strong>Projection :<strong>'))
        self.select = QComboBox()
        backends = list(backend.lower() for backend in binoculars.util.get_backends())
        #we add the list of different backends on the select combobox
        self.select.addItems(backends)
        self.add = QPushButton('add')
        self.add.clicked.connect(self.AddCommand)
        self.scan = QLineEdit()
        self.scan.setToolTip('scan selection example: 820 824')

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.Dis)
        splitter.addWidget(self.Inp)
        splitter.addWidget(self.Pro)
        hbox.addWidget(splitter)

        commandbox = QHBoxLayout()
        commandbox.addWidget(self.add)
        commandbox.addWidget(self.scan)

        vbox.addWidget(self.select)
        vbox.addLayout(hbox)
        vbox.addLayout(commandbox)

        #the dispositon of all elements of the gui
        #Layout = QGridLayout()
        #Layout.addWidget(label1,1,1,1,2)
        #Layout.addWidget(label2,1,0,1,2)
        #Layout.addWidget(label3,1,2,1,2)
        #Layout.addWidget(self.select,0,0)
        #Layout.addWidget(self.Dis,2,1)
        #Layout.addWidget(self.Inp,2,0)
        #Layout.addWidget(self.Pro,2,2)
        #Layout.addWidget(self.add,3,0)
        #Layout.addWidget(self.scan,3,1)
        self.setLayout(vbox)

        #Here we call all methods for selected an ellement on differents combobox
        self.Dis.add_to_combo(binoculars.util.get_dispatchers())
        self.select.activated['QString'].connect(self.DataCombo)
        self.Inp.combobox.activated.connect(self.DataTableInp)
        self.Pro.combobox.activated.connect(self.DataTableInpPro)
        self.Dis.combobox.activated.connect(self.DataTableInpDis)

    def DataCombo(self, text):
        self.Inp.add_to_combo(binoculars.util.get_inputs(str(text)))
        self.Pro.add_to_combo(binoculars.util.get_projections(str(text)))
        self.DataTableInp()
        self.DataTableInpPro()
        self.DataTableInpDis()

    def DataTableInp(self):
        backend = str(self.select.currentText())
        inp = binoculars.util.get_input_configkeys(backend, str(self.Inp.combobox.currentText()))
        self.Inp.addDataConf(inp)

    def DataTableInpPro(self):
        backend = str(self.select.currentText())
        proj = binoculars.util.get_projection_configkeys(backend, str(self.Pro.combobox.currentText()))
        self.Pro.addDataConf(proj)

    def DataTableInpDis(self):
        disp = binoculars.util.get_dispatcher_configkeys(str(self.Dis.combobox.currentText()))
        self.Dis.addDataConf(disp)

    #The save method we take all ellements on tables and we put them in this format {0} = {1} #{2}
    def save(self, filename):
        with open(filename, 'w') as fp:
            fp.write('[dispatcher]\n')
            # cycles over the iterator object
            for key, value, comment in self.Dis.getParam():
                fp.write(f'{key} = {value} #{comment}\n')
            fp.write('[input]\n')
            for key, value, comment in self.Inp.getParam():
                if key == 'type':
                    value = f'{self.select.currentText()}:{value}'
                fp.write(f'{key} = {value} #{comment}\n')
            fp.write('[projection]\n')
            for key, value, comment in self.Pro.getParam():
                if key == 'type':
                    value = f'{self.select.currentText()}:{value}'
                fp.write(f'{key} = {value} #{comment}\n')

    #This method take the name of objects and values for run the script
    def get_configobj(self):

        inInp = {}
        inDis = {}
        inPro = {}

        inDis = {key: value for key, value, comment in self.Dis.getParam()}

        for key, value, comment in self.Inp.getParam():
            if key == 'type':
                value = f'{str(self.select.currentText()).strip()}:{value}'
            inInp[key] = value

        for key, value, comment in self.Pro.getParam():
            if key == 'type':
                value = f'{str(self.select.currentText()).strip()}:{value}'
            inPro[key] = value

        cfg = binoculars.util.ConfigFile('processgui {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
        setattr(cfg, 'input', inInp)
        setattr(cfg, 'dispatcher', inDis)
        setattr(cfg, 'projection', inPro)
        return cfg

    #This method take elements on a text file or the binocular script and put them on tables
    def read_data(self, filename):
        cfg = binoculars.util.ConfigFile.fromtxtfile(str(filename))
        input_type = cfg.input['type']
        backend, value = input_type.strip(' ').split(':')
        self.select.setCurrentIndex(self.select.findText(backend, Qt.MatchFixedString))
        self.DataCombo(backend)
        self.Dis.addData(cfg.dispatcher)
        self.Inp.addData(cfg.input)
        self.Pro.addData(cfg.projection)

    #we add command on the DockWidget
    def AddCommand(self):
        scan = [str(self.scan.text())]
        cfg = self.get_configobj()
        commandconfig = (scan, cfg)
        self.command.emit(commandconfig)

def main():
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
