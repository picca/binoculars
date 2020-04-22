# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:13:29 2020

@author: Prevot
"""
import numpy as np

from guidata.configtools import get_icon
from guidata.qt.QtGui import QAction,QColor,QMenu,QKeySequence,QKeyEvent
from guidata.qt.QtCore import QPoint,Qt,QEvent
from guidata.qthelpers import add_actions,get_std_icon

from guiqwt.config import _
from guiqwt.events import QtDragHandler,RectangularSelectionHandler,setup_standard_tool_filter,EventMatch,KeyEventMatch
from guiqwt.signals import SIG_VALIDATE_TOOL, SIG_END_RECT, SIG_START_TRACKING, SIG_MOVE, SIG_STOP_NOT_MOVING, SIG_STOP_MOVING
from guiqwt.tools import CommandTool,DefaultToolbarID,RectangularActionTool,InteractiveTool,ToggleTool
from guiqwt.shapes import EllipseShape,PolygonShape
from guiqwt.baseplot import canvas_to_axes
from grafit.signals import SIG_TRIGGERED,SIG_MODEL_ADDED,SIG_KEY_PRESSED_EVENT

SHAPE_Z_OFFSET = 1000


class KeyMatch(EventMatch):
    """
    A callable returning True if it matches a key event
    keysequence:  integer
    """
    def __init__(self, keysequence):
        super(KeyMatch, self).__init__()
        assert isinstance(keysequence, int)
        self.keyseq = keysequence

    def get_event_types(self):
        return frozenset((QEvent.KeyPress,))

    def __call__(self, event):
        return event.type() == QEvent.KeyPress and event.key()==self.keyseq

class ShowWeightsTool(ToggleTool):
    def __init__(self, manager, title="show weights",icon = "weight.png" ,
                 tip="show weights associated to the data", toolbar_id=DefaultToolbarID):
        super(ShowWeightsTool,self).__init__(manager, title, icon, toolbar_id=toolbar_id)
                 
    def activate(self, checked=True):
        self.emit(SIG_VALIDATE_TOOL)
  
class RunTool(CommandTool):
    def __init__(self, manager, title="Run",
                 icon="apply.png", tip="run 2D fit", toolbar_id=DefaultToolbarID):
        super(RunTool,self).__init__(manager, title, icon, toolbar_id=toolbar_id)
                                        
    def setup_context_menu(self, menu, plot):
        pass
      
    def activate(self, checked=True):
        self.emit(SIG_VALIDATE_TOOL)
        
class SaveFitTool(CommandTool):  
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        super(SaveFitTool,self).__init__(manager, _("Save fit result"),
                                        get_std_icon("DialogSaveButton", 16),
                                        toolbar_id=toolbar_id)
    def activate(self, checked=True):
        self.emit(SIG_VALIDATE_TOOL)
      
class PrefTool(CommandTool):  
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        CommandTool.__init__(self, manager, _("Run"),icon="settings.png",
                                            tip=_("Preferences"),
                                            toolbar_id=toolbar_id)
        self.manager=manager

    def create_action_menu(self, manager):
        #Create and return menu for the tool's action
        self.saveprefaction = QAction("Save as...",self)
        self.showtagsaction = QAction("Show tags",self)
        self.scaleprefaction = QAction("Autoscale",self)
        self.scaleprefaction.setCheckable(True)
        self.scaleprefaction.setChecked(True)
        self.restartaction = QAction("Restart from last parameters saved",self)
        self.restartaction.setCheckable(True)
        self.restartaction.setChecked(True)
        self.samescaleaction = QAction("Same scale for data and model images",self)
        self.samescaleaction.setCheckable(True)
        self.samescaleaction.setChecked(True)
        self.showdiffaction = QAction("Show difference profiles",self)
        self.showdiffaction.setCheckable(True)
        self.showdiffaction.setChecked(True)
        
        
        menu = QMenu()
        add_actions(menu, (self.saveprefaction,self.showtagsaction,self.scaleprefaction,self.restartaction,self.samescaleaction,self.showdiffaction))
                
        self.action.setMenu(menu)
        return menu

class EllipseSelectionHandlerCXY(RectangularSelectionHandler):
    #on utilise la classe heritee dont on surcharge la methode move

    def __init__(self, filter, btn, mods=Qt.NoModifier, start_state=0):
        super(EllipseSelectionHandlerCXY, self).__init__(filter=filter, btn=btn, mods=mods, start_state=start_state)
        filter.add_event(start_state, KeyEventMatch((42,43,45,47)),
                         self.key_press, start_state)
                                          
    def move(self, filter, event):
        """methode surchargee par la classe """
        x1,y1=canvas_to_axes(self.shape, event.pos())
        x0,y0=canvas_to_axes(self.shape, self.start)
        dx=x1-x0
        dy=y1-y0
        self.shape.points=np.array([[x0-dx,y0],[x0+dx,y0],[x0,y0-dy],[x0,y0+dy]])
        self.move_action(filter, event)
        filter.plot.replot()
    
    def key_press(self,filter,event):
        self.emit(SIG_KEY_PRESSED_EVENT, filter, event.key())   
        
    def set_shape(self, shape, h0, h1, h2, h3, 
                  setup_shape_cb=None, avoid_null_shape=False):
        self.shape = shape
        self.shape_h0 = h0
        self.shape_h1 = h1
        self.shape_h2 = h2
        self.shape_h3 = h3
        self.setup_shape_cb = setup_shape_cb
        self.avoid_null_shape = avoid_null_shape
    

class RectangularSelectionHandlerCXY(RectangularSelectionHandler):
    #on utilise la classe heritee dont on surcharge la methode move
    
    def move(self, filter, event):
        """methode surchargee par la classe """
        sympos=QPoint(2*self.start.x()-event.pos().x(),2*self.start.y()-event.pos().y())
        print self.shape_h0,self.shape_h1
        self.shape.move_local_point_to(self.shape_h0, sympos)        
        self.shape.move_local_point_to(self.shape_h1, event.pos())
        self.move_action(filter, event)
        filter.plot.replot()




class RectangularActionToolCXY(RectangularActionTool):
    #outil de tracage de rectangles centre en x
    #on utilise la classe heritee dont on surcharge la methode setup_filter

    def setup_filter(self, baseplot):      #partie utilisee pendant le mouvement a la souris
        #utilise a l'initialisation de la toolbar
        #print "setup_filter"
        filter = baseplot.filter
        start_state = filter.new_state()
        handler = RectangularSelectionHandlerCXY(filter, Qt.LeftButton,      #gestionnaire du filtre
                                              start_state=start_state)
        shape, h0, h1 = self.get_shape()
        shape.pen.setColor(QColor("#00bfff"))

        handler.set_shape(shape, h0, h1, self.setup_shape,
                          avoid_null_shape=self.AVOID_NULL_SHAPE)
        self.connect(handler, SIG_END_RECT, self.end_rect)
        #self.connect(handler, SIG_CLICK_EVENT, self.start)   #a definir aussi dans RectangularSelectionHandler2 
        return setup_standard_tool_filter(filter, start_state)

    def activate(self):
        """Activate tool"""
        
        #print "commande active",self
        for baseplot, start_state in self.start_state.items():
            baseplot.filter.set_state(start_state, None)
        self.action.setChecked(True)
        self.manager.set_active_tool(self)
        #plot = self.get_active_plot()
        #plot.newcurve=True
        
    def deactivate(self):
        """Deactivate tool"""
        #print "commande desactivee",self
        self.action.setChecked(False)
        
        
        
class CircularActionToolCXY(RectangularActionToolCXY):
    TITLE = _("Circle")
    ICON = "circle.png"
    
    def __init__(self, manager, func1, func2, shape_style=None,
                 toolbar_id=DefaultToolbarID, title=None, icon=None, tip=None,
                 fix_orientation=False, switch_to_default_tool=None):

        self.key_func = func2  #function for keys
        super(CircularActionToolCXY, self).__init__(manager, func1, shape_style=shape_style, toolbar_id=toolbar_id,
                                 title=title, icon=icon, tip=tip, fix_orientation=fix_orientation,
                                 switch_to_default_tool=switch_to_default_tool)  

                                 
    def setup_filter(self, baseplot):      #partie utilisee pendant le mouvement a la souris
        #utilise a l'initialisation de la toolbar
        #print "setup_filter"
        filter = baseplot.filter
        start_state = filter.new_state()
        handler = EllipseSelectionHandlerCXY(filter, Qt.LeftButton,      #gestionnaire du filtre
                                              start_state=start_state)
        shape, h0, h1, h2, h3 = self.get_shape()
        shape.pen.setColor(QColor("#00bfff"))

        handler.set_shape(shape, h0, h1, h2, h3, self.setup_shape,
                          avoid_null_shape=self.AVOID_NULL_SHAPE)
        self.connect(handler, SIG_END_RECT, self.end_rect)
        self.connect(handler, SIG_KEY_PRESSED_EVENT, self.key_pressed)   
        return setup_standard_tool_filter(filter, start_state)
    
    def key_pressed(self, filter, key):
        plot = filter.plot
        self.key_func(plot,key)


    def get_shape(self):
        """Reimplemented RectangularActionTool method"""
        shape, h0, h1, h2, h3 = self.create_shape()
        self.setup_shape(shape)
        return shape, h0, h1, h2, h3

    def create_shape(self):
        shape = EllipseShape(0, 0, 1, 1)
        self.set_shape_style(shape)
        shape.switch_to_ellipse()
        return shape, 0, 1, 2, 3  #give the values of h0,h1,h2,h3 use to move the shape 


class MultiPointTool(InteractiveTool):
    TITLE = _("Polyline")
    ICON = "polyline.png"
    CURSOR = Qt.ArrowCursor

    def __init__(self, manager, func, handle_final_shape_cb=None, shape_style=None,
                 toolbar_id=DefaultToolbarID, title=None, icon=None, tip=None,
                 switch_to_default_tool=None):
        super(MultiPointTool, self).__init__(manager, toolbar_id,
                                title=title, icon=icon, tip=tip,
                                switch_to_default_tool=switch_to_default_tool)
        self.handle_final_shape_cb = handle_final_shape_cb
        self.shape = None
        self.current_handle = None
        self.init_pos = None
        self.move_func = func
        if shape_style is not None:
            self.shape_style_sect = shape_style[0]
            self.shape_style_key = shape_style[1]
        else:
            self.shape_style_sect = "plot"
            self.shape_style_key = "shape/drag"

    def reset(self):
        self.shape = None
        self.current_handle = None
    
    def create_shape(self, filter, pt):
        self.shape = PolygonShape(closed=False)
        filter.plot.add_item_with_z_offset(self.shape, SHAPE_Z_OFFSET)
        self.shape.setVisible(True)
        self.shape.set_style(self.shape_style_sect, self.shape_style_key)
        self.shape.add_local_point(pt)
        return self.shape.add_local_point(pt)
    
    def setup_filter(self, baseplot):
        filter = baseplot.filter
        # Initialisation du filtre
        start_state = filter.new_state()
        # Bouton gauche :
        handler = QtDragHandler(filter, Qt.LeftButton, start_state=start_state)
        filter.add_event(start_state,
                         KeyEventMatch( (Qt.Key_Enter, Qt.Key_Return,
                                         Qt.Key_Space) ),
                         self.validate, start_state)
        filter.add_event(start_state,
                         KeyEventMatch( (Qt.Key_Backspace,Qt.Key_Escape,) ),
                         self.cancel_point, start_state)
        self.connect(handler, SIG_START_TRACKING, self.mouse_press)
        self.connect(handler, SIG_MOVE, self.move)
        self.connect(handler, SIG_STOP_NOT_MOVING, self.mouse_release)
        self.connect(handler, SIG_STOP_MOVING, self.mouse_release)
        return setup_standard_tool_filter(filter, start_state)

    def validate(self, filter, event):
        super(MultiPointTool, self).validate(filter, event)
        """
        if self.handle_final_shape_cb is not None:
            self.handle_final_shape_cb(self.shape)
        """
        self.shape.detach()
        filter.plot.replot()
        self.reset()

    def cancel_point(self, filter, event):
        if self.shape is None:
            return
        points = self.shape.get_points()
        if points is None:
            return
        elif len(points) <= 2:
            filter.plot.del_item(self.shape)
            self.reset()
        else:
            if self.current_handle:
                newh = self.shape.del_point(self.current_handle)
            else:
                newh = self.shape.del_point(-1)
            self.current_handle = newh
        filter.plot.replot()

    def mouse_press(self, filter, event):
        """We create a new shape if it's the first point
        otherwise we add a new point
        """
        if self.shape is None:
            self.init_pos = event.pos()
            self.current_handle = self.create_shape(filter, event.pos())
            filter.plot.replot()
        else:
            self.current_handle = self.shape.add_local_point(event.pos())
            if self.current_handle>2:
                self.shape.del_point(0)
                self.current_handle = 2

    def move(self, filter, event):
        """moving while holding the button down lets the user
        position the last created point
        """
        if self.shape is None or self.current_handle is None:
            # Error ??
            return
        self.shape.move_local_point_to(self.current_handle, event.pos())
        self.move_func(filter.plot,self.shape.points)
        filter.plot.replot()

    def mouse_release(self, filter, event):
        """Releasing the mouse button validate the last point position"""
        if self.current_handle is None:
            return
        if self.init_pos is not None and self.init_pos == event.pos():
            self.shape.del_point(-1)
        else:
            self.shape.move_local_point_to(self.current_handle, event.pos())
        self.init_pos = None
        self.current_handle = None
        self.move_func(filter.plot,self.shape.points)
        filter.plot.replot()

    def deactivate(self):
        """Deactivate tool"""
        if self.shape is not None:
            self.shape.detach()
            self.get_active_plot().replot()
            self.reset()
        self.action.setChecked(False)







class AddModelTool(CommandTool):
    
    def __init__(self, manager, list_of_models, toolbar_id=DefaultToolbarID):
        self.list_of_models=list_of_models
        CommandTool.__init__(self, manager, _("Fit"),icon= get_icon("edit_add.png"),
                                            tip=_("Add curve for fit"),
                                            toolbar_id=toolbar_id)
        self.manager=manager
        
    def create_action_menu(self, manager):
        #Create and return menu for the tool's action
        menu = QMenu()
        for i,modelclass in enumerate(self.list_of_models):
            action = QAction(get_icon(modelclass.ICON), modelclass.NAME, self)
            add_actions(menu,(action,))
            self.connect(action, SIG_TRIGGERED, lambda arg=i: self.add_model(arg)) 
        
        self.action.setMenu(menu)
        return menu
        
    def add_model(self,k):
        print k
        i=k
        self.emit(SIG_MODEL_ADDED,i)

        