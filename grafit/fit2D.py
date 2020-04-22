# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:44:47 2020

@author: Prevot
"""
#we use x=np.nonzero(contributions) -> for example x[0]=h values ,x[1]=k values

from copy import deepcopy
from collections import OrderedDict
import numpy as np
from os import path,getcwd
import sys, weakref

from lmfit import  minimize, Parameters, Parameter, lineshapes
from lmfit import  Model as LmfitModel
from lmfit.models import COMMON_INIT_DOC,COMMON_GUESS_DOC,update_param_vals

from guidata.qt.QtCore import Qt,SIGNAL,QRegExp,QRect,QSize,QPoint
from guidata.qt.QtGui import (QSyntaxHighlighter,QTextCharFormat,QListWidget,QListWidgetItem,QTextEdit,QTableWidget,QTableWidgetItem,
                              QLabel,QMenu,QAction,QCursor,QColor,QIcon,QPixmap,
                              QMessageBox,QInputDialog,QFileDialog,QDialog,QMainWindow,QSplitter,QApplication,QWidget,
                              QPushButton,QButtonGroup,QSpinBox,QComboBox,QRadioButton,QCheckBox,
                              QVBoxLayout,QHBoxLayout,QGridLayout)


from guidata.qtwidgets import DockableWidgetMixin
from guidata.qthelpers import add_actions,get_std_icon

from guiqwt.builder import PlotItemBuilder as GuiPlotItemBuilder
from guiqwt.config import _
from guidata.configtools import get_icon,add_image_path
from guiqwt.curve import PlotItemList,CurveItem,CurvePlot
from guiqwt.events import RectangularSelectionHandler,setup_standard_tool_filter
from guiqwt.histogram import ContrastAdjustment,lut_range_threshold


from guiqwt.interfaces import ICurveItemType,IShapeItemType,IPanel
from guiqwt.image import ImagePlot,ImageItem
from guiqwt.panels import ID_ITEMLIST,PanelWidget
from guiqwt.plot import  CurveWidget,CurveDialog,PlotManager,BaseCurveWidget,ImageDialog
from guiqwt._scaler import _histogram
from guiqwt.styles import ImageParam
from guiqwt.shapes import RectangleShape,PolygonShape,XRangeSelection
from guiqwt.signals import SIG_VALIDATE_TOOL,SIG_END_RECT,SIG_TOOL_JOB_FINISHED, SIG_START_TRACKING, SIG_CLICK_EVENT,SIG_RANGE_CHANGED,SIG_ITEM_REMOVED
from guiqwt.tools import OpenFileTool,InteractiveTool,CommandTool,DefaultToolbarID,RectangleTool,RectangularActionTool,RectangularShapeTool,BaseCursorTool


from grafit.calculation import get_xywxwy_cxy
from grafit.signals import SIG_TRY_CHANGED,SIG_TRIGGERED,SIG_MODEL_ADDED,SIG_MODEL_REMOVED
from grafit.tools import ShowWeightsTool,RunTool,SaveFitTool,PrefTool,CircularActionToolCXY,AddModelTool,MultiPointTool
from grafit.models import list_of_2D_models

abspath=path.abspath(__file__)
dirpath=path.dirname(abspath)

#we add the current directory to have access to the images
add_image_path(dirpath)


ID_PARAMETER = "parameters"

minimizationmethods=[('leastsq','Levenberg-Marquardt'),
                     ('least_squares','Least-Squares'),
                     ('differential_evolution','Differential evolution'),
                     ('brute','Brute force'),
                     ('nelder','Nelder-Mead'),
                     ('lbfgsb','L-BFGS-B'),
                     ('powell','Powell'),
                     #('cg','Conjugate-Gradient'),   :Jacobian is required
                     #('newton','Newton-Congugate-Gradient'), Jacobian is required
                     ('cobyla','Cobyla'),
                     #('tnc','Truncate Newton'),Jacobian is required
                     #('trust-ncg','Trust Newton-Congugate-Gradient'),Jacobian is required
                     #('dogleg','Dogleg'),Jacobian is required
                     ('basinhopping','Basin-hopping'),
                     ('slsqp','Sequential Linear Squares Programming')]
                     
def _nanmin(data):
    if data.dtype.name in ("float32","float64", "float128"):
        return np.nanmin(data)
    else:
        return data.min()

def _nanmax(data):
    if data.dtype.name in ("float32","float64", "float128"):
        return np.nanmax(data)
    else:
        return data.max()
        

                
        
class ImageNan(ImageItem):
  
  
    def auto_lut_scale(self):  
        _min, _max = _nanmin(self.data), _nanmax(self.data)
        self.set_lut_range([_min, _max])

    def auto_lut_scale_sym(self):  
        _max = max(abs(_nanmin(self.data)), abs(_nanmax(self.data)))
        self.set_lut_range([-_max, _max])

    def get_histogram(self, nbins):
        """interface de IHistDataSource"""
        if self.data is None:
            return [0,], [0,1]
        if self.histogram_cache is None or nbins != self.histogram_cache[0].shape[0]:
            #from guidata.utils import tic, toc
            if False:
                #tic("histo1")
                res = np.histogram(self.data[~np.isnan(self.data)], nbins)
                #toc("histo1")
            else:
                #TODO: _histogram is faster, but caching is buggy
                # in this version
                #tic("histo2")
                _min = _nanmin(self.data)
                _max = _nanmax(self.data)
                if self.data.dtype in (np.float64, np.float32):
                    bins = np.unique(np.array(np.linspace(_min, _max, nbins+1),
                                              dtype=self.data.dtype))
                else:
                    bins = np.arange(_min, _max+2,
                                     dtype=self.data.dtype)
                res2 = np.zeros((bins.size+1,), np.uint32)
                _histogram(self.data.flatten(), bins, res2)
                #toc("histo2")
                res = res2[1:-1], bins
            self.histogram_cache = res
        else:
            res = self.histogram_cache
        return res
        
class PlotItemBuilder(GuiPlotItemBuilder):
    def __init__(self):
        super(PlotItemBuilder,self).__init__()
   
        
    def imagenan(self, data=None, filename=None, title=None, alpha_mask=None,
            alpha=None, background_color=None, colormap=None,
            xdata=[None, None], ydata=[None, None],
            pixel_size=None, center_on=None,
            interpolation='linear', eliminate_outliers=None,
            xformat='%.1f', yformat='%.1f', zformat='%.1f'):
        """
        Make an image `plot item` from data
        (:py:class:`guiqwt.image.ImageItem` object or 
        :py:class:`guiqwt.image.RGBImageItem` object if data has 3 dimensions)
        """
        assert isinstance(xdata, (tuple, list)) and len(xdata) == 2
        assert isinstance(ydata, (tuple, list)) and len(ydata) == 2
        param = ImageParam(title="Image", icon='image.png')
        data, filename, title = self._get_image_data(data, filename, title,
                                                     to_grayscale=True)
        assert data.ndim == 2, "Data must have 2 dimensions"
        if pixel_size is None:
            assert center_on is None, "Ambiguous parameters: both `center_on`"\
                                      " and `xdata`/`ydata` were specified"
            xmin, xmax = xdata
            ymin, ymax = ydata
        else:
            xmin, xmax, ymin, ymax = self.compute_bounds(data, pixel_size,
                                                         center_on)
        self.__set_image_param(param, title, alpha_mask, alpha, interpolation,
                               background=background_color,
                               colormap=colormap,
                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                               xformat=xformat, yformat=yformat,
                               zformat=zformat)
        
        image = ImageNan(data, param)
        image.set_filename(filename)
        if eliminate_outliers is not None:
            image.set_lut_range(lut_range_threshold(image, 256,
                                                    eliminate_outliers))
        return image



make=PlotItemBuilder()



class ExtendedParams():
    #dictionnaire de la liste des courbes pour le fit
    #utilise pour comprendre les Parameters de lmfit
    #ne pilote jamais le tracage des courbes et l'ecriture dans le tableau
    def __init__(self):
        self.reset()
        
    def reset(self):  
        self.modelclasses=list()  #list of the Model classes that compose the Model for fitting the data
        self.models=list()  #list of the Model instances that compose the Model for fitting the data
        self.composite_model=None
        self.ncv=len(self.models)
        self.partry=Parameters()   #parametres d'essai
        self.paropt=Parameters()   #parametres optimises
        self.tags=[]
        self.method='leastsq'
        self.chi2opt = None
        self.dataimage = None   #associated data to be fitted
        self.isfitted = False #is the data fitted?
        self.issaved = False  #has the fit result been saved?
        self.tags=[]  #tags is a list of (name,value)
              
    def printparams(self):
        print "printparams"
        for key in self.partry.keys():
            print key,self.partry[key].value,self.partry[key].vary

    def make_composite_model(self,partry=True):
        #return a composite model based on the list of modelclasses
        #restart the building of models and make new try parameters if partry==True
        self.models=[]
        if len(self.modelclasses)==0:
            self.composite_model = None
            return None
        else:
            model=self.modelclasses[0](prefix='cv0_')
            self.models.append(model)
            self.composite_model = self.models[0]
            if len(self.modelclasses)>2: 
                for k,modelclass in enumerate(self.modelclasses[1:],1):
                    model=self.modelclasses[0](prefix='cv%d_'%k)
                    self.models.append(model)
                    self.composite_model=self.composite_model+model
        if partry:            
            self.partry=self.composite_model.make_params() 
            self.update_tags(self.tags)
        
        
    def add_model_class(self,modelclass,paramvalues=None):
        #ajoute une courbe dans la liste parametres
        self.modelclasses.append(modelclass)
        k=len(self.modelclasses)-1
        model=modelclass(prefix='cv%d_'%k)
        self.models.append(model) 
        if k>0:  #there is already a model
            self.composite_model=self.composite_model+model
        else:
            self.composite_model=model
            
        self.partry=self.composite_model.make_params() 
        self.update_tags(self.tags)
        return model
        
    def remove_model_class(self,k):
        correspondance=range(k)+[-1]+range(k,len(self.modelclasses)-1)
        self.modelclasses.pop(k)
        self.make_composite_model(partry=False)
        self.rebuild_partry(correspondance)

    def rebuild_partry(self,correspondance):
        #rebuild a set of try parameters from correspondance between indices
        #correspondance is the list of old indices pointing toward new ones or nothing if equal to -1

        #first we set values and restrains
        parameters=Parameters()
        for old_entry in self.partry:
            k = self.cvnumber(old_entry)
            if k is not None:  #this is a curve
                if correspondance[k] >= 0:       #otherwise the curve has been suppressed         
                    old_txt = 'cv%d'%k
                    new_txt = 'cv%d'%correspondance[k]
                    new_entry = old_entry.replace(old_txt,new_txt)
                    parameters.add(new_entry, value=self.partry[old_entry].value,vary=self.partry[old_entry].vary,min=self.partry[old_entry].min,max=self.partry[old_entry].max)                
            else:  #this is a fixed parameter (tag)
                parameters.add(old_entry, value=self.partry[old_entry].value,vary=self.partry[old_entry].vary,min=self.partry[old_entry].min,max=self.partry[old_entry].max)                
                
        #then we set expressions that also have to be converted
        for old_entry in self.partry:
            j = self.cvnumber(old_entry)
            if j is not None:  #this is a curve
                if correspondance[j] >= 0:       #otherwise the curve has been suppressed         
                    old_txt = 'cv%d'%j
                    new_txt = 'cv%d'%correspondance[j]
                    new_entry = old_entry.replace(old_txt,new_txt)  #new entry                    
                    new_expr = self.partry[old_entry].expr            #start with old expression
                    if new_expr is not None and len(new_expr)>0:
                        for k in correspondance:
                            old_txt = 'cv%d'%k
                            if old_txt in new_expr:
                                if correspondance[k] == -1:
                                    #the curve referenced in expression has been removed! we cannot maintain the expression
                                    new_expr = ''
                                else:
                                    new_txt = 'cv%d'%correspondance[k]
                                    new_expr = new_expr.replace(old_txt,new_txt)
                        parameters[new_entry].set(expr=new_expr)                
        self.partry = parameters
        self.update_tags(self.tags)
            
    def set_dataimage(self,dataimage):
        self.dataimage = dataimage
        self.chi2opt = None
        self.isfitted = None
        self.issaved = None
        self.update_tags(dataimage.tags)
                    
    def cvnumber(self,entry):
        if entry.startswith('cv'):      
            #return the number of the curve
            return int(entry.split('_')[0][2:])
        else:
            #entry is not a curve parameter
            return None
        
    def freezecurve(self,k):
        #freeze the given curve parameters
        for name in self.models[k].param_names():
            self.partry[name].vary=False
        
    def releasecurve(self,k):
        #release the given curve parameters
        for name in self.models[k].param_names():
            self.partry[name].vary=True
                
    def update_from_estimate(self,estimate):
        for p,v in estimate.iteritems():
            if p in self.partry.keys():
                self.partry[p].value = v
                
    def scale_from_estimate(self,estimate):
        for p,change in estimate.iteritems():
            if p in self.partry.keys():
                self.partry[p].value = change[1]*self.partry[p].value+change[0]
            
    def set_whole_as_try(self):
        for kk in self.partry:
            if kk in self.paropt:
                self.partry[kk].value = self.paropt[kk].value
    
    def eval_try(self,x):
        return self.composite_model.eval(self.partry,x=x)
    
    def eval_opt(self,x):
        return self.composite_model.eval(self.paropt,x=x)
    
    def do_fit(self,data,x,weights):  
        if self.composite_model is None:
            return
        try:
            result=self.composite_model.fit(data,params=self.partry,method=self.method,weights=weights,x=x)
        except Exception as e:
            print e
            QMessageBox.warning(self,'error','error while trying to fit')
        self.paropt = result.params
        self.chi2opt = result.chisqr
        if result.success:
            self.isfitted = True
            self.issaved = False
        
    def eval_integrals_opt(self):
        #evaluate integrals of the different peaks
        return self.composite_model.eval_component_integrals(self.paropt)

    def save_opt_results(self):
        #return lines describing the results that may be saved in a result file
                #we compute the integrals
        #writing names of fitting parameters
        if self.chi2opt is None:
            return
            
        lig="title "
        for kk in self.paropt:
            lig = lig + kk + " "

        integrals = self.eval_integrals_opt()
        
        print "computation of integrals",integrals
        #OrderedDict of {prefix: (integral,error)}
        
        intnames = ''
        intvalues = ''
        interrors = ''
                        
        for prefix in integrals.keys():
            intnames = intnames + prefix + 'integral '  
            value = integrals[prefix]
            intvalues = intvalues + '%g '%(value[0])
            interrors = interrors + '%g '%(value[1])
        
        
        lig=lig+intnames+'chi2 model \n'
        
        #writing names of fitting parameters
        title=self.dataimage.title
        
        lig = lig + title + " "        
              
        for kk in self.partry:
            lig = lig + '%g '%(self.paropt[kk].value)


        lig = lig + intvalues + '%g '%(self.chi2opt) 
        model = self.composite_model.name
        model = model.replace(" ", "")

        lig = lig + model + '\n'
        #print lig
        
        #on ecrit une ligne avec les erreurs
        lig = lig + "sigma "
        for kk in self.partry:
            if self.paropt[kk].stderr is None:
              lig = lig + 'None '
            else:
              lig = lig + '%g '%(self.paropt[kk].stderr)
              
        lig = lig + interrors + '0. '       #no uncertainty on chi2

        lig = lig + model + '\n'             
        return lig         


    def update_tags(self,tags):
        #on supprime les entrees precedentes:
        
        for tag in self.tags:  #remove old list
            name = tag[0]
            if name in self.partry.keys():      
                del self.partry[name] #we suppress the old tag entries
        
        for tag in tags:    #put new list
            name = tag[0]
            self.partry.add(name, value=tag[1], vary=False)  #we reinitialize the tag entries        

        self.tags=tags






class DataImage():
    #a class for describing 2D data to be fitted
    #contains x ranges, y ranges, values, errors
    def __init__(self,data,x_range=[0,1],y_range=[0,1],weights=None,title="",tags=[],xlabel='x',ylabel='y'):
        self.data = data
        self.x_range = x_range
        self.y_range = y_range
        if weights is None:
            weights = np.ones_like(data)
        self.weights = weights
        self.title = title
        self.tags = tags #list of fixed parameters (name,value) associated to the data
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.rf0 = np.sum(self.data[np.nonzero(np.isfinite(self.data))]*self.weights[np.nonzero(np.isfinite(self.data))])





class ImageList(QListWidget):
    ''' A specialized QListWidget that displays the list
        of all images that could be fitted '''
    def __init__(self, parent=None):
        QListWidget.__init__(self, parent)
        self.setSelectionMode(1)
        self.dataimages=[]           
        
    def add_item(self,dataimage):
        ''' add a dataimage item to the list '''  
        self.dataimages.append(dataimage)
        item = QListWidgetItem(self)
        item.setText(dataimage.title)
  
class FunctionList(QListWidget):
    ''' A specialized QListWidget that displays the list
        of functions used in the model '''
    def __init__(self, parent=None):
        QListWidget.__init__(self, parent)
        self.setSelectionMode(1)
        self.current_item_number = -1
        self.initcontexts()
        self.connect(self, SIGNAL("currentRowChanged (int)"),self.row_changed)
        self.connect(self, SIGNAL("itemClicked (QListWidgetItem*)"),self.item_clicked)
        
    def add_item(self,function_name):
        ''' add a function name to the list '''  
        item = QListWidgetItem(self)
        item.setText(function_name)
        self.current_item_number = self.count()-1
        
    def row_changed(self,i):
        #activated when selected row has changed
        self.current_item_number = i
        
    def item_clicked(self):
        self.context.popup(QCursor.pos())
        
    def initcontexts(self):
        #menu pour la colonne 0
        self.context=QMenu(self)

        self.removeAction = QAction("Remove",self)
        self.context.addAction(self.removeAction)
        self.connect(self.removeAction, SIG_TRIGGERED, self.remove_item)            
        
    def remove_item(self): 
        i=self.currentRow()
        self.takeItem(i)
        self.emit(SIG_MODEL_REMOVED, i)
        
    def initialize(self,function_names):
        self.clear()
        for function_name in function_names:
            self.add_item(function_name)
            
    
class ParameterTable(QTableWidget):
    __implements__ = (IPanel,)
    PANEL_ID = ID_PARAMETER
    PANEL_TITLE = "parameters"
    PANEL_ICON = None # string
    #objet qui gere les entree de parametres, l'affichage des courbes, le lancement des fits, l'enregistrement du resultat
    def __init__(self,parent=None,extendedparams=None):    
        QTableWidget.__init__(self,parent=parent)
        
        self.setColumnCount(8) 
        self.setHorizontalHeaderLabels(("Parameters","Estimation","Fit result","Sigma","Restrains","Min","Max","Expression"))
        
        if extendedparams==None:
            self.extendedparams=ExtendedParams()
        else:
            self.extendedparams=extendedparams
        #on remet le tableau a zero : deux rangees pour un polynome du 1er degre
        self.reset()      
                        
        #on definit les differents menus contextuels
        self.initcontexts()
        
        #au cas ou
        self.int1=0
        
        #On connecte les signaux
        self.connect(self, SIGNAL("cellChanged(int,int)"), self.cellChanged)
        self.connect(self, SIGNAL("cellPressed(int,int)"), self.cellPressed)        
        self.connect(self.horizontalHeader(), SIGNAL("sectionPressed(int)"), self.sectionClicked)
        
        #pas de fit sauve
        self.isfitted=False
        self.issaved=False
        self.tags=list()
        self.saveint=False
        self.ints=[]
        self.updatestart=False   #si coche, on active setwholeastry apres enregistrement
        self.cwd=abspath

    def register_plot(self, baseplot):
        pass
                
    def register_panel(self, manager):
        """Register panel to plot manager"""
        pass
      
    def configure_panel(self):
        """Configure panel"""
        pass
      
    def reset(self): 
        self.extendedparams.reset()
        self.initparams(settry=True)
        self.savename=None
        self.isfitted=False
        
    def inititems(self,listrow):
        #met pour les lignes i de listrow un item vide non editable pour colonnes 0,2,3,4
        for i in listrow: 
            for j in range(8):
                self.setItem(i,j,QTableWidgetItem(0))
            self.item(i,0).setFlags(Qt.ItemFlags(33))
            self.item(i,2).setFlags(Qt.ItemFlags(33))
            self.item(i,3).setFlags(Qt.ItemFlags(33))
            self.item(i,4).setFlags(Qt.ItemFlags(33))
            
    def initparams(self,settry=False,setopt=False):
        blocked=self.blockSignals(True)
        
        #on redessine le tableau a l'aide du dictionnaire extendedparams
        #si settry, on remplit aussi les valeurs d'essai, les bornes, les expressions
        
        oldrc=self.rowCount()
        newrc=len(self.extendedparams.partry)
        self.setRowCount(newrc)
        
        if newrc>oldrc:
            #on rajoute des rangees vides
            self.inititems(range(oldrc,newrc))
            
        n=0
        for entry, value in self.extendedparams.partry.iteritems():
            self.setrow(entry,n,settry,setopt)
            n+=1
        
        self.blockSignals(blocked)
        
    def setrow(self,nom,n,settry=True,setopt=False):
        blocked=self.blockSignals(True)
        
        ptry=self.extendedparams.partry
        popt=self.extendedparams.paropt
        self.item(n,0).setText(nom)
        
        if settry:
            vtry=ptry[nom].value       

            mintry=ptry[nom].min
            maxtry=ptry[nom].max
            print nom,mintry,maxtry
            varytry=ptry[nom].vary
            exprtry=ptry[nom].expr
            self.item(n,1).setText("%f" % (vtry))
            
            if exprtry is not None:
                self.item(n,4).setText("Expr")
                self.item(n,7).setText(exprtry)
                
            elif varytry is True:                
                self.item(n,4).setText("Free")
            else:
                self.item(n,4).setText("Fixed")
                
            if mintry is not None:
                self.item(n,5).setText("%f" % (mintry))
                
            if maxtry is not None:
                self.item(n,6).setText("%f" % (maxtry))
                        
        if setopt:
            vopt=popt[nom].value
            self.item(n,2).setText("%f" % (vopt))
            verr=popt[nom].stderr
            if verr is not None:
              self.item(n,3).setText("%f" % (verr))
            
        else:
            self.item(n,2).setText("")
            self.item(n,3).setText("")
            
        self.blockSignals(blocked)  #going back to previous setting
    
    def initcontexts(self):
        #initialise une liste de menus contextuels pour les cases, associes a chaque colonne
        self.contexts=list(None for i in range(self.columnCount()))
        #initialise une liste de menus contextuels pour les sectionheaders, associes a chaque colonne
        self.Contexts=list(None for i in range(self.columnCount()))
        
        #menu pour la colonne 0

        context=QMenu(self)

        self.setastry = QAction("Set as try",self)
        context.addAction(self.setastry)
        self.connect(self.setastry, SIG_TRIGGERED, self.set_as_try)            

        self.showfit = QAction("Display Fit",self)
        context.addAction(self.showfit)
        self.connect(self.showfit, SIG_TRIGGERED, self.show_fit)            
        
        self.showtry = QAction("Display Try",self)
        context.addAction(self.showtry)        
        self.connect(self.showtry, SIG_TRIGGERED, self.show_try)            
   
        self.contexts[2]=context

        #menu pour la colonne 4
        context=QMenu(self)
        
        self.setfixed = QAction("Fixed",self)
        context.addAction(self.setfixed)
        self.connect(self.setfixed, SIG_TRIGGERED, self.set_fixed)           

        self.setfree = QAction("Free",self)
        context.addAction(self.setfree)
        self.connect(self.setfree, SIG_TRIGGERED, self.set_free) 

        self.useexpr = QAction("Expr",self)
        context.addAction(self.useexpr)
        self.connect(self.useexpr, SIG_TRIGGERED, self.use_expr) 
                   
        self.contexts[4]=context
                
        #menu pour le header 2
        context=QMenu(self)
        self.setwholeastry = QAction("Set whole as try",self)
        context.addAction(self.setwholeastry)
        self.connect(self.setwholeastry, SIG_TRIGGERED, self.set_whole_as_try)           

        self.Contexts[2]=context
        
        #menu pour le header 4
        context=QMenu(self)
        self.inverserestrains = QAction("Inverse restrains",self)
        context.addAction(self.inverserestrains)
        self.connect(self.inverserestrains, SIG_TRIGGERED, self.inverse_restrains)           

        self.fixall = QAction("All fixed",self)
        context.addAction(self.fixall)
        self.connect(self.fixall, SIG_TRIGGERED, self.fix_all)           

        self.releaseall = QAction("All free",self)
        context.addAction(self.releaseall)
        self.connect(self.releaseall, SIG_TRIGGERED, self.release_all)           

        self.Contexts[4]=context
        
        
    def cellPressed(self,int1,int2):
        self.int1=int1
        
        context=self.contexts[int2]
        
        if context is not None:
            context.popup(QCursor.pos())
         
    def cellChanged(self,int1,int2):        
        if int2==1:  #on change la valeur du parametre                
            txt= str(self.item(int1,1).text())
            entry= str(self.item(int1,0).text())
            try:
                self.extendedparams.partry[entry].value=float(txt)
                self.emit(SIG_TRY_CHANGED)
            except ValueError:
                QMessageBox.warning('warning','unable to convert text to float')
                            
          
        if int2==5:  #on change la borne inferieure
            txt= str(self.item(int1,5).text())
            entry= str(self.item(int1,0).text())
            try:
                self.extendedparams.partry[entry].min=float(txt)
            except ValueError:
                QMessageBox.warning('warning','unable to convert text to float')
          
        if int2==6:  #on change la borne superieure
            txt= str(self.item(int1,6).text())
            entry= str(self.item(int1,0).text())
            try:
                self.extendedparams.partry[entry].max=float(txt)
            except ValueError:
                QMessageBox.warning('warning','unable to convert text to float')
          
        if int2==7:  #on change l'expression
            txt= str(self.item(int1,7).text())
            entry= str(self.item(int1,0).text())
            if len(txt) is 0:
                self.extendedparams.partry[entry].expr=None                  
                self.extendedparams.partry[entry].vary=True
            else:
                self.extendedparams.partry[entry].expr=txt      
                self.item(int1,4).setText("Expr")

                                        
    def sectionClicked(self,int2):
        #quand une colonne entiere est selectionnee valable pour les colonnes 2 et 4
        context=self.Contexts[int2]
        if context is not None:
            context.popup(QCursor.pos())
        
    def freeze_cv(self) :
        #on met fixe tous les parametres d'une courbe
        entry=str(self.item(self.int1,0).text())
        icurve=self.extendedparams.cv_number(entry)
        self.extendedparams.freezecurve(icurve)
        self.initparams(settry=True,setopt=True)
        
    def release_cv(self) :
        #on met fixe tous les parametres d'une courbe
        entry=str(self.item(self.int1,0).text())
        icurve=self.extendedparams.cv_number(entry)
        self.extendedparams.releasecurve(icurve)
        self.initparams(settry=True,setopt=True)
                
    def release_all(self) :
        #on met fixe tous les parametres du fond
        for int1 in range(self.rowCount()):
            item=self.item(int1,4)
            item.setText("Free")
            entry=str(self.item(int1,0).text())
            self.extendedparams.partry[entry].vary=True
                
    def fix_all(self) :
        #on met fixe tous les parametres du fond
        for int1 in range(self.rowCount()):
            item=self.item(int1,4)
            item.setText("Fixed")
            entry=str(self.item(int1,0).text())
            self.extendedparams.partry[entry].vary=False
                
    def set_cv(self,curvetype=None,params=None,bg=0.):
        #icurve numero de la courbe a changer le cas echeant -1 c'est la derniere
        pass
        
    def rm_cv(self):
        #on supprime la courbe selectionnee        
        pass

    def ch_cv(self):
        #a partir de la souris, on retrace la courbe selectionnee
        pass
    
    def show_fit():
        pass
    
    def show_try():
        pass
        
    def set_fixed(self):
        item=self.item(self.int1,4)
        item.setText("Fixed")
        entry=str(self.item(self.int1,0).text())
        self.extendedparams.partry[entry].vary=False
        self.extendedparams.partry[entry].expr=None
        
    def set_free(self):
        item=self.item(self.int1,4)
        item.setText("Free")
        entry=str(self.item(self.int1,0).text())
        self.extendedparams.partry[entry].vary=True
        self.extendedparams.partry[entry].expr=None
        
    def use_expr(self):
        item=self.item(self.int1,4)
        item.setText("Expr")
        entry=str(self.item(self.int1,0).text())
        
        txt= str(self.item(self.int1,7).text())
        self.extendedparams.partry[entry].express=txt    #in principle, should have been already set
        
        if len(txt) is not 0:    #on met l'expression indiquee
           self.extendedparams.partry[entry].expr=txt
           self.extendedparams.partry[entry].vary=False
        
    def inverse_restrains(self):
        #inverse les contraintes: les parametres libres deviennent fixes et vice-versa
        for int1 in range(self.rowCount()):
            item=self.item(int1,4)
            entry=str(self.item(int1,0).text())
            vary=self.extendedparams.partry[entry].vary
            if vary:
                self.extendedparams.partry[entry].vary=False
                item.setText("Fixed")
            else:
                self.extendedparams.partry[entry].vary=True
                item.setText("Free")
        
    def set_whole_as_try(self):
        self.extendedparams.set_whole_as_try()
        self.initparams(settry=True,setopt=True)
        
    def set_as_try(self):
        item=self.item(self.int1,1)
        entry=str(self.item(self.int1,0).text())
        if entry in self.extendedparams.paropt:
            self.extendedparams.partry[entry].value=self.extendedparams.paropt[entry].value
            item.setText("%f" %(self.extendedparams.paropt[entry].value))
            self.drawtry()
                

#this widget does nothing by itself but contains a layout for plots 
class LayoutWidget(QWidget):
    def __init__(self,parent,orientation='horizontal'):
        self.parent=parent
        QWidget.__init__(self,parent=parent)
        if orientation == 'horizontal':
            self.layout = QHBoxLayout()
        else:
            self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
    def addWidget(self,widget):
        self.layout.addWidget(widget)

    def save_widget(self, fname):
        """Grab widget's window and save it to filename (*.png)"""
        fname = unicode(fname)
        pixmap = QPixmap.grabWidget(self)
        pixmap.save(fname, 'PNG')
        
        
class FitWidget(QWidget):
   def __init__(self,parent=None,extendedparams=None):    
       QWidget.__init__(self,parent=parent)
       
       self.extendedparams=extendedparams
       
       layout=QVBoxLayout()
       self.setLayout(layout)
       
       self.method=QComboBox(self)
       for couple in minimizationmethods:
           self.method.addItem(couple[1])
       self.connect(self.method, SIGNAL(_("currentIndexChanged(int)")), self.index_changed)
       
       self.chi2=QLabel('chi2=')
       self.rf=QLabel('Rf=')
       self.di=QLabel('dI=')
  
       #signal connected in rodeo       
       layout.addWidget(self.method)     
       layout.addWidget(self.chi2)
       layout.addWidget(self.rf)
       layout.addWidget(self.di)

     
   def set_method_number(self,i):
       self.method.setCurrentIndex(i)
   
   def set_method_name(self):
       i=self.extendedparams.methods.index(self.extendedparams.method)  
       self.method.setCurrentIndex(i)
   
   def index_changed(self,int):
       self.extendedparams.method=minimizationmethods[int][0]
      
   def set_chi2(self,chi2,rf,di):
       self.chi2.setText('chi2=%g'%chi2)
       self.rf.setText('Rf=%g'%rf)
       self.di.setText('dI=%g'%di)
       
   def set_difference(self,delta):
       self.difference.setText('DI=%g'%delta)
           
   def add_mem(self):
       #we memorize the current fit result, if any
       pass
      
   def remove_mem(self):
       #we suppress the corresponding entry
       pass
       
               
#this window holds the toolbar and the centralwidget         
class Fit2DWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.extendedparams=ExtendedParams()
        self.configure_splitter()
        self.set_default_images()
        self.setGeometry(10,30,1260,950)
                        
        toolbar = self.addToolBar("tools")
        self.manager.add_toolbar(toolbar, id(toolbar))
#       widget.manager.set_default_toolbar(toolbar)
        self.register_tools()
        self.create_connect()
        self.set_default_params()
        
        self.cwd = getcwd()
        self.savename = None  #name where to save the fit
    
    def set_default_images(self):
        self.data_image = make.imagenan(np.zeros((1,1)),title='data image',xdata=[0,1],ydata=[0,1])
        self.model_image = make.imagenan(np.zeros((1,1)),title='model image',xdata=[0,1],ydata=[0,1])
        self.difference_image = make.imagenan(np.zeros((1,1)),title='difference image',xdata=[0,1],ydata=[0,1],colormap='RdBu')
        
        z=np.empty((0,))
        self.data_hprofile = make.curve(z,z,title='data horizontal profile',marker="Diamond",markerfacecolor='r',markeredgecolor='r',markersize=4,linestyle="NoPen")
        self.data_vprofile = make.curve(z,z,title='data vertical profile',marker="Diamond",markerfacecolor='r',markeredgecolor='r',markersize=4,linestyle="NoPen")

        self.model_hprofile = make.curve(z,z,title='model horizontal profile',color='k')
        self.model_vprofile = make.curve(z,z,title='model vertical profile',color='k')
        
        self.difference_hprofile = make.curve(z,z,title='model horizontal difference',color='b')
        self.difference_vprofile = make.curve(z,z,title='model vertical difference',color='b')
       
        self.data_plot.add_item(self.data_image)
        self.model_plot.add_item(self.model_image)
        self.difference_plot.add_item(self.difference_image)
        
        self.hprofile.add_item(self.data_hprofile)
        self.hprofile.add_item(self.model_hprofile)
        self.hprofile.add_item(self.difference_hprofile)
        
        self.vprofile.add_item(self.data_vprofile)
        self.vprofile.add_item(self.model_vprofile)
        self.vprofile.add_item(self.difference_vprofile)
        
        self.data = None              
      
    def configure_splitter(self):
        
        centralwidget = QSplitter(Qt.Horizontal, self)
        splitter = QSplitter(Qt.Vertical, self)
        self.setCentralWidget(centralwidget)
        
        centralwidget.addWidget(splitter)
        
        self.data_plot = ImagePlot(self,title='data',yreverse=False,lock_aspect_ratio=False,xunit='',yunit='')        
        self.model_plot = ImagePlot(self,title='model',yreverse=False,lock_aspect_ratio=False,xunit='',yunit='')        
        self.difference_plot = ImagePlot(self,title='difference',yreverse=False,lock_aspect_ratio=False,xunit='',yunit='')
        
        self.contrast = ContrastAdjustment(self)
        self.hprofile = CurvePlot(self,title='horizontal profile')
        self.vprofile = CurvePlot(self,title='vertical profile')
        
        #adding layout widget with 3 plots and itemlist
        self.imagelayoutw = LayoutWidget(self,'horizontal')
        self.imagelayoutw.addWidget(self.data_plot)
        self.imagelayoutw.addWidget(self.model_plot)
        self.imagelayoutw.addWidget(self.difference_plot)
        splitter.addWidget(self.imagelayoutw)
                
        #adding layout widget with 3 plots with profiles
        self.plotlayoutw = LayoutWidget(self,'horizontal')
        self.plotlayoutw.addWidget(self.contrast)
        self.plotlayoutw.addWidget(self.hprofile)
        self.plotlayoutw.addWidget(self.vprofile)
        splitter.addWidget(self.plotlayoutw)
                
        #adding parameter table panel
        self.parametertable= ParameterTable(self,extendedparams=self.extendedparams)
        splitter.addWidget(self.parametertable)
        
        #adding a vertical layout with itemlist, fitwidget, functionlist
        self.vlayoutw = LayoutWidget(self,'vertical')
        self.image_list = ImageList(self)
        self.fitwidget = FitWidget(self,self.extendedparams)
        self.function_list = FunctionList(self)
        self.vlayoutw.addWidget(self.image_list)
        self.vlayoutw.addWidget(self.fitwidget)
        self.vlayoutw.addWidget(self.function_list)
        centralwidget.addWidget(self.vlayoutw)

        #registring panels to the plot manager
        self.manager = PlotManager(self)
        for plot in (self.data_plot, self.model_plot, self.difference_plot, self.hprofile, self.vprofile):
            self.manager.add_plot(plot)
        for panel in (self.contrast,self.parametertable):
            self.manager.add_panel(panel)
    
    def register_tools(self):
        self.manager.register_all_image_tools()
        self.addmodeltool=self.manager.add_tool(AddModelTool, list_of_2D_models)
        self.spottool=self.manager.add_tool(CircularActionToolCXY,self.estimate_spot,self.scale_spot)
        self.leveltool=self.manager.add_tool(MultiPointTool,self.estimate_bg)
        self.runtool = self.manager.add_tool(RunTool)
        self.savefittool = self.manager.add_tool(SaveFitTool)
        self.preftool = self.manager.add_tool(PrefTool)
        self.showweightstool = self.manager.add_tool(ShowWeightsTool)
        
    def estimate_bg(self,plot, pts):
        if plot == self.data_plot:
            intensities=[]
            for pt in pts:
                intensities.append(self.data_image.get_data(pt[0],pt[1]))
            i = self.function_list.current_item_number  #number of the curve to ajust
            prefix='cv%d_'%i
            estimated_values = {prefix+'bg':np.mean(np.array(intensities))}
            if len(pts) > 2:
                #try:
                    mat=np.ones((3,3))
                    mat[:,0:2]=pts
                    sol=np.linalg.solve(mat,intensities)
                    estimated_values = {prefix+'slope_x':sol[0], prefix+'slope_y':sol[1], prefix+'bg':sol[2]}
                #except np.linalg.LinAlgError:
                #   pass
                  
            self.extendedparams.update_from_estimate(estimated_values)
            self.parametertable.initparams(settry=True)
            self.update_try()

    def estimate_spot(self,plot, p0, p1):
        #from graphical positions of the circle drawn, estimate the shape of the spot
        loc0,loc1,width0,width1=get_xywxwy_cxy(plot, p0, p1)
        i = self.function_list.current_item_number  #number of the curve to ajust
        prefix='cv%d_'%i
        estimated_values={prefix+'loc0':loc0,prefix+'loc1':loc1,prefix+'width0':width0,prefix+'width1':width1}
        self.extendedparams.update_from_estimate(estimated_values)
        self.parametertable.initparams(settry=True)
        self.update_try()
        
    def scale_spot(self,plot,key):
        i = self.function_list.current_item_number  #number of the curve to ajust
        prefix='cv%d_'%i
        
        zaxis = plot.colormap_axis
        #_min,_max=
        _min,_max= plot.get_axis_limits(zaxis)

        shift=0.
        scale=1.
        
        if key == 42:
            scale=1.1
        elif key == 47:
            scale=1./1.1
                
        if key == 43:
            shift=(_max-_min)/10.
        elif key == 45:
            shift=-(_max-_min)/10.
            
        estimated_values={prefix+'amp':(shift,scale),prefix+'bg':(shift,scale)}
        self.extendedparams.scale_from_estimate(estimated_values)            
        self.parametertable.initparams(settry=True)
        self.update_try()        
      
    def create_connect(self):
        self.connect(self.parametertable,SIG_TRY_CHANGED,self.update_try)
        self.connect(self.addmodeltool,SIG_MODEL_ADDED,self.add_model_class)
        self.connect(self.runtool,SIG_VALIDATE_TOOL,self.do_fit)
        self.connect(self.savefittool,SIG_VALIDATE_TOOL,self.save_fit)
        self.connect(self.image_list, SIGNAL("currentRowChanged (int)"),self.image_list_row_changed)
        self.connect(self.preftool.showtagsaction, SIG_TRIGGERED, self.show_tags) 
        self.connect(self.preftool.saveprefaction, SIG_TRIGGERED, self.set_save_filename) 
        self.connect(self.function_list,SIG_MODEL_REMOVED, self.remove_model)
        self.connect(self.showweightstool,SIG_VALIDATE_TOOL,self.show_weigths)
        self.connect(self.preftool.showdiffaction, SIG_TRIGGERED, self.show_diff)
        
    def set_default_params(self):
        #create a fit with a linear bg and a lorenzian polar function     
        self.add_model_class(0)
        self.add_model_class(1)
            
    def add_model_class(self,i):
        modelclass = self.addmodeltool.list_of_models[i]
        added_model=self.extendedparams.add_model_class(modelclass)
        self.function_list.add_item(added_model.name)
        self.parametertable.initparams(settry=True)
        if self.data is not None:
            self.update_try()
            
    def remove_model(self,i):
        self.extendedparams.remove_model_class(i)
        self.parametertable.initparams(settry=True)   #refill the parametertable with new values
        self.function_list.initialize(list((model.name for model in self.extendedparams.models))) #refill the function table

    def set_save_filename(self):
        self.savename = str(QFileDialog.getSaveFileName(None, 'Save fit parameters',self.cwd,filter="*.txt"))
        if len(self.savename) is 0:
          self.savename=None
        else:
          self.cwd=path.dirname(path.realpath(self.savename))
          
    def save_fit(self):
        #sauvegarde des donnees
        if not self.extendedparams.isfitted:
            #no fit has been performed!
            return        
        
        if self.savename is None:
            self.set_save_filename() #definit le nom de fichier
            if self.savename is None:
                return
             
        fic=open(self.savename,'a')
        fic.write(self.extendedparams.save_opt_results())         
        fic.close()
        
        #on sauve la figure de fit
        ific=1
        
        figshortname=self.dataimage.title+'_%.3d.png'%(ific)
        figname=path.join(self.cwd,figshortname)
        while path.isfile(figname):
            ific=ific+1        
            figshortname=self.dataimage.title+'_%.3d.png'%(ific)
            figname=path.join(self.cwd,figshortname)
        print figname,' sauve'
        self.imagelayoutw.save_widget(figname)  
        if self.preftool.restartaction.isChecked():
            self.extendedparams.set_whole_as_try()
            self.parametertable.initparams(settry=True)
        self.extendedparams.issaved = True   
          
    def show_tags(self):
        pass
        
    def add_data(self,data,x_range=[0.,1.],y_range=[0.,1.],weights=None,title="",tags=[],xlabel='x',ylabel='y'):
        dataimage = DataImage(data,x_range,y_range,weights,title,tags,xlabel,ylabel)
        self.image_list.add_item(dataimage)
        self.dataimage=dataimage  #pointer to the current dataimage
        self.set_image(dataimage)
        
    def image_list_row_changed(self,i):
        #the current image list row has changed
        self.dataimage = self.image_list.dataimages[i]
        if self.extendedparams.isfitted and not self.extendedparams.issaved:
            #a fit has been performed but not saved
            i=QMessageBox.question(self, "save", "do you want to save the fit?", "Yes","No", "Cancel")
        
            if i==0:
                self.save_fit()
                
        self.set_image(self.dataimage)
        
    def set_image(self,dataimage):
        #set image to fit where dataimage contains data: a 2D array, xdata and ydata: the limits, weights: 1/sigma
        #update image representation
        self.data_image.set_xdata(*dataimage.x_range)
        self.data_image.set_ydata(*dataimage.y_range)
        self.data_image.set_data(np.array(dataimage.data))   #ImageItem.set_data does not make a copy

        self.data_plot.set_title(dataimage.title)
        
        self.model_image.set_xdata(*dataimage.x_range)
        self.model_image.set_ydata(*dataimage.y_range)
        self.model_image.set_data(np.array(dataimage.data))
        
        self.difference_image.set_xdata(*dataimage.x_range)
        self.difference_image.set_ydata(*dataimage.y_range)
        self.difference_image.set_data(np.array(dataimage.data))
        
        #update axis labels
        self.data_plot.set_axis_title('left',dataimage.ylabel)
        self.data_plot.set_axis_title('bottom',dataimage.xlabel)
        self.model_plot.set_axis_title('left',dataimage.ylabel)
        self.model_plot.set_axis_title('bottom',dataimage.xlabel)
        self.difference_plot.set_axis_title('left',dataimage.ylabel)
        self.difference_plot.set_axis_title('bottom',dataimage.xlabel)
    
        #update values to be fitted
        self.fit_indices = np.nonzero(np.isfinite(dataimage.data))  #indices to fit
        self.model_image.data[self.fit_indices]=0.  #start from 0
        
        self.data = dataimage.data[self.fit_indices]     #value at indices, 1D array!
        
        if dataimage.weights is not None:
            self.weights = dataimage.weights[self.fit_indices]       #error at indices, 1D array!
        else:
            self.weights = None
            
        sx=dataimage.data.shape[1]
        sy=dataimage.data.shape[0]
        fx=(dataimage.x_range[1]-dataimage.x_range[0])/sx
        fy=(dataimage.y_range[1]-dataimage.y_range[0])/sy
         
        #we put first x and y in the list 
        self.x=[self.fit_indices[1]*fx+dataimage.x_range[0]+fx/2.,self.fit_indices[0]*fy+dataimage.y_range[0]+fy/2.]
        
        hrange = np.arange(dataimage.x_range[0]+fx/2,dataimage.x_range[1],fx)
        vrange = np.arange(dataimage.y_range[0]+fy/2,dataimage.y_range[1],fy)
        
        array1 = np.nan_to_num(dataimage.data)
        array2 = np.isfinite(dataimage.data)
        
        sum1=(np.sum(array1,axis=0)/np.sum(array2,axis=0))
        sum2=(np.sum(array1,axis=1)/np.sum(array2,axis=1))
        
        hrange=hrange[np.isfinite(sum1)]
        vrange=vrange[np.isfinite(sum2)]
        
        self.data_hprofile.set_data(hrange,sum1[np.isfinite(sum1)])
        self.data_vprofile.set_data(vrange,sum2[np.isfinite(sum2)])
        
        if self.preftool.scaleprefaction.isChecked():
            self.data_plot.do_autoscale(replot=True)
        else:
            self.data_plot.replot()
        self.data_plot.update_colormap_axis(self.data_image)

        self.extendedparams.set_dataimage(dataimage)
        self.parametertable.initparams(settry=True)    #if tags have changed, put them into the table
        self.update_try()
        
    def show_weigths(self):
        if self.showweightstool.action.isChecked():
            self.data_image.set_data(self.dataimage.weights)
        else:
            self.data_image.set_data(self.dataimage.data)
        self.data_plot.replot()
        
    def show_diff(self):
        self.difference_hprofile.setVisible(self.preftool.showdiffaction.isChecked())
        self.difference_vprofile.setVisible(self.preftool.showdiffaction.isChecked())
        self.hprofile.replot()
        self.vprofile.replot()
      
    def update_try(self):
        #evaluate model with try parameters and update model and difference image
        values = self.extendedparams.eval_try(self.x)
        self.model_image.data[self.fit_indices] = values
        
        diff = values-self.data
        self.difference_image.data[self.fit_indices] = diff
        diff = diff*self.weights
        chi2 = np.sum((diff)**2)
        
        di=np.sum((diff))
        rf=np.sum(np.absolute(diff))/self.dataimage.rf0
        
        self.fitwidget.set_chi2(chi2,rf,di)
        self.update_images_and_profiles()
        
    def update_opt(self):
        #evaluate model with try parameters and update model and difference image
        values=self.extendedparams.eval_opt(self.x)
        self.model_image.data[self.fit_indices] = values
        
        diff = values-self.data
        self.difference_image.data[self.fit_indices] = diff
        diff = diff*self.weights
        chi2 = np.sum((diff)**2)
        
        di=np.sum((diff))
        rf=np.sum(np.absolute(diff))/self.dataimage.rf0
        
        self.fitwidget.set_chi2(chi2,rf,di)
        self.update_images_and_profiles()
                
    def update_images_and_profiles(self):     
        
        hrange = self.data_hprofile.get_data()[0]
        vrange = self.data_vprofile.get_data()[0]
        
        array1 = np.nan_to_num(self.model_image.data)
        array2 = np.isfinite(self.model_image.data)
        
        sum1=(np.sum(array1,axis=0)/np.sum(array2,axis=0))
        sum2=(np.sum(array1,axis=1)/np.sum(array2,axis=1))
        
        self.model_hprofile.set_data(hrange,sum1[np.isfinite(sum1)])
        self.model_vprofile.set_data(vrange,sum2[np.isfinite(sum2)])
                
        array1 = np.nan_to_num(self.difference_image.data)
        array2 = np.isfinite(self.difference_image.data)
        
        sum1=(np.sum(array1,axis=0)/np.sum(array2,axis=0))
        sum2=(np.sum(array1,axis=1)/np.sum(array2,axis=1))

        self.difference_hprofile.set_data(hrange,sum1[np.isfinite(sum1)])
        self.difference_vprofile.set_data(vrange,sum2[np.isfinite(sum2)])
                        
        
        if self.preftool.samescaleaction.isChecked():
            _min, _max = _nanmin(self.dataimage.data), _nanmax(self.dataimage.data)
            self.model_image.set_lut_range([_min, _max])
        elif self.preftool.scaleprefaction.isChecked():
            self.model_image.auto_lut_scale()
        
        if self.preftool.scaleprefaction.isChecked():
            self.difference_image.auto_lut_scale_sym()
            self.model_plot.do_autoscale(replot=True)
            self.difference_plot.do_autoscale(replot=True)
            self.hprofile.do_autoscale(replot=True)
            self.vprofile.do_autoscale(replot=True)
        else:
            self.model_plot.replot()        
            self.difference_plot.replot()        
            self.hprofile.replot()
            self.vprofile.replot()
        self.model_plot.update_colormap_axis(self.model_image)
        self.difference_plot.update_colormap_axis(self.difference_image)

    def do_fit(self):
        self.extendedparams.do_fit(self.data,self.x,self.weights)
        self.update_opt()
        self.parametertable.initparams(setopt=True)

def test(win):
    #make a test model
    xmin=-1.
    xmax=1.  #bornes
    ymin=-1.
    ymax=1.
    
    data=np.random.rand(100,100)
    x=np.arange(xmin,xmax,0.02)
    y=np.arange(ymin,ymax,0.02)
    xv,yv=np.meshgrid(x,y)
    data=data+10./(1.+xv*xv+yv*yv)
    tags=[('Qz',0.15)]
    win.add_data(data,(xmin,xmax),(ymin,ymax),title='test1',tags=tags)
    
    xmin=-1.2
    xmax=1.2  #bornes
    ymin=-1.5
    ymax=1.5
    data=np.random.rand(150,120)
    x=np.arange(xmin,xmax,0.02)
    y=np.arange(ymin,ymax,0.02)
    xv,yv=np.meshgrid(x,y)
    data=data+5./(1.+xv*xv+0.5*yv*yv)
    tags=[('l',0.3)]
    win.add_data(data,(xmin,xmax),(ymin,ymax),title='test2',tags=tags)
    

if __name__ == "__main__":
    
    from guidata import qapplication
    _app = qapplication()
    #win = FitDialog()   
    win = Fit2DWindow()
    test(win)
        
    """
    
    x=np.nonzero(data)
    fx=(xmax-xmin)/s
    fy=(ymax-ymin)/s
    
    xx=(np.array(x[0])+0.5)*fx+xmin
    xy=(np.array(x[1])+0.5)*fy+ymin
    
    models=(Linear2DModel,Lorenzian2DModel)
    
    win.extendedparams.add_model_class(Linear2DModel)
    win.extendedparams.add_model_class(Lorenzian2DModel)
    win.parametertable.initparams(settry=True)
    
    
    data[x]=win.extendedparams.composite_model.eval(win.extendedparams.partry,x=[xx,xy])   
    image1=make.image(data,title='test',xdata=[xmin,xmax],ydata=[ymin,ymax])
    """
    win.show()
    #test=ParameterTable(extendedparams=ExtendedParams())
    #test.show()
    _app.exec_()
