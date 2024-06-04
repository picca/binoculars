# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:15:11 2020

@author: Prevot
"""
from guidata.qt.QtCore import SIGNAL

# Emitted when a fit is performed
SIG_FIT_DONE = SIGNAL("fit_done")

# Emitted when a fit is performed
SIG_FIT_SAVED = SIGNAL("fit_saved")

# Emitted by parameter table when try is
SIG_TRY_CHANGED = SIGNAL("try_changed")

SIG_TRIGGERED = SIGNAL("triggered()")

SIG_MODEL_ADDED = SIGNAL("model_added(int)")

SIG_MODEL_REMOVED = SIGNAL("model_removed(int)")

SIG_KEY_PRESSED_EVENT = SIGNAL("keyPressedEvent")
