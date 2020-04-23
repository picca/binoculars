# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:57:49 2020

@author: Prevot
"""


def get_xywxwy_cxy(plot, p0, p1):
    # return position and width from a centered circular shape drawn from p0 to p1
    ax, ay = plot.get_axis_id("bottom"), plot.get_axis_id("left")
    x1, y1 = plot.invTransform(ax, p0.x()), plot.invTransform(ay, p0.y())
    x2, y2 = plot.invTransform(ax, p1.x()), plot.invTransform(ay, p1.y())
    return x1, y1, 2.0 * abs(x2 - x1), 2 * abs(y1 - y2)
