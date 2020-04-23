# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:02:23 2020

@author: Prevot
"""
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from scipy.integrate import dblquad
from scipy.special import erf
from lmfit import lineshapes
from lmfit import Model as LmfitModel
from lmfit.models import COMMON_INIT_DOC, COMMON_GUESS_DOC, update_param_vals
import operator


def linear2D(x, slope_x=0.0, slope_y=0.0, bg=0.0):
    return bg + x[0] * slope_x + x[1] * slope_y


"""*****************************************************************************************************"""


def lorenzianpower2D(
    x, amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, th=0.0, power=2.0
):
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x[0] - loc0
    x1 = x[1] - loc1
    x0, x1 = x0 * cth - x1 * sth, x0 * sth + x1 * cth  # rotation
    if power == 2.0:
        return amp / (1.0 + ((x0 / width0) ** 2 + ((x1 / width1) ** 2)))
    else:
        return amp / (
            1.0 + ((x0 / width0) ** 2 + ((x1 / width1) ** 2)) ** (power / 2.0)
        )


def lorenzianpower(
    y, x, amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, th=0.0, power=2.0
):
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x - loc0
    x1 = y - loc1
    x0, x1 = x0 * cth - x1 * sth, x0 * sth + x1 * cth  # rotation
    if power == 2.0:
        return amp / (1.0 + ((x0 / width0) ** 2 + ((x1 / width1) ** 2)))
    else:
        return amp / (
            1.0 + ((x0 / width0) ** 2 + ((x1 / width1) ** 2)) ** (power / 2.0)
        )


def lorenzianpowerint(
    amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, th=0.0, power=2.0
):
    return dblquad(
        lambda y, x: lorenzianpoly(y, x, amp, loc0, loc1, width0, width1, th, power),
        -np.inf,
        np.inf,
        lambda x: -np.inf,
        lambda x: np.inf,
    )


"""*****************************************************************************************************"""


def lorenziangauss2D(x, amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, th=0.0):
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x[0] - loc0
    x1 = x[1] - loc1
    x0, x1 = x0 * cth - x1 * sth, x0 * sth + x1 * cth  # rotation
    return amp * np.exp(-((x1 / width1) ** 2)) / (1.0 + ((x0 / width0) ** 2))


def lorenziangaussint(amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, th=0.0):
    return (amp * np.pi * np.sqrt(np.pi) * width0 * width1, 0.0)


"""*****************************************************************************************************"""


def lorenzianpoly2D(
    x, amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, eta0=1.0, eta1=1.0, th=0.0
):
    # eta is the ratio of lorenzian component, if eta=0, it is lorenzian, if eta=1 it is like 1/(1+x**4)
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x[0] - loc0
    x1 = x[1] - loc1
    x0, x1 = x0 * cth - x1 * sth, x0 * sth + x1 * cth  # rotation
    u0 = (x0 / width0) ** 2
    u1 = (x1 / width1) ** 2
    return amp / (1.0 + (1.0 - eta0 + eta0 * u0) * u0 + (1.0 - eta1 + eta1 * u1) * u1)


def lorenzianpoly(
    y,
    x,
    amp=1.0,
    loc0=0.0,
    loc1=0.0,
    width0=1.0,
    width1=1.0,
    eta0=1.0,
    eta1=1.0,
    th=0.0,
):
    # same with x y coordinates
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x - loc0
    y0 = y - loc1
    x0, y0 = x0 * cth - y0 * sth, x0 * sth + y0 * cth  # rotation
    u0 = (x0 / width0) ** 2
    u1 = (y0 / width1) ** 2
    return amp / (1.0 + (1.0 - eta0 + eta0 * u0) * u0 + (1.0 - eta1 + eta1 * u1) * u1)


def lorenzianpolyint(
    amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, eta0=1.0, eta1=1.0, th=0.0
):
    return dblquad(
        lambda y, x: lorenzianpoly(
            y, x, amp, loc0, loc1, width0, width1, eta0, eta1, th
        ),
        -np.inf,
        np.inf,
        lambda x: -np.inf,
        lambda x: np.inf,
    )


"""*****************************************************************************************************"""


def lorenziandoor2D(
    x, amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, eta=1.0, th=0.0
):
    cth = np.cos(th)
    sth = np.sin(th)
    x0 = x[0] - loc0
    x1 = x[1] - loc1
    x0, x1 = x0 * cth - x1 * sth, x0 * sth + x1 * cth  # rotation
    return (
        amp
        / (1.0 + ((x0 / width0) ** 2))
        * (
            erf((x1 + width1 / 2.0) / (eta * width1))
            - erf((x1 - width1 / 2.0) / (eta * width1))
        )
        / 2.0
    )


def lorenziandoorint(
    amp=1.0, loc0=0.0, loc1=0.0, width0=1.0, width1=1.0, eta=1.0, th=0.0
):
    return (amp * np.pi * width0 * width1, 0.0)


"""*****************************************************************************************************"""


def nointegration(*args, **kwargs):
    return (0.0, 0.0)


# we jsut add the possibility to compute integrals for modelss
class Model(LmfitModel):
    def __init__(
        self,
        func,
        independent_vars=None,
        param_names=None,
        nan_policy="raise",
        missing=None,
        prefix="",
        name=None,
        **kws
    ):
        self.intfunc = nointegration
        super(Model, self).__init__(
            func=func,
            independent_vars=independent_vars,
            param_names=param_names,
            nan_policy=nan_policy,
            missing=missing,
            prefix=prefix,
            name=name,
            **kws
        )

    def __add__(self, other):
        """+"""
        return CompositeModel(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeModel(self, other, operator.mul)

    def __div__(self, other):
        """/"""
        return CompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        """/"""
        return CompositeModel(self, other, operator.truediv)

    def eval_integral(self, params=None, **kwargs):
        # evaluate the integral of the peak, and associated scipy.integrate error (not the fit uncertainty)
        # needs to define self.intfunc first!
        return self.intfunc(**self.make_funcargs(params, kwargs))

    def eval_component_integrals(self, params=None, **kwargs):
        """Evaluate the model integrals with the supplied parameters.

        Parameters
        -----------
        params : Parameters, optional
            Parameters to use in Model.
        **kwargs : optional
            Additional keyword arguments to pass to model function.

        Returns
        -------
        OrderedDict
            Keys are prefixes for component model, values are value of each
            component.

        """
        key = self._prefix
        if len(key) < 1:
            key = self._name
        return {
            key: self.eval_integral(params=params, **kwargs)
        }  # will be used to update the dictionnary of a composite model


class CompositeModel(Model):
    """Combine two models (`left` and `right`) with a binary operator (`op`)
    into a CompositeModel.

    Normally, one does not have to explicitly create a `CompositeModel`,
    but can use normal Python operators `+`, '-', `*`, and `/` to combine
    components as in::

    >>> mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

    """

    _names_collide = (
        "\nTwo models have parameters named '{clash}'. " "Use distinct names."
    )
    _bad_arg = "CompositeModel: argument {arg} is not a Model"
    _bad_op = "CompositeModel: operator {op} is not callable"
    _known_ops = {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
    }

    def __init__(self, left, right, op, **kws):
        """
        Parameters
        ----------
        left : Model
            Left-hand model.
        right : Model
            Right-hand model.
        op : callable binary operator
            Operator to combine `left` and `right` models.
        **kws : optional
            Additional keywords are passed to `Model` when creating this
            new model.

        Notes
        -----
        1. The two models must use the same independent variable.

        """
        if not isinstance(left, Model):
            raise ValueError(self._bad_arg.format(arg=left))
        if not isinstance(right, Model):
            raise ValueError(self._bad_arg.format(arg=right))
        if not callable(op):
            raise ValueError(self._bad_op.format(op=op))

        self.left = left
        self.right = right
        self.op = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ""
            for collision in name_collisions:
                msg += self._names_collide.format(clash=collision)
            raise NameError(msg)

        # we assume that all the sub-models have the same independent vars
        if "independent_vars" not in kws:
            kws["independent_vars"] = self.left.independent_vars
        if "nan_policy" not in kws:
            kws["nan_policy"] = self.left.nan_policy

        def _tmp(self, *args, **kws):
            pass

        Model.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints["%s%s" % (prefix, basename)] = hint

    def _parse_params(self):
        self._func_haskeywords = (
            self.left._func_haskeywords or self.right._func_haskeywords
        )
        self._func_allargs = self.left._func_allargs + self.right._func_allargs
        self.def_vals = deepcopy(self.right.def_vals)
        self.def_vals.update(self.left.def_vals)
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=False):
        return "(%s %s %s)" % (
            self.left._reprstring(long=long),
            self._known_ops.get(self.op, self.op),
            self.right._reprstring(long=long),
        )

    def eval(self, params=None, **kwargs):
        """TODO: docstring in public method."""
        return self.op(
            self.left.eval(params=params, **kwargs),
            self.right.eval(params=params, **kwargs),
        )

    def eval_components(self, **kwargs):
        """Return OrderedDict of name, results for each component."""
        out = OrderedDict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    def eval_component_integrals(self, params=None, **kwargs):
        """Return OrderedDict of name, results for each component."""
        out = OrderedDict(self.left.eval_component_integrals(params, **kwargs))
        out.update(self.right.eval_component_integrals(params, **kwargs))
        return out

    @property
    def param_names(self):
        """Return parameter names for composite model."""
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """Return components for composite model."""
        return self.left.components + self.right.components

    def _get_state(self):
        return (self.left._get_state(), self.right._get_state(), self.op.__name__)

    def _set_state(self, state, funcdefs=None):
        return _buildmodel(state, funcdefs=funcdefs)

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out


def _buildmodel(state, funcdefs=None):
    """Build model from saved state.

    Intended for internal use only.

    """
    if len(state) != 3:
        raise ValueError("Cannot restore Model")
    known_funcs = {}
    for fname in lineshapes.functions:
        fcn = getattr(lineshapes, fname, None)
        if callable(fcn):
            known_funcs[fname] = fcn
    if funcdefs is not None:
        known_funcs.update(funcdefs)

    left, right, op = state
    if op is None and right is None:
        (fname, fcndef, name, prefix, ivars, pnames, phints, nan_policy, opts) = left
        if not callable(fcndef) and fname in known_funcs:
            fcndef = known_funcs[fname]

        if fcndef is None:
            raise ValueError("Cannot restore Model: model function not found")

        model = Model(
            fcndef,
            name=name,
            prefix=prefix,
            independent_vars=ivars,
            param_names=pnames,
            nan_policy=nan_policy,
            **opts
        )

        for name, hint in phints.items():
            model.set_param_hint(name, **hint)
        return model
    else:
        lmodel = _buildmodel(left, funcdefs=funcdefs)
        rmodel = _buildmodel(right, funcdefs=funcdefs)
        return CompositeModel(lmodel, rmodel, getattr(operator, op))


# Models derived from the lmfit Model class
# Need to give param_names if we want to keep the correct order


class LorenzianPower2DModel(Model):
    """Lorentzian model with variable power, with seven Parameters.

    Defined as:

    f(x; amp, loc0, loc1, width0, width1, th) = amp/(1.+((x0/width0)**2+(x1/width1)**2)**(power/2.)))

    """

    EVALUATE = "peak"
    ICON = "lorenzpower.jpg"
    NAME = "Lorenzian with extra power"

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super(LorenzianPower2DModel, self).__init__(
            lorenzianpower2D,
            param_names=["amp", "loc0", "loc1", "width0", "width1", "th", "power"],
            **kwargs
        )
        self.set_param_hint("amp", value=1.0)
        self.set_param_hint("loc0", value=0.0)
        self.set_param_hint("loc1", value=0.0)
        self.set_param_hint("width0", value=1.0, min=0)
        self.set_param_hint("width1", value=1.0, min=0)
        self.set_param_hint("th", value=0.0, vary=False)
        self.set_param_hint("power", value=2.0, vary=False)
        self.intfunc = lorenzianpowerint

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        amp, loc0, loc1, width0, width1, th, power = 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0

        if x is not None:
            n = np.argmax(data)
            amp = data[n]
            loc0 = x[0][n]
            loc1 = x[1][n]
            width0 = (np.amax(x) - np.amin(x)) / 2.0
            width1 = (np.amax(x) - np.amin(x)) / 2.0
            if width0 == 0:
                width0 = 1.0
            if width1 == 0:
                width1 = 1.0

        pars = self.make_params(
            amp=amp,
            loc0=loc0,
            loc1=loc1,
            width0=width0,
            width1=width1,
            th=th,
            power=power,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class LorenzianPoly2DModel(Model):
    """Lorentzian model with polynomial, with seven Parameters.

    Defined as:

    f(x; amp, loc0, loc1, width0, width1, th) = amp/(1.+((x0/width0)**2+(x1/width1)**2)**(power/2.)))

    """

    EVALUATE = "peak"
    ICON = "lorenzpoly.jpg"
    NAME = "Lorenzian with x**4 term"

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super(LorenzianPoly2DModel, self).__init__(
            lorenzianpoly2D,
            param_names=[
                "amp",
                "loc0",
                "loc1",
                "width0",
                "width1",
                "eta0",
                "eta1",
                "th",
            ],
            **kwargs
        )
        self.set_param_hint("amp", value=1.0)
        self.set_param_hint("loc0", value=0.0)
        self.set_param_hint("loc1", value=0.0)
        self.set_param_hint("width0", value=1.0, min=0)
        self.set_param_hint("width1", value=1.0, min=0)
        self.set_param_hint("eta0", value=0.5, min=0, max=1)
        self.set_param_hint("eta1", value=0.5, min=0, max=1)
        self.set_param_hint("th", value=0.0, min=-np.pi, max=np.pi)
        self.intfunc = lorenzianpolyint

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        amp, loc0, loc1, width0, width1, th, power = 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0

        if x is not None:
            n = np.argmax(data)
            amp = data[n]
            loc0 = x[0][n]
            loc1 = x[1][n]
            width0 = (np.amax(x) - np.amin(x)) / 2.0
            width1 = (np.amax(x) - np.amin(x)) / 2.0
            if width0 == 0:
                width0 = 1.0
            if width1 == 0:
                width1 = 1.0

        pars = self.make_params(
            amp=amp,
            loc0=loc0,
            loc1=loc1,
            width0=width0,
            width1=width1,
            th=th,
            power=power,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    def eval_integral(self, params=None, **kwargs):
        """Evaluate the integral of the model with supplied parameters and keyword arguments.

        Parameters
        -----------
        params : Parameters, optional
            Parameters to use in Model.
        **kwargs : optional
            Additional keyword arguments to pass to model function.

        Returns
        -------
        numpy.ndarray
            Value of model given the parameters and other arguments.

        Notes
        -----
        1. if `params` is None, the values for all parameters are
        expected to be provided as keyword arguments.  If `params` is
        given, and a keyword argument for a parameter value is also given,
        the keyword argument will be used.

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        """
        return self.intfunc(**self.make_funcargs(params, kwargs))

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class LorenzianDoorModel(Model):
    """Lorentzian in one direction, door (with 2 erf) in the other direction, with seven Parameters.

    Defined as:

    f(x; amp, loc0, loc1, width0, width1, eta, th) = (amp/(1.+((x0/width0)**2)*(erf((y0-width0/2)/eta)-erf((y0+width0/2)))

    """

    EVALUATE = "peak"
    ICON = "lorenzgauss.jpg"
    NAME = "Lorenzian/door"

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super(LorenzianDoorModel, self).__init__(
            lorenziandoor2D,
            param_names=["amp", "loc0", "loc1", "width0", "width1", "eta", "th"],
            **kwargs
        )
        self.set_param_hint("amp", value=1.0)
        self.set_param_hint("loc0", value=0.0)
        self.set_param_hint("loc1", value=0.0)
        self.set_param_hint("width0", value=1.0, min=0)
        self.set_param_hint("width1", value=1.0, min=0)
        self.set_param_hint("eta", value=0.2, min=0)
        self.set_param_hint("th", value=0.0, min=-np.pi, max=np.pi)
        self.intfunc = lorenziandoorint

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        amp, loc0, loc1, width0, width1, eta, th = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0

        if x is not None:
            n = np.argmax(data)
            amp = data[n]
            loc0 = x[0][n]
            loc1 = x[1][n]
            width0 = (np.amax(x) - np.amin(x)) / 2.0
            width1 = (np.amax(x) - np.amin(x)) / 2.0
            if width0 == 0:
                width0 = 1.0
            if width1 == 0:
                width1 = 1.0

        pars = self.make_params(
            amp=amp, loc0=loc0, loc1=loc1, width0=width0, width1=width1, eta=eta, th=th
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class LorenzianGaussModel(Model):
    """Lorentzian in one direction, gaussian in the other direction, with six Parameters.

    Defined as:

    f(x; amp, loc0, loc1, width0, width1, th) = (amp/(1.+((x0/width0)**2)*np.exp(-(x1/width1)**2)))

    """

    EVALUATE = "peak"
    ICON = "lorenzgauss.jpg"
    NAME = "Lorenzian/gaussian"

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super(LorenzianGaussModel, self).__init__(
            lorenziangauss2D,
            param_names=["amp", "loc0", "loc1", "width0", "width1", "th"],
            **kwargs
        )
        self.set_param_hint("amp", value=1.0)
        self.set_param_hint("loc0", value=0.0)
        self.set_param_hint("loc1", value=0.0)
        self.set_param_hint("width0", value=1.0, min=0)
        self.set_param_hint("width1", value=1.0, min=0)
        self.set_param_hint("th", value=0.0, min=-np.pi, max=np.pi)
        self.intfunc = lorenziangaussint

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        amp, loc0, loc1, width0, width1, th = 1.0, 0.0, 0.0, 1.0, 1.0, 0.0

        if x is not None:
            n = np.argmax(data)
            amp = data[n]
            loc0 = x[0][n]
            loc1 = x[1][n]
            width0 = (np.amax(x) - np.amin(x)) / 2.0
            width1 = (np.amax(x) - np.amin(x)) / 2.0
            if width0 == 0:
                width0 = 1.0
            if width1 == 0:
                width1 = 1.0

        pars = self.make_params(
            amp=amp, loc0=loc0, loc1=loc1, width0=width0, width1=width1, th=th
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class Linear2DModel(Model):
    """Lorentzian model, with six Parameters.

    Defined as:

    f(x; slope_x,slope_y,bg) = amp/(1.+(x0/width0)**2+(x1/width1)**2)

    """

    EVALUATE = "points"
    ICON = "planar.jpg"
    NAME = "Linear background"

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super(Linear2DModel, self).__init__(
            linear2D, param_names=["bg", "slope_x", "slope_y"], **kwargs
        )
        self.set_param_hint("bg", value=1.0)
        self.set_param_hint("slope_x", value=0.0)
        self.set_param_hint("slope_y", value=0.0)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        slope_x, slope_y, bg = 0.0, 0.0, 0.0

        if x is not None:
            sx2 = np.sum(x[0] * x[0])
            sy2 = np.sum(x[1] * x[1])
            sxy = np.sum(x[0] * x[1])
            sx = np.sum(x[0])
            sy = np.sum(x[0])
            szx = np.sum(x[0] * data)
            szy = np.sum(x[0] * data)
            sz = np.sum(data)
            npts = x[0].shape[0]
            a = np.array([[sx2, sxy, sx], [sx, sy2, sy], [sx, sy, npts]])
            b = np.array([szx, szy, sz])
            res = np.linalg.solve(a, b)
            slope_x = res[0]
            slope_y = res[1]
            bg = res[2]
        pars = self.make_params(slope_x=slope_x, slope_y=slope_y, bg=bg)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


list_of_models = []
list_of_2D_models = [
    Linear2DModel,
    LorenzianGaussModel,
    LorenzianPoly2DModel,
    LorenzianPower2DModel,
    LorenzianDoorModel,
]
