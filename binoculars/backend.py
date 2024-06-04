from . import util, errors, dispatcher


class ProjectionBase(util.ConfigurableObject):
    def parse_config(self, config):
        super(ProjectionBase, self).parse_config(config)
        res = config.pop("resolution")  # or just give 1 number for

        # all dimensions Optional, set the limits of the space object
        # in projected coordinates. Syntax is same as numpy
        # e.g. 'limits = [:0,-1:,:], [0:,:-1,:], [:0,:-1,:],
        # [0:,-1:,:]'
        self.config.limits = util.parse_pairs(config.pop("limits", None))
        labels = self.get_axis_labels()
        if self.config.limits is not None:
            for lim in self.config.limits:
                if len(lim) != len(labels):
                    raise errors.ConfigError(
                        f"dimension mismatch between projection"
                        f" axes ({labels})"
                        f" and limits specification"
                        f" ({self.config.limits}) in {self.__class__.__name__}"
                    )
        if "," in res:
            self.config.resolution = util.parse_tuple(res, type=float)
            if not len(labels) == len(self.config.resolution):
                raise errors.ConfigError(
                    f"dimension mismatch between projection"
                    f" axes ({labels})"
                    f" and resolution specification"
                    f" ({self.config.resolution}) in {self.__class__.__name__}"
                )
        else:
            self.config.resolution = tuple([float(res)] * len(labels))

    def project(self, *args):
        raise NotImplementedError

    def get_axis_labels(self):
        raise NotImplementedError


class Job(object):
    weight = 1.0  # estimate of job difficulty (arbitrary units)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class InputBase(util.ConfigurableObject):
    """Generate and process Job()s.

    Note: there is no guarantee that generate_jobs() and process_jobs() will
    be called on the same instance, not even in the same process or on the
    same computer!"""

    def parse_config(self, config):
        super(InputBase, self).parse_config(config)
        # approximate number of images per job, only useful when
        # running on the oar cluster
        self.config.target_weight = int(config.pop("target_weight", 1000))

    def generate_jobs(self, command):
        """Receives command from user, yields Job() instances"""
        raise NotImplementedError

    def process_job(self, job):
        """Receives a Job() instance, yields (intensity,
args_to_be_sent_to_a_Projection_instance)

        Job()s could have been pickle'd and distributed over a cluster

        """
        self.metadata = util.MetaBase("job", job.__dict__)

    def get_destination_options(self, command):
        """Receives the same command as generate_jobs(), but returns
        dictionary that will be used to .format() the dispatcher:destination
        configuration value."""
        return {}


def get_dispatcher(config, main, default=None):
    return _get_backend(
        config, "dispatcher", dispatcher.DispatcherBase, default=default,
        args=[main]
    )


def get_input(config, default=None):
    return _get_backend(config, "input", InputBase, default=default)


def get_projection(config, default=None):
    return _get_backend(config, "projection", ProjectionBase, default=default)


def _get_backend(config, section, basecls, default=None, args=[], kwargs={}):
    if isinstance(config, util.ConfigSection):
        return config.class_(config, *args, **kwargs)

    type = config.pop("type", default)
    if type is None:
        raise errors.ConfigError(
            f"required option 'type' not given in section '{section}'"
        )
    type = type.strip()

    if ":" in type:
        try:
            modname, clsname = type.split(":")
        except ValueError:
            raise errors.ConfigError(
                f"invalid type '{type}' in section '{section}'"
            )
        try:
            backend = __import__(
                f"backends.{modname}", globals(), locals(), [], 1
            )
        except ImportError as e:
            raise errors.ConfigError(
                f"unable to import module backends.{modname}: {e}"
            )
        module = getattr(backend, modname)
    elif section == "dispatcher":
        module = dispatcher
        clsname = type
    else:
        raise errors.ConfigError(
            f"invalid type '{type}' in section '{section}'"
        )

    clsname = clsname.lower()
    names = dict((name.lower(), name) for name in dir(module))
    if clsname in names:
        cls = getattr(module, names[clsname])

        if issubclass(cls, basecls):
            return cls(config, *args, **kwargs)
        else:
            raise errors.ConfigError(
                f"type '{type}' not compatible in section '{section}':"
                f" expected class derived from '{basecls.__name__}',"
                f" got '{cls.__name__}'"
            )
    else:
        raise errors.ConfigError(
            f"invalid type '{type}' in section '{section}'"
        )
