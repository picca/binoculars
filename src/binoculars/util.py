import argparse
import binascii
import configparser
import contextlib
import copy
import glob
import gzip
import inspect
import io
import itertools
import json
import os
import pickle
import random
import re
import struct
import socket
import sys
import time

import h5py
import numpy

from ast import literal_eval

from PyMca5.PyMca import EdfFile

from . import errors

### ARGUMENT HANDLING


def as_string(text):
    if hasattr(text, "decode"):
        text = text.decode()
    return text


class OrderedOperation(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        oops = getattr(namespace, "ordered_operations", [])
        oops.append((self.dest, values))
        setattr(namespace, "ordered_operations", oops)


def argparse_common_arguments(parser, *args):
    for arg in args:
        # (ORDERED) OPERATIONS
        if arg == "project":
            parser.add_argument(
                "-p",
                "--project",
                metavar="AXIS",
                action=OrderedOperation,
                help="project space on AXIS",
            )
        elif arg == "slice":
            parser.add_argument(
                "--slice",
                nargs=2,
                metavar=("AXIS", "START:STOP"),
                action=OrderedOperation,
                help="slice AXIS from START to STOP (replace minus signs by 'm')",
            )
        elif arg == "pslice":
            parser.add_argument(
                "--pslice",
                nargs=2,
                metavar=("AXIS", "START:STOP"),
                action=OrderedOperation,
                help="like slice, but also project on AXIS after slicing",
            )
        elif arg == "transform":
            parser.add_argument(
                "--transform",
                metavar="VAR@RES=EXPR;VAR2@RES2=EXPR2;...",
                action=OrderedOperation,
                help="perform coordinate transformation, rebinning data on new axis named VAR with resolution RES defined by EXPR, example: Q@0.1=sqrt(H**2+K**2+L**2)",
            )
        elif arg == "rebin":
            parser.add_argument(
                "--rebin",
                metavar="N,M,...",
                action=OrderedOperation,
                help="reduce binsize by factor N in first dimension, M in second, etc",
            )

        # SUBTRACT
        elif arg == "subtract":
            parser.add_argument(
                "--subtract", metavar="SPACE", help="subtract SPACE from input file"
            )

        # PRESENTATION
        elif arg == "nolog":
            parser.add_argument(
                "--nolog", action="store_true", help="do not use logarithmic axis"
            )
        elif arg == "clip":
            parser.add_argument(
                "-c",
                "--clip",
                metavar="FRACTION",
                default=0.00,
                help="clip color scale to remove FRACTION datapoints",
            )

        # OUTPUT
        elif arg == "savepdf":
            parser.add_argument(
                "-s",
                "--savepdf",
                action="store_true",
                help="save output as pdf, automatic file naming",
            )
        elif arg == "savefile":
            parser.add_argument(
                "--savefile",
                metavar="FILENAME",
                help="save output as FILENAME, autodetect filetype",
            )

        # ERROR!
        else:
            raise ValueError(f"unsupported argument '{arg}'")


def parse_transform_args(transform):
    for t in transform.split(";"):
        lhs, expr = t.split("=")
        ax, res = lhs.split("@")
        yield ax.strip(), float(res), expr.strip()


def handle_ordered_operations(space, args, auto3to2=False):
    info = []
    for command, opts in getattr(args, "ordered_operations", []):

        if command == "slice" or command == "pslice":
            ax, key = opts
            axindex = space.axes.index(ax)
            axlabel = space.axes[axindex].label
            if ":" in key:
                start, stop = key.split(":")
                if start:
                    start = float(start.replace("m", "-"))
                else:
                    start = space.axes[axindex].min
                if stop:
                    stop = float(stop.replace("m", "-"))
                else:
                    stop = space.axes[axindex].max
                key = slice(start, stop)

                info.append(f"sliced in {axlabel} from {start} to {stop}")
            else:
                key = float(key.replace("m", "-"))
                info.append(f"sliced in {axlabel} at {key}")
            space = space.slice(axindex, key)

            if command == "pslice":
                try:
                    projectaxis = space.axes.index(ax)
                except ValueError:
                    pass
                else:
                    info.append(
                        f"projected on {space.axes[projectaxis].label}."
                    )
                    space = space.project(projectaxis)

        elif command == "project":
            projectaxis = space.axes.index(opts)
            info.append(f"projected on {space.axes[projectaxis].label}")
            space = space.project(projectaxis)

        elif command == "transform":
            labels, resolutions, exprs = list(zip(*parse_transform_args(opts)))
            transformation = transformation_from_expressions(space, exprs)
            expression = ", ".join(f"{label} = {expr}"
                                   for (label, expr) in zip(labels, exprs))
            info.append(f"transformed to {expression}")
            space = space.transform_coordinates(resolutions, labels, transformation)

        elif command == "rebin":
            if "," in opts:
                factors = tuple(int(i) for i in opts.split(","))
            else:
                factors = (int(opts),) * space.dimension
            space = space.rebin(factors)

        else:
            raise ValueError(f"unsported Ordered Operation '{command}'")

    if auto3to2 and space.dimension == 3:  # automatic projection on smallest axis
        projectaxis = numpy.argmin(space.photons.shape)
        info.append(f"projected on {space.axes[projectaxis].label}")
        space = space.project(projectaxis)

    return space, info


### STATUS LINES

_status_line_length = 0


def status(line, eol=False):
    """Prints a status line to sys.stdout, overwriting the previous one.
    Set eol to True to append a newline to the end of the line"""

    global _status_line_length
    sys.stdout.write(f"\r{' ' * _status_line_length}\r{line}")
    if eol:
        sys.stdout.write("\n")
        _status_line_length = 0
    else:
        _status_line_length = len(line)

    sys.stdout.flush()


def statusnl(line):
    """Shortcut for status(..., eol=True)"""
    return status(line, eol=True)


def statuseol():
    """Starts a new status line, keeping the previous one intact"""
    global _status_line_length
    _status_line_length = 0
    sys.stdout.write("\n")
    sys.stdout.flush()


def statuscl():
    """Clears the status line, shortcut for status('')"""
    return status("")


### Dispatcher, projection and input finder
def get_backends():
    modules = glob.glob(os.path.join(os.path.dirname(__file__), "backends", "*.py"))
    names = list()

    for module in modules:
        if not module.endswith("__init__.py"):
            names.append(os.path.splitext(os.path.basename(module))[0])
    return names


def get_projections(module):
    from . import backend

    return get_base(module, backend.ProjectionBase)


def get_inputs(module):
    from . import backend

    return get_base(module, backend.InputBase)


def get_dispatchers():
    from . import dispatcher
    from inspect import isclass

    items = dir(dispatcher)

    options = []
    for item in items:
        obj = getattr(dispatcher, item)
        if isclass(obj):
            if issubclass(obj, dispatcher.DispatcherBase):
                options.append(item)

    return options


def get_base(modname, base):
    from inspect import isclass

    if modname not in get_backends():
        raise KeyError(f"{modname} is not an available backend")
    try:
        backends = __import__(
            f"backends.{modname}", globals(), locals(), [], 1
        )
    except ImportError as e:
        raise ImportError(
            f"Unable to import module backends.{modname}: {e!r}"
        )

    backend = getattr(backends, modname)
    items = dir(backend)

    options = []
    for item in items:
        obj = getattr(backend, item)
        if isclass(obj):
            if issubclass(obj, base):
                options.append(item)
    return options


### Dispatcher, projection and input configuration options finder


def get_dispatcher_configkeys(classname):
    from . import dispatcher

    cls = getattr(dispatcher, classname)
    return get_configkeys(cls)


def get_projection_configkeys(modname, classname):
    return get_backend_configkeys(modname, classname)


def get_input_configkeys(modname, classname):
    return get_backend_configkeys(modname, classname)


def get_backend_configkeys(modname, classname):
    backends = __import__(f"backends.{modname}", globals(), locals(), [], 1)
    backend = getattr(backends, modname)
    cls = getattr(backend, classname)
    return get_configkeys(cls)


def get_configkeys(cls):
    from inspect import getsource

    items = list()
    while hasattr(cls, "parse_config"):
        code = getsource(cls.parse_config)
        for line in code.split("\n"):
            key = parse_configcode(line)
            if key:
                if key not in items:
                    items.append(key)
        cls = cls.__base__
    return items


def parse_configcode(line):
    try:
        comment = "#".join(line.split("#")[1:])
        line = line.split("#")[0]
        index = line.index("config.pop")
        item = line[index:].split("'")[1]
        if item == "action":
            return  # action is reserved for internal use!
        return item, comment
    except ValueError:
        pass


### CONFIGURATION MANAGEMENT


def parse_float(config, option, default=None):
    value = default
    value_as_string = config.pop(option, None)
    if value_as_string is not None:
        try:
            value = float(value_as_string)  # noqa
        except ValueError:
            pass
    return value


def parse_range(r):
    if "-" in r:
        a, b = r.split("-")
        return list(range(int(a), int(b) + 1))
    elif r:
        return [int(r)]
    else:
        return []


def parse_multi_range(s):
    if not s:
        return s
    out = []
    ranges = s.split(",")
    for r in ranges:
        out.extend(parse_range(r))
    return out


def parse_tuple(s, length=None, type=str):
    if not s:
        return s
    t = tuple(type(i) for i in s.split(","))
    if length is not None and len(t) != length:
        raise ValueError(
            f"invalid tuple length: expected {length} got {len(t)}"
        )
    return t


def parse_bool(s):
    l = s.lower()
    if l in ("1", "true", "yes", "on"):
        return True
    elif l in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid input for boolean: '{s}'")


def parse_pairs(s):
    if not s:
        return s
    limits = []
    for lim in re.findall(r"\[(.*?)\]", s):
        parsed = []
        for pair in re.split(",", lim):
            mi, ma = tuple(m.strip() for m in pair.split(":"))
            if mi == "" and ma == "":
                parsed.append(slice(None))
            elif mi == "":
                parsed.append(slice(None, float(ma)))
            elif ma == "":
                parsed.append(slice(float(mi), None))
            else:
                if float(ma) < float(mi):
                    raise ValueError(
                        f"invalid input. maximum is larger than minimum: '{s}'"
                    )
                else:
                    parsed.append(slice(float(mi), float(ma)))
        limits.append(parsed)
    return limits


def parse_dict(config, option, default=None) -> dict:
    value = default
    value_as_string = config.pop(option, None)
    if value_as_string is not None:
        try:
            value = literal_eval(value_as_string)  # noqa
        except ValueError:
            pass
    return value


def limit_to_filelabel(s):
    return tuple(
        f"[{lim.replace('-', 'm').replace(':', '-').replace(' ', '')}]"
        for lim in re.findall(r"\[(.*?)\]", s)
    )


class MetaBase:
    def __init__(self, label=None, section=None):
        self.sections = []
        if label is not None and section is not None:
            self.sections.append(label)
            setattr(self, label, section)
        elif label is not None:
            self.sections.append(label)
            setattr(self, label, dict())

    def add_section(self, label, section=None):
        self.sections.append(label)
        if section is not None:
            setattr(self, label, section)
        else:
            setattr(self, label, dict())

    def __repr__(self):
        str = f"{self.__class__.__name__}{{\n"
        for section in self.sections:
            str += f"  [{section}]\n"
            s = getattr(self, section)
            for entry in s:
                str += f"    {entry} = {s[entry]}\n"
        str += "}\n"
        return str

    def copy(self):
        return copy.deepcopy(self)

    def serialize(self):
        sections = {}
        for section in self.sections:
            section_dict = {}
            attr = getattr(self, section)
            for key in list(attr.keys()):
                if isinstance(
                    attr[key], numpy.ndarray
                ):  # to be able to include numpy arrays in the serialisation
                    sio = io.StringIO()
                    numpy.save(sio, attr[key])
                    sio.seek(0)
                    section_dict[key] = binascii.b2a_hex(
                        sio.read()
                    )  # hex codation is needed to let json work with the string
                else:
                    section_dict[key] = attr[key]
            sections[section] = section_dict
        return json.dumps(sections)

    @classmethod
    def fromserial(cls, s):
        obj = cls()
        data = json.loads(s)
        for section in list(data.keys()):
            section_dict = data[section]
            for key in list(section_dict.keys()):
                if isinstance(
                    section_dict[key], str
                ):  # find and replace all the numpy serialised objects
                    if section_dict[key].startswith(
                        "934e554d505901004600"
                    ):  # numpy marker
                        sio = io.StringIO()
                        sio.write(binascii.a2b_hex(section_dict[key]))
                        sio.seek(0)
                        section_dict[key] = numpy.load(sio)
            setattr(obj, section, data[section])
            if section not in obj.sections:
                obj.sections.append(section)
        return obj


class MetaData:  # a collection of metadata objects
    def __init__(self):
        self.metas = []

    def add_dataset(self, dataset):
        if not isinstance(dataset, MetaBase) and not isinstance(dataset, ConfigFile):
            raise ValueError("MetaBase instance expected")
        else:
            self.metas.append(dataset)

    def __add__(self, other):
        new = self.__class__()
        new += self
        new += other
        return new

    def __iadd__(self, other):
        self.metas.extend(other.metas)
        return self

    @classmethod
    def fromfile(cls, filename):
        if isinstance(filename, str):
            if not os.path.exists(filename):
                raise OSError(
                    "Error importing configuration file."
                    f" filename {filename} does not exist"
                )

        metadataobj = cls()
        with open_h5py(filename, "r") as fp:
            try:
                metadata = fp["metadata"]
            except KeyError:
                metadata = []  # when metadata is not present, proceed without Error
            for label in metadata:
                meta = MetaBase()
                for section in list(metadata[label].keys()):
                    group = metadata[label][section]
                    setattr(
                        meta, section, {key: group[key][()] for key in group}
                    )
                    meta.sections.append(section)
                metadataobj.metas.append(meta)
        return metadataobj

    def tofile(self, filename):
        with open_h5py(filename, "w") as fp:
            metadata = fp.create_group("metadata")
            for meta in self.metas:
                label = find_unused_label("metasection", list(metadata.keys()))
                metabase = metadata.create_group(label)
                for section in meta.sections:
                    sectiongroup = metabase.create_group(section)
                    s = getattr(meta, section)
                    for key in list(s.keys()):
                        sectiongroup.create_dataset(key, data=s[key])

    def __repr__(self):
        str = f"{self.__class__.__name__}{{\n"
        for meta in self.metas:
            for line in meta.__repr__().split("\n"):
                str += "    " + line + "\n"
        str += "}\n"
        return str

    def serialize(self):
        return json.dumps(list(meta.serialize() for meta in self.metas))

    @classmethod
    def fromserial(cls, s):
        obj = cls()
        for item in json.loads(s):
            obj.metas.append(MetaBase.fromserial(item))
        return obj


# Contains the unparsed config dicts


class ConfigFile(MetaBase):
    def __init__(self, origin="n/a", command=[]):
        self.origin = origin
        self.command = command
        super().__init__()
        self.sections = ["dispatcher", "projection", "input"]
        for section in self.sections:
            setattr(self, section, dict())

    @classmethod
    def fromfile(cls, filename):
        if isinstance(filename, str):
            if not os.path.exists(filename):
                raise OSError(
                    "Error importing configuration file."
                    f" filename {filename} does not exist"
                )

        configobj = cls(str(filename))
        with open_h5py(filename, "r") as fp:
            try:
                config = fp["configuration"]
                if "command" in config.attrs:
                    configobj.command = json.loads(as_string(config.attrs["command"]))
                for section in config:
                    if isinstance(config[section], h5py._hl.group.Group):  # new
                        setattr(
                            configobj,
                            section,
                            {
                                key: config[section][key][()]
                                for key in config[section]
                            },
                        )
                    else:  # old
                        setattr(configobj, section, dict(config[section]))
            except KeyError:
                pass  # when config is not present, proceed without Error
        return configobj

    @classmethod
    def fromtxtfile(cls, filename, command=[], overrides=[]):
        if not os.path.exists(filename):
            raise OSError(
                "Error importing configuration file."
                f" filename {filename} does not exist"
            )

        config = configparser.RawConfigParser()
        config.read(filename)

        for section, option, value in overrides:
            config.set(section, option, value)

        configobj = cls(filename, command=command)
        for section in configobj.sections:
            setattr(
                configobj,
                section,
                {k: v.split("#")[0].strip() for (k, v) in config.items(section)},
            )
        return configobj

    def tofile(self, filename):
        with open_h5py(filename, "w") as fp:
            conf = fp.create_group("configuration")
            conf.attrs["origin"] = str(self.origin)
            conf.attrs["command"] = json.dumps(self.command)
            for section in self.sections:
                sectiongroup = conf.create_group(section)
                s = getattr(self, section)
                for key in list(s.keys()):
                    sectiongroup.create_dataset(key, data=s[key])

    def totxtfile(self, filename):
        with open(filename, "w") as fp:
            fp.write(f"# Configurations origin: {self.origin}\n")
            for section in self.sections:
                fp.write(f"[{section}]\n")
                s = getattr(self, section)
                for entry in s:
                    fp.write(f"{entry} = {s[entry]}\n")

    def __repr__(self):
        str = super().__repr__()
        str += f"origin = {self.origin}\n"
        str += f"command = {','.join(self.command)}"
        return str


# contains one parsed dict, for distribution to dispatcher, input or projection class


class ConfigSection:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self):
        return copy.deepcopy(self)


# contains the parsed configsections


class ConfigSectionGroup:
    def __init__(self, origin="n/a"):
        self.origin = origin
        self.sections = "dispatcher", "projection", "input"
        for section in self.sections:
            setattr(self, section, ConfigSection())
        self.configfile = ConfigFile()


class ConfigurableObject:
    def __init__(self, config):
        if isinstance(config, ConfigSection):
            self.config = config
        elif not isinstance(config, dict):
            raise ValueError(
                f"expecting dict or Configsection, not: {type(config)}"
            )
        else:
            self.config = ConfigSection()
            try:
                allkeys = list(config.keys())
                self.parse_config(config)
            except KeyError as exc:
                raise errors.ConfigError(
                    f"Configuration option {exc} is missing"
                    " from the configuration file."
                    " Please specify this option in the configuration file"
                )
            except Exception as exc:
                missing = {
                    key
                    for key in allkeys
                    if key not in list(self.config.__dict__.keys())
                } - set(config.keys())
                exc.args = errors.addmessage(
                    exc.args,
                    ". Unable to parse configuration option"
                    f" '{','.join(missing)}'."
                    " The error can quite likely be solved by modifying"
                    " the option in the configuration file."
                )
                raise
            for k in config:
                print(
                    f"warning: unrecognized configuration option {k}"
                    f" for {self.__class__.__name__}"
                )
            self.config.class_ = self.__class__

    def parse_config(self, config):
        # every known option should be pop()'ed from config, converted to a
        # proper type and stored as property in self.config, for example:
        # self.config.foo = int(config.pop('foo', 1))
        pass


### FILES
def best_effort_atomic_rename(src, dest):
    if sys.platform == "win32" and os.path.exists(dest):
        os.remove(dest)
    os.rename(src, dest)


def filename_enumerator(filename, start=0):
    base, ext = os.path.splitext(filename)
    for count in itertools.count(start):
        yield f"{base}_{ext}{count}"


def find_unused_filename(filename):
    if not os.path.exists(filename):
        return filename
    for f in filename_enumerator(filename, 2):
        if not os.path.exists(f):
            return f


def label_enumerator(label, start=0):
    for count in itertools.count(start):
        yield f"{label}_{count}"


def find_unused_label(label, labellist):
    for l in label_enumerator(label):
        if l not in labellist:
            return l


def yield_when_exists(filelist, timeout=None):
    """Wait for files in 'filelist' to appear, for a maximum of 'timeout' seconds,
    yielding them in arbitrary order as soon as they appear.
    If 'filelist' is a set, it will be modified in place, and on timeout it will
    contain the files that have not appeared yet."""
    if not isinstance(filelist, set):
        filelist = set(filelist)
    delay = loop_delayer(5)
    start = time.time()
    while filelist:
        next(delay)
        exists = {f for f in filelist if os.path.exists(f)}
        yield from exists
        filelist -= exists
        if timeout is not None and time.time() - start > timeout:
            break


def wait_for_files(filelist, timeout=None):
    """Wait until the files in 'filelist' have appeared, for a maximum of 'timeout' seconds.
    Returns True on success, False on timeout."""
    filelist = set(filelist)
    for i in yield_when_exists(filelist, timeout):
        pass
    return not filelist


def wait_for_file(file, timeout=None):
    return wait_for_files([file], timeout=timeout)


def space_to_edf(space, filename):
    header = {}
    for a in space.axes:
        header[str(a.label)] = f"{a.min} {a.max} {a.res}"
    edf = EdfFile.EdfFile(filename)
    edf.WriteImage(header, space.get_masked().filled(0), DataType="Float")


def space_to_txt(space, filename):
    data = [coord.flatten() for coord in space.get_grid()]
    data.append(space.get_masked().filled(0).flatten())
    data = numpy.array(data).T

    with open(filename, "w") as fp:
        fp.write("\t".join(ax.label for ax in space.axes))
        fp.write("\tintensity\n")
        numpy.savetxt(fp, data, fmt="%.6g", delimiter="\t")

def space_to_npy(space, filename):
    data = space.get_masked().filled(0)
    numpy.save(filename, data)


@contextlib.contextmanager
def open_h5py(fn, mode):
    if isinstance(fn, h5py._hl.group.Group):
        yield fn
    else:
        with h5py.File(fn, mode) as fp:
            if mode == "w":
                fp.create_group("binoculars")
                yield fp["binoculars"]
            if mode == "r":
                if "binoculars" in fp:
                    yield fp["binoculars"]
                else:
                    yield fp


### VARIOUS


def uniqid():
    return f"{random.randint(0, 2 ** 32 - 1):08x}"


def grouper(iterable, n):
    while True:
        chunk = list(itertools.islice(iterable, n))
        if not chunk:
            break
        yield chunk


_python_executable = None


def register_python_executable(scriptname):
    global _python_executable
    _python_executable = sys.executable, scriptname


def get_python_executable():
    return _python_executable


def chunk_slicer(count, chunksize):
    """yields slice() objects that split an array of length 'count' into equal sized chunks of at most 'chunksize'"""
    chunkcount = int(numpy.ceil(float(count) / chunksize))
    realchunksize = int(numpy.ceil(float(count) / chunkcount))
    for i in range(chunkcount):
        yield slice(i * realchunksize, min(count, (i + 1) * realchunksize))


def cluster_jobs(jobs, target_weight):
    jobs = sorted(jobs, key=lambda job: job.weight)

    # we cannot split jobs here, so just yield away all jobs that are overweight or just right
    while jobs and jobs[-1].weight >= target_weight:
        yield [jobs.pop()]

    while jobs:
        cluster = [jobs.pop()]  # take the biggest remaining job
        size = cluster[0].weight
        for i in range(
            len(jobs) - 1, -1, -1
        ):  # and exhaustively search for all jobs that can accompany it (biggest first)
            if size + jobs[i].weight <= target_weight:
                size += jobs[i].weight
                cluster.append(jobs.pop(i))
        yield cluster


def cluster_jobs2(jobs, target_weight):
    """Taking the first n jobs that together add up to target_weight.
       Here as opposed to cluster_jobs the total number of jobs does not have to be known beforehand
    """
    jobslist = []
    for job in jobs:
        jobslist.append(job)
        if sum(j.weight for j in jobslist) >= target_weight:
            yield jobslist[:]
            jobslist = []
    if len(jobslist) > 0:  # yield the remainder of the jobs
        yield jobslist[:]


def loop_delayer(delay):
    """Delay a loop such that it runs at most once every 'delay' seconds. Usage example:
    delay = loop_delayer(5)
    while some_condition:
        next(delay)
        do_other_tasks
    """

    def generator():
        polltime = 0
        while 1:
            diff = time.time() - polltime
            if diff < delay:
                time.sleep(delay - diff)
            polltime = time.time()
            yield

    return generator()


def transformation_from_expressions(space, exprs):
    def transformation(*coords):
        ns = {i: getattr(numpy, i) for i in dir(numpy)}
        ns.update(**{ax.label: coord for ax, coord in zip(space.axes, coords)})
        return tuple(eval(expr, ns) for expr in exprs)

    return transformation


def format_bytes(bytes):
    units = "kB", "MB", "GB", "TB"
    exp = min(max(int(numpy.log(bytes) / numpy.log(1024.0)), 1), 4)
    return f"{bytes / 1024 ** exp:.1f} {units[exp - 1]}"


### GZIP PICKLING (zpi)

# handle old zpi's
def _pickle_translate(module, name):
    if module in ("__main__", "ivoxoar.space") and name in ("Space", "Axis"):
        return "BINoculars.space", name
    return module, name


if inspect.isbuiltin(pickle.Unpickler):
    # real cPickle: cannot subclass
    def _find_global(module, name):
        module, name = _pickle_translate(module, name)
        __import__(module)
        return getattr(sys.modules[module], name)

    def pickle_load(fileobj):
        unpickler = pickle.Unpickler(fileobj)
        unpickler.find_global = _find_global
        return unpickler.load()


else:
    # pure python implementation
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            module, name = _pickle_translate(module, name)
            return pickle.Unpickler.find_class(self, module, name)

    def pickle_load(fileobj):
        unpickler = _Unpickler(fileobj)
        return unpickler.load()


@contextlib.contextmanager
def atomic_write(filename):
    """Atomically write data into 'filename' using a temporary file and os.rename()

    Rename on success, clean up on failure (any exception).

    Example:
    with atomic_write(filename) as tmpfile
        with open(tmpfile, 'w') as fp:
            fp.write(...)
    """

    if isinstance(filename, h5py._hl.group.Group):
        yield filename
    else:
        tmpfile = f"{os.path.splitext(filename)[0]}-{uniqid()}.tmp"
        try:
            yield tmpfile
        except:
            raise
        else:
            best_effort_atomic_rename(tmpfile, filename)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


def zpi_save(obj, filename):
    with atomic_write(filename) as tmpfile:
        fp = gzip.open(tmpfile, "wb")
        try:
            pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)
        finally:
            fp.close()


def zpi_load(filename):
    if hasattr(filename, "read"):
        fp = gzip.GzipFile(filename.name, fileobj=filename)
    else:
        fp = gzip.open(filename, "rb")
    try:
        return pickle_load(fp)
    finally:
        fp.close()


def serialize(space, command):
    # first 48 bytes contain length of the message, whereby the first 8 give the length of the command, the second 8 the length of the configfile etc..
    message = io.StringIO()
    message.write(struct.pack("QQQQQQ", 0, 0, 0, 0, 0, 0))

    message.write(command)
    commandlength = message.len - 48

    message.write(space.config.serialize())
    configlength = message.len - commandlength - 48

    message.write(space.metadata.serialize())
    metalength = message.len - configlength - commandlength - 48

    numpy.save(message, space.axes.toarray())
    arraylength = message.len - metalength - configlength - commandlength - 48

    numpy.save(message, space.photons)
    photonlength = (
        message.len - arraylength - metalength - configlength - commandlength - 48
    )

    numpy.save(message, space.contributions)
    contributionlength = (
        message.len
        - photonlength
        - arraylength
        - metalength
        - configlength
        - commandlength
        - 48
    )

    message.seek(0)
    message.write(
        struct.pack(
            "QQQQQQ",
            commandlength,
            configlength,
            metalength,
            arraylength,
            photonlength,
            contributionlength,
        )
    )
    message.seek(0)

    return message


def packet_slicer(length, size=1024):  # limit the communication to 1024 bytes
    while length > size:
        length -= size
        yield size
    yield length


def socket_send(ip, port, mssg):
    try:
        mssglengths = struct.unpack(
            "QQQQQQ", mssg.read(48)
        )  # the lengths of all the components
        mssg.seek(0)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))

        sock.send(mssg.read(48))
        for l in mssglengths:
            for packet in packet_slicer(l):
                sock.send(mssg.read(packet))
        sock.close()
    except OSError:  # in case of failure to send. The data will be saved anyway so any loss of communication unfortunate but not critical
        pass


def socket_recieve(RequestHandler):  # pass one the handler to deal with incoming data
    def get_msg(length):
        msg = io.StringIO()
        for packet in packet_slicer(length):
            p = RequestHandler.request.recv(
                packet, socket.MSG_WAITALL
            )  # wait for full mssg
            msg.write(p)
        if msg.len != length:
            raise errors.CommunicationError(
                f"recieved message is too short. expected length {length},"
                f" recieved length {msg.len}"
            )
        msg.seek(0)
        return msg

    command, config, metadata, axes, photons, contributions = tuple(
        get_msg(msglength)
        for msglength in struct.unpack(
            "QQQQQQ", RequestHandler.request.recv(48, socket.MSG_WAITALL)
        )
    )
    return (
        command.read(),
        config.read(),
        metadata.read(),
        numpy.load(axes),
        numpy.load(photons),
        numpy.load(contributions),
    )
