from typing import NamedTuple, Optional, Text, Union
from os.path import join

from functools import partial
from h5py import Dataset, File, Group

from ..util import as_string

# Generic hdf5 access types.


class DatasetPathContains(NamedTuple):
    path: Text


class DatasetPathWithAttribute(NamedTuple):
    attribute: Text
    value: bytes


class HItem(NamedTuple):
    name: Text
    optional: bool


DatasetPath = Union[DatasetPathContains,
                    DatasetPathWithAttribute,
                    HItem]


def _v_attrs(attribute: Text, value: Text, _name: Text, obj) -> Dataset:
    """visite each node and check the attribute value"""
    if isinstance(obj, Dataset):
        if attribute in obj.attrs and obj.attrs[attribute] == value:
            return obj

def _v_item(key: Text, name: Text, obj: Dataset) -> Dataset:
    """visite each node and check that the path contain the key"""
    if key in name:
        return obj


def get_dataset(h5file: File, path: DatasetPath) -> Optional[Dataset]:
    res = None
    if isinstance(path, DatasetPathContains):
        res = h5file.visititems(partial(_v_item, path.path))
    elif isinstance(path, DatasetPathWithAttribute):
        res = h5file.visititems(partial(_v_attrs,
                                        path.attribute, path.value))
    elif isinstance(path, HItem):
        res = h5file.visititems(partial(_v_item,
                                        join("scan_data", path.name)))
        if not path.optional and res is None:
            raise
    return res


# tables here...

class GroupPathWithAttribute(NamedTuple):
    attribute: Text
    value: bytes


class GroupPathNxClass(NamedTuple):
    value: bytes


GroupPath = Union[GroupPathWithAttribute,
                  GroupPathNxClass,
                  str]


def _g_attrs(attribute: Text,
             value: Text,
             _name: Text, obj) -> Optional[Group]:
    """visite each node and check the attribute value"""
    if isinstance(obj, Group):
        if attribute in obj.attrs and obj.attrs[attribute] == value:
            return obj
    return None


def get_nxclass(h5file: File,
                gpath: GroupPath) -> Optional[Group]:
    res = None
    if isinstance(gpath, GroupPathWithAttribute):
        res = h5file.visititems(partial(_g_attrs,
                                        gpath.attribute, gpath.value))
    elif isinstance(gpath, GroupPathNxClass):
        res = h5file.visititems(partial(_g_attrs,
                                        'NX_class', gpath.value))
    elif isinstance(gpath, str):
        res = h5file.visititems(partial(_g_attrs,
                                        'NX_class', gpath.encode()))

    return res


def node_as_string(node):
    if node.shape == ():
        content = node
    else:
        content = node[0]
    return as_string(content)
