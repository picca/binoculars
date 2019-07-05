from typing import NamedTuple, Optional, Text, Union

from functools import partial
from h5py import Dataset, File
from tables.exceptions import NoSuchNodeError

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
        # BEWARE in that case h5file is a tables File object and not a
        # h5py File.
        for group in h5file.get_node('/'):
            scan_data = group._f_get_child("scan_data")
            try:
                res = scan_data._f_get_child(path.name)
            except NoSuchNodeError:
                if not path.optional:
                    raise
    return res


# tables here...

def get_nxclass(hfile, nxclass, path="/"):
    """
    :param hfile: the hdf5 file.
    :type hfile: tables.file.
    :param nxclass: the nxclass to extract
    :type nxclass: str
    """
    for node in hfile.walk_nodes(path):
        try:
            if nxclass == as_string(node._v_attrs['NX_class']):
                return node
        except KeyError:
            pass
    return None


def node_as_string(node):
    if node.shape == ():
        content = node.read().tostring()
    else:
        content = node[0]
    return as_string(content)
