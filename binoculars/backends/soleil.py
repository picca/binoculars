from typing import NamedTuple, Optional, Text, Union

from os.path import join
from functools import partial
from h5py import Dataset, File
from tables.exceptions import NoSuchNodeError

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
        for group in h5file.get_node('/'):
            scan_data = group._f_get_child("scan_data")
            try:
                res = scan_data._f_get_child(path.name)
            except NoSuchNodeError:
                if not path.optional:
                    raise
    return res
