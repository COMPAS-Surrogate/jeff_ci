"""
Small runtime compatibility patches for third-party libraries.

These are intentionally local, opt-in monkeypatches to work around upstream
issues in specific dependency/version combinations used by this repository.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, Tuple


def _iter_unique_parameters(params: Iterable[object]) -> Tuple[object, ...]:
    seen: set[int] = set()
    unique: list[object] = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(p)
    return tuple(unique)


def patch_gpflow_trainable_parameters_for_py312_tf216() -> None:
    """
    Work around TensorFlow 2.16 + Python 3.12 `tf.Module._flatten` failures.

    Trieste calls `gpflow.Module.trainable_parameters`, which (in GPflow 2.10)
    delegates to `tf.Module._flatten(...)`. With TF 2.16 on Python 3.12 this can
    raise:
      TypeError: this __dict__ descriptor does not support '_TupleWrapper' objects

    The failure is triggered by TensorFlow's trackable wrappers around tuples
    (e.g. GPflow GPR's `.data` attribute). We avoid TF's flattening and instead
    traverse the module graph using GPflow's own traversal utilities.
    """

    if sys.version_info < (3, 12):
        return

    try:
        import tensorflow as tf  # noqa: F401
        import gpflow
    except Exception:
        return

    if not tf.__version__.startswith("2.16"):
        return

    from gpflow.base import Module, Parameter
    from gpflow.utilities import leaf_components

    if getattr(Module, "_cosmic_patched_trainable_parameters", False):
        return

    logger = logging.getLogger(__name__)
    logger.info(
        "Applying GPflow trainable_parameters patch for Python %s + TensorFlow %s",
        sys.version.split()[0],
        tf.__version__,
    )

    def _parameters_via_traversal(self: Module) -> Tuple[Parameter, ...]:
        items = sorted(leaf_components(self).items(), key=lambda kv: kv[0])
        params = (v for _, v in items if isinstance(v, Parameter))
        return _iter_unique_parameters(params)  # type: ignore[return-value]

    def _trainable_parameters_via_traversal(self: Module) -> Tuple[Parameter, ...]:
        items = sorted(leaf_components(self).items(), key=lambda kv: kv[0])
        params = (v for _, v in items if isinstance(v, Parameter) and v.trainable)
        return _iter_unique_parameters(params)  # type: ignore[return-value]

    Module.parameters = property(_parameters_via_traversal)  # type: ignore[assignment]
    Module.trainable_parameters = property(_trainable_parameters_via_traversal)  # type: ignore[assignment]
    Module._cosmic_patched_trainable_parameters = True  # type: ignore[attr-defined]


def patch_tensorflow_nest_protocol_for_py312_tf216() -> None:
    """
    Work around TF nest's Protocol isinstance crash on Python 3.12.

    TensorFlow 2.16 uses `isinstance(x, CustomNestProtocol)` inside nest utilities.
    On Python 3.12, Protocol instance checks can raise `TypeError` for some
    proxy/trackable objects (notably TF's `_TupleWrapper`). This breaks
    `tf.Module.trainable_variables` and GPflow/Trieste optimization.
    """

    if sys.version_info < (3, 12):
        return

    try:
        import tensorflow as tf
    except Exception:
        return

    if not tf.__version__.startswith("2.16"):
        return

    try:
        from tensorflow.python.util import nest_util  # type: ignore
    except Exception:
        return

    if getattr(nest_util, "_cosmic_patched_nest_protocol", False):
        return

    logger = logging.getLogger(__name__)
    logger.info(
        "Applying TensorFlow nest Protocol patch for Python %s + TensorFlow %s",
        sys.version.split()[0],
        tf.__version__,
    )

    def _safe_is_custom_nest_protocol(iterable: object) -> bool:
        try:
            return isinstance(iterable, nest_util.CustomNestProtocol)
        except TypeError:
            return False

    def _tf_core_yield_sorted_items_patched(iterable):  # noqa: ANN001
        if isinstance(iterable, list):
            for item in enumerate(iterable):
                yield item
        elif type(iterable) == tuple:  # pylint: disable=unidiomatic-typecheck
            for item in enumerate(iterable):
                yield item
        elif isinstance(iterable, (dict, nest_util._collections_abc.Mapping)):
            for key in nest_util._tf_core_sorted(iterable):
                yield key, iterable[key]
        elif nest_util._is_attrs(iterable):
            for item in nest_util._get_attrs_items(iterable):
                yield item
        elif nest_util.is_namedtuple(iterable):
            for field in iterable._fields:
                yield field, getattr(iterable, field)
        elif nest_util._is_composite_tensor(iterable):
            type_spec = iterable._type_spec  # pylint: disable=protected-access
            yield type_spec.value_type.__name__, type_spec._to_components(iterable)  # pylint: disable=protected-access
        elif nest_util._is_type_spec(iterable):
            yield iterable.value_type.__name__, iterable._component_specs  # pylint: disable=protected-access
        elif _safe_is_custom_nest_protocol(iterable):
            flat_component = iterable.__tf_flatten__()[1]
            assert isinstance(flat_component, tuple)
            yield from enumerate(flat_component)
        else:
            for item in enumerate(iterable):
                yield item

    def _tf_data_yield_value_patched(iterable):  # noqa: ANN001
        if isinstance(iterable, nest_util._collections_abc.Mapping):
            for key in nest_util._tf_data_sorted(iterable):
                yield iterable[key]
        elif iterable.__class__.__name__ == "SparseTensorValue":
            yield iterable
        elif nest_util._is_attrs(iterable):
            for _, attr in nest_util._get_attrs_items(iterable):
                yield attr
        elif _safe_is_custom_nest_protocol(iterable):
            flat_component = iterable.__tf_flatten__()[1]
            assert isinstance(flat_component, tuple)
            yield from flat_component
        else:
            for value in iterable:
                yield value

    nest_util._tf_core_yield_sorted_items = _tf_core_yield_sorted_items_patched  # type: ignore[assignment]
    nest_util._tf_data_yield_value = _tf_data_yield_value_patched  # type: ignore[assignment]
    nest_util._cosmic_patched_nest_protocol = True  # type: ignore[attr-defined]
