from __future__ import annotations

import glob
import os
import shutil
from typing import Optional

import tensorflow as tf
from trieste.models.utils import get_module_with_variables


def save_round_model(
    *,
    current_model,
    result,
    outdir: str,
    round_idx: int,
    dim: int,
) -> None:
    model_dir = os.path.join(outdir, f"models/round_{round_idx}")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    if result is None:
        return

    gpr_model = current_model.model
    module = get_module_with_variables(result.try_get_final_model())
    module.predict_f = tf.function(
        gpr_model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, dim], dtype=tf.float64)],
    )
    tf.saved_model.save(module, model_dir)


def load_round_model(model_dir: str, round_idx: Optional[int] = None) -> tf.Module:
    models = glob.glob(os.path.join(model_dir, "round_*"))
    if len(models) == 0:
        raise FileNotFoundError(f"No models found in {model_dir}.")

    if round_idx is not None:
        model_path = os.path.join(model_dir, f"round_{round_idx}")
        if model_path not in models:
            raise FileNotFoundError(f"Model for round {round_idx} does not exist in {model_dir}.")
    else:
        model_path = max(models, key=os.path.getmtime)

    module = tf.saved_model.load(model_path)
    return module

