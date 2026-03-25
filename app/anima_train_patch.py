import sys
import os

# Add sd-scripts to the system path to allow importing its modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD_SCRIPTS_DIR = os.path.join(BASE_DIR, "sd-scripts")
sys.path.insert(0, SD_SCRIPTS_DIR)

import torch
from library import anima_models, train_util

# Monkey-patch Anima.forward to fix the keyword argument mismatch in anima_train.py
original_forward = anima_models.Anima.forward

def patched_forward(self, x, timesteps, context=None, **kwargs):
    # anima_train.py incorrectly passes 't5_input_ids' and 't5_attn_mask'.
    # We map them to the names expected by the model's forward method.
    if "t5_input_ids" in kwargs and kwargs.get("target_input_ids") is None:
        kwargs["target_input_ids"] = kwargs.pop("t5_input_ids")
    if "t5_attn_mask" in kwargs and kwargs.get("target_attention_mask") is None:
        kwargs["target_attention_mask"] = kwargs.pop("t5_attn_mask")
    return original_forward(self, x, timesteps, context, **kwargs)

anima_models.Anima.forward = patched_forward

# Import and execute the original training script logic
import anima_train

if __name__ == "__main__":
    parser = anima_train.setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Maintain backward compatibility for attention mode
    if hasattr(args, "attn_mode") and args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    anima_train.train(args)
