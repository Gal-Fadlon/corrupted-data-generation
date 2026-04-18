"""Registered models that are trained locally."""
models = {
    "sdxl": {
        "dataset_size": "full",  # how many examples were in the dataset. Use either full or number.
        "consistency": False,  # whether this model was trained with consistency.
        "trained_with_lora": False,
        "timestep_nature": 0,  # level of noise that was present in the training examples. Number should be between 0 and 1000 for SDXL.
        "ckpt_path": None,  # add local checkpoint path
        "desc": "SDXL vanilla model.",
    },
}
