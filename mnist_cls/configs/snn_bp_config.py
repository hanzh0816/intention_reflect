import sys

sys.path.append("/home/hzh/code/planning/planTF")
from src.models.planTF.modules.snn_utlis import get_default_snn_config

SNN_BP_CONFIG = {
    "model": {
        "input_size": 784,
        "hidden_dim": 256,
        "num_classes": 10,
        "use_stdp": False,
        "snn_cfg": {**get_default_snn_config(), "time_steps": 256, "use_stdp": False},
    },
    "training": {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "device": "cuda",
        "num_workers": 4,
    },
    "data": {
        "data_dir": "./data/mnist",
        "normalize": True,
        "augment": False,
    },
    "logging": {
        "checkpoint_dir": "./checkpoints/snn_bp",
        "log_interval": 100,
    },
}
