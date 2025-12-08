import sys

sys.path.append("/home/hzh/code/planning/planTF")
from src.models.planTF.modules.snn_utlis import get_default_snn_config

SNN_STDP_CONFIG = {
    "model": {
        "input_size": 784,
        "hidden_dim1": 512,
        "hidden_dim2": 256,
        "num_classes": 10,
        "dropout": 0.2,
        "use_stdp": True,
        "population_size": 1,
        "snn_cfg": {
            **get_default_snn_config(),
            "time_steps": 8,
            "use_stdp": True,
            "stdp_cfg": {
                "learning_rate": 0.001,
                "A_pre": 0.01,
                "A_post": -0.01,
                "tau_pre": 10.0,
                "tau_post": 10.0,
            },
        },
    },
    "training": {
        "batch_size": 128,
        "epochs": 100,
        "stdp_lr": 0.001,
        "stdp_a_pre": 0.01,
        "stdp_a_post": -0.01,
        "stdp_tau_pre": 10.0,
        "stdp_tau_post": 10.0,
        "device": "cuda",
        "num_workers": 4,
    },
    "data": {
        "data_dir": "./data/mnist",
        "normalize": True,
        "augment": False,
    },
    "logging": {
        "checkpoint_dir": "./checkpoints/snn_stdp",
        "log_interval": 100,
    },
}
