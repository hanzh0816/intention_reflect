import sys

sys.path.append("/home/hzh/code/planning/planTF")
from src.models.planTF.modules.snn_utlis import get_default_snn_config

SNN_STDP_CONFIG = {
    "model": {
        "input_size": 784,
        "hidden_dim": 256,
        "num_classes": 10,
        "use_stdp": True,
        "snn_cfg": {
            **get_default_snn_config(),
            "time_steps": 256,
            "use_stdp": True,
            "stdp_cfg": {
                "learning_rate": 0.001,
                # SpikingJelly使用tau参数，移除A_pre和A_post
                "tau_pre": 10.0,  # 从2.0增加到10.0以获得更长的时间窗口
                "tau_post": 10.0,  # 从2.0增加到10.0以获得更长的时间窗口
            },
        },
    },
    "training": {
        "batch_size": 128,
        "epochs": 100,
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
