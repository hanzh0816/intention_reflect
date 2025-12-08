ANN_CONFIG = {
    "model": {
        "input_size": 784,
        "hidden_dim1": 512,
        "hidden_dim2": 256,
        "num_classes": 10,
        "dropout": 0.2,
        "use_batchnorm": True,
    },
    "training": {
        "batch_size": 128,
        "epochs": 50,
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
        "checkpoint_dir": "./checkpoints/ann",
        "log_interval": 100,
    },
}
