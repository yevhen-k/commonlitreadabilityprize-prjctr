import torch

config = {
    "train_dataset_path": "./dataset/train.csv",
    "batch_size": 64,
    "device": torch.device(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    "epochs": 5,
}
