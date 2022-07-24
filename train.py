import torch
from dataset import Dataset
from model import RegressionModel
from utils import set_seed, train_loop, test_loop
from conf import config
from transformers import BertTokenizer
from torch.utils.data import random_split, DataLoader

set_seed(1337)

device = config["device"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = Dataset(config["train_dataset_path"], tokenizer, device)
dataset_len = len(dataset)
train_size = int(0.8 * dataset_len)
test_size = dataset_len - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size])
train_dataloader = DataLoader(
    train_dataset, config["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_dataset, config["batch_size"], shuffle=True)


model = RegressionModel().to(device)
model.eval()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params)
reduce_on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer)
loss_fn = torch.nn.MSELoss()


for t in range(config["epochs"]):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    reduce_on_plateau_scheduler.step(test_loss)
print("Done!")

model.to(torch.device('cpu'))
torch.save(model.state_dict(), 'bert-base-uncased-regression-weights.pth')
