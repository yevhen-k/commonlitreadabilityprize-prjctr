import torch
from dataset import Dataset
from model import RegressionModel
from utils import set_seed
from transformers import BertTokenizer
from torch.utils.data import random_split, DataLoader

set_seed(1337)

config = {
    "train_dataset_path": "./dataset/train.csv",
    "batch_size": 64,
    "device": torch.device(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    "epochs": 5,
}

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

# sample = ["test test test", "echo", "some text here as well???", "the fuck??"]
# sample = ["test test test"]
# encodings = tokenizer(sample, return_tensors='pt', padding=True)
# print("\n#####")
# print(model(encodings))


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (y, X) in enumerate(dataloader):
        # Compute prediction and loss
        # X_input_ids = X["input_ids"]
        # X_token_type_ids = X["token_type_ids"]
        # X_attention_mask = X["attention_mask"]
        # pred = model(X_input_ids, X_token_type_ids, X_attention_mask)
        pred = model(X)
        loss = loss_fn(pred, y["label"])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(y["label"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for y, X in dataloader:
            # X_input_ids = X["input_ids"]
            # X_token_type_ids = X["token_type_ids"]
            # X_attention_mask = X["attention_mask"]
            # pred = model(X_input_ids, X_token_type_ids, X_attention_mask)
            pred = model(X)
            test_loss += loss_fn(pred, y["label"]).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


for t in range(config["epochs"]):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    reduce_on_plateau_scheduler.step(test_loss)
print("Done!")

model.to(torch.device('cpu'))
torch.save(model.state_dict(), 'bert-base-uncased-regression-weights.pth')
