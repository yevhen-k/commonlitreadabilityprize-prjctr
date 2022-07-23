from typing import Dict, Tuple
import torch
import transformers
from transformers import BertTokenizer
import pandas as pd

__all__ = ["Dataset"]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, device: torch.device) -> None:
        super(Dataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.X_encoded = self.tokenizer(self.df.excerpt.values.tolist(),
                                        return_tensors='pt', padding=True)
        X_input_ids = torch.tensor(self.X_encoded["input_ids"]).to(device)
        X_token_type_ids = torch.tensor(
            self.X_encoded["token_type_ids"]).to(device)
        X_attention_mask = torch.tensor(
            self.X_encoded["attention_mask"]).to(device)
        self.X_encoded = {
            "input_ids": X_input_ids,
            "token_type_ids": X_token_type_ids,
            "attention_mask": X_attention_mask
        }
        self.y_labels = torch.tensor(
            self.df["target"].values, dtype=torch.float32).to(device)

    def __len__(self) -> int:
        return len(self.y_labels)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(index, torch.Tensor):
            index = index.item()
        return (
            {"label": torch.unsqueeze(self.y_labels[index], 0)},
            {"input_ids": self.X_encoded["input_ids"][index],
             "token_type_ids": self.X_encoded["token_type_ids"][index],
             "attention_mask": self.X_encoded["attention_mask"][index],
             })
