from torch.utils.data import Dataset
import torch

class DialogueAttributionDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.samples = []
        for d in data:
            quote = d['quote']
            context = d['context']
            input_text = quote + tokenizer.sep_token + context
            label = label2id[d['speaker']]
            self.samples.append((input_text, label))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text,
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }
