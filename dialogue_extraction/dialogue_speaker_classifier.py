from transformers import BertModel
import torch.nn as nn

class DialogueSpeakerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DialogueSpeakerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids = None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_token = output.last_hidden_state[:, 0, :]  # [CLS] token
        x = self.dropout(cls_token)
        logits = self.classifier(x)
        return logits