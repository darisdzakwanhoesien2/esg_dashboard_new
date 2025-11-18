# dashboard/finbert_model.py
import torch
import torch.nn as nn

class MultiTaskFinBERT(nn.Module):
    """
    Minimal skeleton. Replace with your real implementation.
    It should accept an encoder (transformers.AutoModel) and implement forward/predict.
    """
    def __init__(self, encoder, hidden_dim=768, num_sentiment_labels=3, num_tone_labels=6):
        super().__init__()
        self.encoder = encoder  # huggingface encoder
        # small heads (examples)
        self.sent_head = nn.Linear(hidden_dim, num_sentiment_labels)
        self.tone_head = nn.Linear(hidden_dim, num_tone_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # take pooled output if available else mean pool
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None \
                 else outputs.last_hidden_state.mean(dim=1)
        s_logits = self.sent_head(pooled)
        t_logits = self.tone_head(pooled)
        return s_logits, t_logits

    def predict(self, tokenizer_outputs):
        """
        Convenience wrapper: tokenized inputs -> returns labels as dict
        """
        input_ids = tokenizer_outputs.get("input_ids")
        attention_mask = tokenizer_outputs.get("attention_mask")
        s_logits, t_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        s_pred = int(torch.argmax(s_logits, dim=-1).cpu().numpy()[0])
        t_pred = int(torch.argmax(t_logits, dim=-1).cpu().numpy()[0])
        return {"sentiment_label": s_pred, "tone_label": t_pred}
