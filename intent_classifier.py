# -*- coding: utf-8 -*-

import json

import pandas as pd
import torch
from machine_learning.IntentTokenizer import IntentTokenizer
import torch.nn.functional as F

class IntentClassifier:
    device=None
    model=None
    tokenizer=None
    def __init__(self):
       self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")


    def is_ready(self):
        try:
            self.model=self.model.to(self.device)
            self.model.eval()
            return True
        except:
            return False

    def load(self, file_path):
        self.model = torch.load(file_path)
        self.tokenizer = tokenizer = tokenizer = \
            IntentTokenizer.load_state(IntentTokenizer,
                                       f"models/IntentClassifierLSTMWithAttention_tokenizer.pickle",
                                       f"models/IntentClassifierLSTMWithAttention_le.pickle")

        return True

    def predict(self, query):
        # Tokenize query
        query_df = pd.DataFrame({"text": [query]})
        input_tensor = self.tokenizer.get_Inference_Tensor(query_df, device=self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3)

        top3_labels = self.tokenizer.le.inverse_transform(top3_indices.data.cpu().numpy()[0])
        top3_confidences = top3_probs.data.cpu().numpy()[0]

        # Prepare top 3 predictions
        predictions = [{"label": label, "confidence": float(conf)} for label, conf in zip(top3_labels, top3_confidences)]
        return predictions

if __name__ == '__main__':
    pass
