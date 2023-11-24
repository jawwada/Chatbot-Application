# -*- coding: utf-8 -*-

import json

import pandas as pd
import torch
from machine_learning.learners.IntentTokenizer import IntentTokenizer
from machine_learning.learners.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
import torch.nn.functional as F

class IntentClassifier:
    device=None
    model=None
    tokenizer=None
    def __init__(self,model_class_name=None):
       self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")
       self.model=self.create_object(model_class_name).to(self.device)

    def is_ready(self):
        try:
            self.model.eval()
            return True
        except:
            return False

    def create_object(self,load_class_name):
        with open(f"config/model_initialization/{load_class_name}.json", 'r') as file:
            parameters = json.load(file)
        class_ = globals()[load_class_name]
        return class_(**parameters)

    def load(self, file_path):

        self.tokenizer = \
            IntentTokenizer.load_state(IntentTokenizer,
                                       f"data/models/IntentClassifierLSTMWithAttention_tokenizer.pickle",
                                       f"data/models/IntentClassifierLSTMWithAttention_le.pickle")
        self.model.load_state_dict(torch.load(f"data/models/{file_path}_state_dict.pth", map_location=self.device))

        return self.is_ready()

    def predict(self, query):
        # Tokenize query
        query_df = pd.DataFrame({"text": [query]})
        input_tensor = self.tokenizer.get_Inference_Tensor(query_df, device=self.device)
        self.model.eval()
        # Inference
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
