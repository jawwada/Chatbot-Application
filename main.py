from machine_learning.learners.intent_classifier import IntentClassifier



# Load the trained model and tokenizer from saved files
class_name="IntentClassifierLSTMWithAttention"
model_serve=IntentClassifier(class_name)
model_serve.load(class_name)
