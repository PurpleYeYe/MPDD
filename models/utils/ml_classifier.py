import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # 用于模型保存

class MLClassifierWrapper:
    def __init__(self, model_type='svm'):
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Unsupported model type: choose 'svm' or 'rf'.")

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def evaluate(self, features, labels):
        preds = self.predict(features)
        acc = accuracy_score(labels, preds)
        return acc
