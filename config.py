import os
import datetime
import torch


class Config:
    def __init__(self, model_name, training_mode, dataset, feature_layer_frozen=None):
        self.model_name = model_name
        self.training_mode = training_mode
        self.dataset = dataset
        self.feature_layer_frozen = feature_layer_frozen
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtg = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.model_dir = f"./models/{dataset}/{model_name}/{training_mode}"
        os.makedirs(self.model_dir, exist_ok=True)
        self.metrics_file = "metrics.json"
        self.class_labels_path = f"{self.model_dir}/class_labels.json"
        self.confusion_matrix_path = f"{self.model_dir}/confusion_matrix.png"

    def save_model_path(self, epoch, val_acc):
        return f"{self.model_dir}/{self.model_name}-{epoch}-{val_acc:.4f}.pth"
