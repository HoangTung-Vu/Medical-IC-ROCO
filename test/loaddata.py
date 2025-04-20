from datasets import Dataset, concatenate_datasets
import os
import re
import torch
# === Load ROCO Dataset ===
data_path = '/home/hoangtungvum/CODE/MIC/data/mdwiratathya___roco-radiology/default/0.0.0/03471f547bb646a1f447add638d46bb3507523e8'

arrow_files_train = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.arrow') and f.startswith('roco-radiology-train')]
arrow_files_test = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.arrow') and f.startswith('roco-radiology-test')]
arrow_files_valid = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.arrow') and f.startswith('roco-radiology-validation')]


datasets_train = [Dataset.from_file(f).with_format("torch") for f in arrow_files_train]
validset = Dataset.from_file(arrow_files_valid[0]).with_format("torch")
testset = Dataset.from_file(arrow_files_test[0]).with_format("torch")
trainset = concatenate_datasets(datasets_train).with_format("torch")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

print(type(trainloader))
print(type(validset))
print(len(testset))