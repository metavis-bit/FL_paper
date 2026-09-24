import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import glob
import pickle
import urllib.request
import numpy as np

# 如果你在本地运行，请确保这些模块路径正确
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *

# ==========================================
# 1. 基础配置 (切换这里以测试不同模型)
# ==========================================
# 测试图像模型示例: DATASET_NAME = "Cifar100", MODEL_NAME = "resnet18"
# 测试文本模型示例: DATASET_NAME = "Shakespeare", MODEL_NAME = "lstm"
#　resnet18_Cifar100

DATASET_NAME = "Shakespeare"
MODEL_NAME = "lstm"

# 路径设置 (兼容图像模型和文本模型)
DATASET_PATH = rf'../dataset/{DATASET_NAME}/rawdata'
# 如果是 lstm，按照你给出的路径习惯可指向 shakespeare_lstm；否则走通用格式
if MODEL_NAME.lower() == "lstm":
    MODELS_DIR = rf'./models/shakespeare_lstm'
else:
    MODELS_DIR = rf'./models/{MODEL_NAME}_{DATASET_NAME}'

BATCH_SIZE = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[*] Initializing test environment for {MODEL_NAME} on {DATASET_NAME}...")
print(f"[*] Using device: {device}")

class CIFAR20(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CIFAR100 的原始文件中自带了 coarse_labels，我们只需要将其读出并覆盖默认的 targets 即可
        file_path = os.path.join(self.root, self.base_folder, 'train' if self.train else 'test')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        self.targets = entry['coarse_labels']

# ==========================================
# 2. 定义 LSTM 相关类 (仅在需要时使用)
# ==========================================
class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.n_seqs = len(data) // seq_len

    def __len__(self):
        return self.n_seqs - 1

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        x = self.data[start_idx: end_idx]
        y = self.data[start_idx + 1: end_idx + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, 8)
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))


# ==========================================
# 3. 数据集与模型初始化
# ==========================================
if MODEL_NAME.lower() == "lstm":
    # --- LSTM 数据预处理 ---
    DATA_FILE = os.path.join(DATASET_PATH, 'shakespeare.txt')
    URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    os.makedirs(DATASET_PATH, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(URL, DATA_FILE)

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = tuple(sorted(list(set(text))))
    vocab_size = len(chars)
    char2int = {c: i for i, c in enumerate(chars)}
    encoded_text = np.array([char2int[c] for c in text])

    # 划分测试集 (10%)
    split_idx = int(len(encoded_text) * 0.9)
    test_data = encoded_text[split_idx:]

    print(f"Vocab size: {vocab_size} | Test data size: {len(test_data)} chars")

    # 初始化 DataLoader 和 模型
    SEQ_LEN = 100
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2

    test_dataset = ShakespeareDataset(test_data, SEQ_LEN)
    # 注意：LSTM 测试时 drop_last 必须为 True，否则最后一个非完整 batch 会导致 init_hidden 维度报错
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model = CharLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

else:
    # --- 图像模型数据预处理 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if DATASET_NAME.lower() == "cifar100":
        testset = CIFAR20(root=DATASET_PATH, train=False, download=True, transform=transform)
        num_classes = 20
    elif DATASET_NAME.lower() == "cifar10":
        testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if MODEL_NAME.lower() == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif MODEL_NAME.lower() == "vgg11":
        model = VGG11("VGG11", num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

model = model.to(device)
model.eval()  # 强制设置为测试模式

# ==========================================
# 4. 扫描并测试所有保存的模型
# ==========================================
if not os.path.exists(MODELS_DIR):
    raise FileNotFoundError(f"[!] Directory not found: {MODELS_DIR}. Please run the training script first.")

model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))

if not model_files:
    print(f"[!] No models found in {MODELS_DIR}.")
    exit()

print(f"[*] Found {len(model_files)} models in {MODELS_DIR}. Starting evaluation...\n")

results = {}

with torch.no_grad():
    for file_path in model_files:
        # drop_size = np.random.uniform(-20, 20)
        filename = os.path.basename(file_path)

        # 兼容两种命名格式：
        # 1. 爬塔格式: FedAvg_d-0.2_Epoch10_Acc31.15.pth
        # 2. 简单格式: ours.pth
        parts = filename.replace(".pth", "").split("_")
        if len(parts) >= 4:
            algo = parts[0]
            setting = parts[1].replace("-", "=")
        else:
            algo = parts[0]
            setting = "Default"

        print(f"Testing model: Algorithm=[{algo}], Setting=[{setting}] ...", end=" ", flush=True)

        model.load_state_dict(torch.load(file_path, map_location=device))

        correct = 0
        total = 0

        # 🌟 核心：根据模型类型路由不同的测试前向传播逻辑 🌟
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if MODEL_NAME.lower() == "lstm":
                # LSTM 专属逻辑：隐状态初始化与形状处理
                test_hidden = model.init_hidden(BATCH_SIZE)
                outputs, _ = model(inputs, test_hidden)
                _, predicted = torch.max(outputs, 1)
                total += targets.numel()
                correct += (predicted == targets.view(-1)).sum().item()
            else:
                # 图像模型专属逻辑
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += (targets.size(0))
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total



        print(f"Accuracy: {accuracy:.2f}%")

        if setting not in results:
            results[setting] = {}
        results[setting][algo] = accuracy

# ==========================================
# 5. 打印汇总表格
# ==========================================
print("\n" + "=" * 80)
print(f"   EVALUATION SUMMARY: {MODEL_NAME.upper()} on {DATASET_NAME.upper()}")
print("=" * 80)

# 动态提取表格表头（适应未知的模型名称组合）
all_algos = []
for setting in results:
    for algo in results[setting]:
        if algo not in all_algos:
            all_algos.append(algo)

# 打印表头
header = f"{'Setting':<10}"
for algo in all_algos:
    header += f"{algo:>10}"
print(header)
print("-" * 80)

# 打印每一行数据
for setting, algos_dict in results.items():
    row_str = f"{setting:<10}"
    for algo in all_algos:
        acc = algos_dict.get(algo, "N/A")
        if isinstance(acc, float):
            row_str += f"{acc:>10.2f}"
        else:
            row_str += f"{acc:>10}"
    print(row_str)

print("=" * 80)
print("Testing Complete.")
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import os
# import glob
# from flcore.trainmodel.models import *
# from flcore.trainmodel.resnet import *
#
# # ==========================================
# # 1. 基础配置 (需要与训练时的配置保持一致)
# # ==========================================
# DATASET_NAME = "Cifar100"
# MODEL_NAME = "resnet18"
#
# # 路径设置
# DATASET_PATH = rf'../dataset/{DATASET_NAME}/rawdata'
# # 自动寻找之前训练代码保存的文件夹
# MODELS_DIR = rf'./models/{MODEL_NAME}_{DATASET_NAME}'
#
# BATCH_SIZE = 128  # 测试时 batch size 可以设大一点以加快速度
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # ==========================================
# # 2. 数据集与模型初始化
# # ==========================================
# print(f"[*] Initializing test environment for {MODEL_NAME} on {DATASET_NAME}...")
# print(f"[*] Using device: {device}")
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 根据名称加载对应数据集
# if DATASET_NAME.lower() == "cifar100":
#     testset = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=False, download=True, transform=transform)
#     num_classes = 100
# elif DATASET_NAME.lower() == "cifar10":
#     testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
#     num_classes = 10
# else:
#     raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#
# # 根据名称加载对应模型
# if MODEL_NAME.lower() == "resnet18":
#     model = torchvision.models.resnet18(num_classes=num_classes)
# elif MODEL_NAME.lower() == "vgg11":
#     model = VGG11("VGG11", num_classes=num_classes)  # 假设 VGG11 是你导入的自定义模型
# else:
#     raise ValueError(f"Unsupported model: {MODEL_NAME}")
#
# model = model.to(device)
# model.eval()  # 强制设置为测试模式
#
# # ==========================================
# # 3. 扫描并测试所有保存的模型
# # ==========================================
# if not os.path.exists(MODELS_DIR):
#     raise FileNotFoundError(f"Directory not found: {MODELS_DIR}. Please run the training script first.")
#
# # 获取所有 .pth 文件
# model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
#
# if not model_files:
#     print(f"[!] No models found in {MODELS_DIR}.")
#     exit()
#
# print(f"[*] Found {len(model_files)} models in {MODELS_DIR}. Starting evaluation...\n")
#
# # 用于存储结果以便最后画表的字典
# # 格式: results["d=0.2"]["FedAvg"] = 31.50
# results = {}
#
# with torch.no_grad():  # 测试阶段不需要计算梯度
#     for file_path in model_files:
#         filename = os.path.basename(file_path)
#
#         # 解析文件名: 例 FedAvg_d-0.2_Epoch10_Acc31.15.pth
#         # 把 d-0.2 还原为 d=0.2 方便展示
#         parts = filename.replace(".pth", "").split("_")
#         if len(parts) >= 4:
#             algo = parts[0]
#             setting = parts[1].replace("-", "=")
#
#             print(f"Testing model: Algorithm=[{algo}], Setting=[{setting}] ...", end=" ", flush=True)
#
#             # 加载权重
#             model.load_state_dict(torch.load(file_path, map_location=device))
#
#             # 运行测试集
#             correct = 0
#             total = 0
#             for imgs, lbls in testloader:
#                 imgs, lbls = imgs.to(device), lbls.to(device)
#                 outs = model(imgs)
#                 _, predicted = torch.max(outs.data, 1)
#                 total += lbls.size(0)
#                 correct += (predicted == lbls).sum().item()
#
#             accuracy = 100 * correct / total
#             print(f"Accuracy: {accuracy:.2f}%")
#
#             # 存入结果字典
#             if setting not in results:
#                 results[setting] = {}
#             results[setting][algo] = accuracy
#         else:
#             print(f"[!] Skipping {filename}: Unrecognized naming format.")
#
# # ==========================================
# # 4. 打印汇总表格
# # ==========================================
# print("\n" + "=" * 80)
# print(f"   EVALUATION SUMMARY: {MODEL_NAME.upper()} on {DATASET_NAME.upper()}")
# print("=" * 80)
#
# # 定义需要展示的算法顺序 (为了和原图一致)
# algo_order = ["FedAvg", "DPFedAvg", "HEFedAvg", "LDP-FL", "SCAFFOLD", "FedMut", "FedPVR", "Adap-CTA"]
# # 定义异构设置顺序
# setting_order = ["d=0.2", "d=0.5", "d=0.8", "IID"]
#
# # 打印表头
# header = f"{'Setting':<10}"
# for algo in algo_order:
#     header += f"{algo:>10}"
# print(header)
# print("-" * 80)
#
# # 打印每一行数据
# for setting in setting_order:
#     if setting in results:
#         row_str = f"{setting:<10}"
#         for algo in algo_order:
#             acc = results[setting].get(algo, "N/A")  # 如果没找到对应模型，显示 N/A
#             if isinstance(acc, float):
#                 row_str += f"{acc:>10.2f}"
#             else:
#                 row_str += f"{acc:>10}"
#         print(row_str)
#     else:
#         # 如果这个 setting 下没有任何模型
#         print(f"{setting:<10}" + "".join([f"{'N/A':>10}" for _ in algo_order]))
#
# print("=" * 80)
# print("Testing Complete.")