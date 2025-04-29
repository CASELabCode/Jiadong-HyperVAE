import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split  
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


from utils import (
    smooth_output_gaussian,
    minmax_scale,
    inverse_minmax_scale,
    wavelength_to_rgb,
    calculate_metrics,
    save_to_csv,
    inverse_transform_by_row,
    plot_results,
    plot_line_comparison_all
)


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # encoder
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        
        self.fc1 = nn.Linear(1024, 256)  
        self.fc2 = nn.Linear(1024, 256)  

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim* 2),
            nn.Softplus()
        )

    def encode(self, x):
        x = self.encoder_layer(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)*0.01
        z = mu + eps * std
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        pred_mu, pred_logvar = torch.chunk(decoded, 2, dim=1)
        pred_logvar = F.softplus(pred_logvar)  # 确保 logvar 是正的
        return pred_mu, pred_logvar

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred_mu, pred_logvar = self.decode(z)
        return pred_mu, pred_logvar, mu, log_var



def loss_function(pred_mu, pred_logvar, target, mu_z, log_var_z, beta=1.0):
    epsilon = 1e-8  
    
    # Gaussian Negative Log Likelihood
    nll = 0.5 * torch.mean(torch.log(2 * torch.tensor(torch.pi)) + pred_logvar + ((target - pred_mu) ** 2) / (torch.exp(pred_logvar) + epsilon))

    # KL divergence from z ~ q(z|x) to N(0,1)
    kl = -0.5 * torch.mean(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())

    l1 = F.l1_loss(pred_mu, target, reduction='mean')

    total_loss = nll + beta * kl+10*l1
    return total_loss, nll, kl

 

def train(model, train_dl, epochs=2000):
    model.train()

    min_total_loss = float('inf')

    best_model_total_path = 'F:\\aphy-chla-predictions\\Model\\vae_trans_model_best_total.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred_mu, pred_logvar, mu_z, log_var_z = model(x)
            loss, nll, kl = loss_function(pred_mu, pred_logvar, y, mu_z, log_var_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\aphy-chla-predictions\\Model\\vae_model.pth')


def evaluate(model, test_dl):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred_mu, pred_logvar, _, _ = model(x)

            predictions.append(pred_mu.cpu().numpy())  
            actuals.append(y.cpu().numpy())

    return np.vstack(predictions), np.vstack(actuals)


def minmax_scale(data, min_val, max_val, feature_range=(1, 10)):
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    min_range = feature_range[0]
    data_scaled = (data - min_val) * scale + min_range
    return data_scaled

def inverse_minmax_scale(data_scaled, min_val, max_val, feature_range=(1, 10)):
    scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
    min_range = feature_range[0]
    data_original = (data_scaled - min_range) * scale + min_val
    return data_original



def load_real_data(aphy_file_path, rrs_file_path, seed=42, test_indices_path="test_indices_PACE.npy"):
    torch.manual_seed(seed)  # 固定 PyTorch 随机种子
    np.random.seed(seed)     # 固定 NumPy 随机种子

    # 读取数据
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    Rrs_real = array2
    log_Rrs_real = np.log10(array2+ 1e-6)
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 3

    # 归一化 Rrs_real
    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]
    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

    # 转换为 PyTorch 张量
    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)

    # 构建数据集
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = len(dataset_real)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    # **固定 test set 索引**
    if os.path.exists(test_indices_path):
        test_indices = np.load(test_indices_path)  # 直接加载已有的 test set 索引
    else:
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)  # 随机打乱索引
        test_indices = indices[train_size:]  # 取 30% 作为 test set
        np.save(test_indices_path, test_indices)  # 保存索引，确保所有代码运行时 test set 一致

    # 计算 train set 索引
    train_indices = np.setdiff1d(np.arange(dataset_size), test_indices)

    # 根据索引划分数据
    train_dataset_real = torch.utils.data.Subset(dataset_real, train_indices)
    test_dataset_real = torch.utils.data.Subset(dataset_real, test_indices)

    # 创建 DataLoader
    train_real_dl = DataLoader(train_dataset_real, batch_size=1024, shuffle=True, num_workers=0)
    test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=0)

    return train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val



def load_real_test(aphy_file_path, rrs_file_path):

    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 2.4

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)  
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=0, pin_memory=True)

    return test_real_dl, input_dim, output_dim, min_val, max_val



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val  = load_real_data('F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\aphy_PACE.csv','F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\Rrs_PACE.csv')
    test_real_Sep, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Sep.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Sep.csv')
    test_real_Oct, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Oct.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Oct.csv')

    save_dir = "F:\\aphy-chla-predictions\\plots\\VIB_aph_PACE"
    os.makedirs(save_dir, exist_ok=True)

    model = VAE(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.004)


    train(model, train_real_dl, epochs=2000)

    model.load_state_dict(torch.load('F:\\aphy-chla-predictions\\Model\\vae_trans_model_best_total.pth', map_location=device))

    predictions, actuals = evaluate(model, test_real_dl)

    predictions_Sep, actuals_Sep = evaluate(model, test_real_Sep)

    predictions_Oct, actuals_Oct = evaluate(model, test_real_Oct)



    plot_line_comparison_all(predictions, actuals, save_dir, mode='test')
    plot_results(predictions, actuals, save_dir, mode='test')

    plot_line_comparison_all(predictions_Sep, actuals_Sep, save_dir, mode='Sep')
    plot_results(predictions_Sep, actuals_Sep, save_dir, mode='Sep')

    plot_line_comparison_all(predictions_Oct, actuals_Oct, save_dir, mode='Oct')
    plot_results(predictions_Oct, actuals_Oct, save_dir, mode='Oct')


