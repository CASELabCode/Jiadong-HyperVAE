
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



class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians=5):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        
        self.pi = nn.Linear(100, num_gaussians)
        self.sigma = nn.Linear(100, num_gaussians * output_dim)
        self.mu = nn.Linear(100, num_gaussians * output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        pi = F.softmax(self.pi(x), dim=-1)
        sigma = torch.exp(self.sigma(x)).view(-1, self.num_gaussians, self.output_dim)
        mu = self.mu(x).view(-1, self.num_gaussians, self.output_dim)
        
        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, y):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    y = y.unsqueeze(1).expand_as(mu)
    log_probs = m.log_prob(y)
    pi = pi.unsqueeze(-1).expand_as(log_probs)
    weighted_log_probs = log_probs + torch.log(pi)
    log_sum = torch.logsumexp(weighted_log_probs, dim=1)
    return -log_sum.mean()

def train_mdn(model, train_dl, epochs=200, lr=1e-3, l2=1e-3):
    weight_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': weight_params, 'weight_decay': l2},
        {'params': bias_params, 'weight_decay': l2}
    ], lr=lr)

    model.train()
    min_total_loss = float('inf')
    best_model_total_path = 'F:\\VAE for aphy-chla\\Model\\mdn_model_best_total.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)
            loss = mdn_loss(pi, sigma, mu, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\VAE for aphy-chla\\Model\\mdn_model.pth')



def evaluate_mdn(model, test_dl, mode='sample'):
    """
    评估 MDN 模型：
    - `mode='max_pi'`：使用权重最大的高斯分布的均值 (`mu`) 作为预测值。
    - `mode='sample'`：在整个混合分布 (`mixture`) 中采样。
    - `mode='gaussian_mixture'`：合并所有高斯分布，得到新的均值 & 方差，从中采样。

    :param model: 训练好的 MDN 模型
    :param test_dl: 测试数据 DataLoader
    :param mode: 选择 'max_pi', 'sample' 或 'gaussian_mixture'
    :return: predictions, actuals
    """
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)  # 获取混合分布参数

            if mode == 'max_pi':
                # **方法 1: 选择权重最大的高斯分布**
                _, max_pi_indices = torch.max(pi, dim=1)  # 找到 pi 最大的索引
                best_mu = mu[torch.arange(mu.shape[0]), max_pi_indices]  # 取对应的 mu
                y_pred = best_mu

            elif mode == 'sample':
                # **方法 2: 在整个混合分布中采样**
                batch_size, num_gaussians, output_dim = mu.shape
                
                # **从 `pi` 采样一个高斯分布索引**
                categorical_dist = torch.distributions.Categorical(pi)
                sampled_indices = categorical_dist.sample()  # 形状: [batch_size]

                # **取出选中的高斯分布的参数**
                sampled_mu = mu[torch.arange(batch_size), sampled_indices]
                sampled_sigma = sigma[torch.arange(batch_size), sampled_indices]

                # **从该高斯分布中采样**
                normal_dist = torch.distributions.Normal(sampled_mu, sampled_sigma)
                y_pred = normal_dist.sample()  # 形状: [batch_size, output_dim]

            elif mode == 'gaussian_mixture':
                # **方法 3: 从合并的混合分布采样**
                batch_size, num_gaussians, output_dim = mu.shape

                # **计算合成的均值**
                mu_mixture = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # [batch_size, output_dim]

                # **计算合成的方差**
                sigma_squared_mixture = torch.sum(
                    pi.unsqueeze(-1) * (sigma ** 2 + (mu - mu_mixture.unsqueeze(1)) ** 2), dim=1
                )  # [batch_size, output_dim]
                sigma_mixture = torch.sqrt(sigma_squared_mixture)  # 取平方根得到标准差

                # **从合成的混合高斯分布采样**
                normal_dist = torch.distributions.Normal(mu_mixture, sigma_mixture)
                y_pred = normal_dist.sample()  # 形状: [batch_size, output_dim]

            else:
                raise ValueError("Invalid mode. Choose 'max_pi', 'sample', or 'gaussian_mixture'.")

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    return np.vstack(predictions), np.vstack(actuals)




def minmax_scale(data, min_val, max_val, feature_range=(0, 10)):
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    min_range = feature_range[0]
    data_scaled = (data - min_val) * scale + min_range
    return data_scaled

def inverse_minmax_scale(data_scaled, min_val, max_val, feature_range=(0, 10)):
    scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
    min_range = feature_range[0]
    data_original = (data_scaled - min_range) * scale + min_val
    return data_original


def load_real_data(aphy_file_path, rrs_file_path, seed=42, test_indices_path="test_indices_EMIT.npy"):
    torch.manual_seed(seed)  # 固定 PyTorch 随机种子
    np.random.seed(seed)     # 固定 NumPy 随机种子

    # 读取数据
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    Rrs_real = array2
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
    train_size = int(0.7 * dataset_size)
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


# def load_real_data(aphy_file_path, rrs_file_path):

#     array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
#     array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
#     Rrs_real = array2
#     a_phy_real = array1

#     input_dim = Rrs_real.shape[1]
#     output_dim = a_phy_real.shape[1]

#     min_val = 0
#     max_val = 3

#     scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

#     Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

#     Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
#     a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)
    
#     dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)

#     train_size = int(0.7 * len(dataset_real))
#     test_size = len(dataset_real) - train_size
#     train_dataset_real, test_dataset_real = random_split(dataset_real, [train_size, test_size])

#     train_real_dl = DataLoader(train_dataset_real, batch_size=1024, shuffle=True, num_workers=0)
#     test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=0)

#     return train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val

def load_real_test(aphy_file_path, rrs_file_path):

    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 3

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    #a_phy_real_normalized = np.array([minmax_scale(row, min_val, max_val, feature_range=(1, 10)) for row in a_phy_real])

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32) 
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=0)

    return test_real_dl, input_dim, output_dim, min_val, max_val


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val  = load_real_data('F:\\aphy-chla-predictions\\Data\\Clean\\EMIT\\aphy_EMIT.csv','F:\\aphy-chla-predictions\\Data\\Clean\\EMIT\\Rrs_EMIT.csv')
    test_real_Sep, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_EMIT_Sep.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_EMIT_Sep.csv')
    test_real_Oct, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_EMIT_Oct.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_EMIT_Oct.csv')

    save_dir = "F:\\aphy-chla-predictions\\plots\\MDN_aph_EMIT_sample"
    os.makedirs(save_dir, exist_ok=True)

    model = MDN(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    train_mdn(model, train_real_dl, epochs=2000)

    model.load_state_dict(torch.load('F:\\VAE for aphy-chla\\Model\\mdn_model_best_total.pth', map_location=device))

    predictions, actuals = evaluate_mdn(model, test_real_dl)
    #predictions_original = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions])
    #actuals_original = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals])
    

    predictions_Sep, actuals_Sep = evaluate_mdn(model, test_real_Sep)
    #predictions_original_Sep = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions_Sep])
    #actuals_original_Sep = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals_Sep])

    predictions_Oct, actuals_Oct = evaluate_mdn(model, test_real_Oct)
    #predictions_original_Oct = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions_Oct])
    #actuals_original_Oct = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals_Oct])

    plot_line_comparison_all(predictions, actuals, save_dir, mode='test')
    plot_results(predictions, actuals, save_dir, mode='test')

    plot_line_comparison_all(predictions_Sep, actuals_Sep, save_dir, mode='Sep')
    plot_results(predictions_Sep, actuals_Sep, save_dir, mode='Sep')

    plot_line_comparison_all(predictions_Oct, actuals_Oct, save_dir, mode='Oct')
    plot_results(predictions_Oct, actuals_Oct, save_dir, mode='Oct')  

    # plot_line_comparison_all(predictions_original, actuals_original, save_dir, mode='test')
    # plot_results(predictions_original, actuals_original, save_dir, mode='test')

    # plot_line_comparison_all(predictions_original_Sep, actuals_original_Sep, save_dir, mode='Sep')
    # plot_results(predictions_original_Sep, actuals_original_Sep, save_dir, mode='Sep')

    # plot_line_comparison_all(predictions_original_Oct, actuals_original_Oct, save_dir, mode='Oct')
    # plot_results(predictions_original_Oct, actuals_original_Oct, save_dir, mode='Oct')


