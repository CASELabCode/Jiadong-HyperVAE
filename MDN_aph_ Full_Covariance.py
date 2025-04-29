
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
from torch.distributions import Categorical, MultivariateNormal


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


class MixtureLayer(nn.Module):
    def __init__(self, n_mix, n_targets, epsilon, insize):
        super(MixtureLayer, self).__init__()

        self.n_mix = n_mix
        self.n_targets = n_targets
        self.epsilon = epsilon
        self.insize = insize
        self.linear = nn.Linear(self.insize, self.n_outputs)

    @property
    def layer_sizes(self):
        sizes = [1, self.n_targets, (self.n_targets * (self.n_targets + 1)) // 2]
        return self.n_mix * torch.tensor(sizes)

    @property
    def n_outputs(self):
        return sum(self.layer_sizes)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        prior, mu, scale = torch.split(outputs, self.layer_sizes.tolist(), dim=1)

        # Softmax 归一化
        prior = F.softmax(prior, dim=-1) + 1e-9
        mu = torch.stack(torch.split(mu, self.n_mix, dim=1), dim=2)

        # 处理协方差矩阵
        scale = torch.stack(torch.split(scale, self.n_mix, dim=1), dim=2)
        scale = fill_triangular(scale)
        norm = torch.eye(self.n_targets).unsqueeze(0).unsqueeze(0).to(scale.device)
        sigma = torch.einsum('abij,abjk->abik', scale.transpose(-1, -2), scale)
        sigma += self.epsilon * norm
        scale = torch.linalg.cholesky(sigma)

        prior = prior.view(-1, self.n_mix)
        mu = mu.view(-1, self.n_mix * self.n_targets)
        scale = scale.reshape(-1, self.n_mix * (self.n_targets ** 2))

        return torch.cat([prior, mu, scale], dim=1)

def fill_triangular(tensor, upper=False):
    batch_size, n_mix, num_elements = tensor.shape
    n_targets = int((1 + (8 * num_elements) ** 0.5) // 2)

    triangular = torch.zeros(batch_size, n_mix, n_targets, n_targets, device=tensor.device)
    rows, cols = torch.tril_indices(n_targets, n_targets, offset=0) if not upper else torch.triu_indices(n_targets, n_targets, offset=0)
    triangular[:, :, rows, cols] = tensor

    return triangular

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians=5, hidden=[100]*5, epsilon=1e-3):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], hidden[3]),
            nn.ReLU(),
            nn.Linear(hidden[3], hidden[4]),
            nn.ReLU()
        )

        self.mixture_layer = MixtureLayer(n_mix=num_gaussians, n_targets=output_dim, epsilon=epsilon, insize=hidden[-1])

    def forward(self, x):
        features = self.mlp(x)
        return self.mixture_layer(features)

    def loss(self, output, y):
        prior, mu, scale_tril = self._parse_outputs(output)

        mvn_components = [MultivariateNormal(mu[:, i, :], scale_tril=scale_tril[:, i, :, :]) for i in range(self.num_gaussians)]
        log_probs = torch.stack([mvn.log_prob(y) for mvn in mvn_components], dim=1)

        weighted_log_probs = log_probs + torch.log(prior)
        mixture_log_likelihood = torch.logsumexp(weighted_log_probs, dim=1)

        return -torch.mean(mixture_log_likelihood)

    def _parse_outputs(self, output):
        prior, mu, scale = torch.split(output, [self.num_gaussians, self.num_gaussians * self.output_dim, self.num_gaussians * self.output_dim * self.output_dim], dim=1)
        prior = prior.view(-1, self.num_gaussians)
        mu = mu.view(-1, self.num_gaussians, self.output_dim)
        scale = scale.reshape(-1, self.num_gaussians, self.output_dim, self.output_dim)
        return prior, mu, scale

    def sample(self, x, n_samples=1):
        output = self.forward(x)
        prior, mu, scale_tril = self._parse_outputs(output)

        categorical = Categorical(prior)
        mvn_components = [MultivariateNormal(mu[:, i, :], scale_tril=scale_tril[:, i, :, :]) for i in range(self.num_gaussians)]

        samples = []
        for _ in range(n_samples):
            indices = categorical.sample().unsqueeze(-1).expand(x.size(0), self.output_dim)
            component_samples = torch.stack([mvn.sample() for mvn in mvn_components], dim=1)
            samples.append(torch.gather(component_samples, 1, indices.unsqueeze(1)).squeeze(1))

        return torch.stack(samples).mean(0)

def train_mdn(model, train_dl, epochs=200, lr=1e-3, l2=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = model.loss(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dl):.4f}')

    torch.save(model.state_dict(), 'F:\\VAE for aphy-chla\\Model\\mdn_model_best_total.pth')

def evaluate_mdn(model, test_dl, mode='max_pi'):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            output = model(x)
            prior, mu, scale_tril = model._parse_outputs(output)

            if mode == 'max_pi':
                _, max_pi_indices = torch.max(prior, dim=1)
                best_mu = mu[torch.arange(mu.shape[0]), max_pi_indices]
                y_pred = best_mu

            elif mode == 'sample':
                y_pred = model.sample(x, n_samples=1)

            else:
                raise ValueError("Invalid mode. Choose 'max_pi' or 'sample'.")

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


def load_real_data(aphy_file_path, rrs_file_path, seed=42, test_indices_path="test_indices_PACE.npy"):
    torch.manual_seed(seed)  # 固定 PyTorch 随机种子
    np.random.seed(seed)     # 固定 NumPy 随机种子

    # 读取数据
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    log_Rrs_real = np.log10(array2+ 1e-6)

    min_val = 0
    max_val = 3

    # 归一化 Rrs_real
    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]
    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

    # 转换为 PyTorch 张量
    Rrs_real_tensor = torch.tensor(log_Rrs_real, dtype=torch.float32)
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
    log_Rrs_real = np.log10(array2+ 1e-6)

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 3

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    #a_phy_real_normalized = np.array([minmax_scale(row, min_val, max_val, feature_range=(1, 10)) for row in a_phy_real])

    Rrs_real_tensor = torch.tensor(log_Rrs_real, dtype=torch.float32) 
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=0)

    return test_real_dl, input_dim, output_dim, min_val, max_val
 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val  = load_real_data('F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\aphy_PACE.csv','F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\Rrs_PACE.csv')
    test_real_Sep, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Sep.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Sep.csv')
    test_real_Oct, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Oct.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Oct.csv')

    save_dir = "F:\\aphy-chla-predictions\\plots\\MDN_aph_PACE"
    os.makedirs(save_dir, exist_ok=True)

    model = MDN(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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


