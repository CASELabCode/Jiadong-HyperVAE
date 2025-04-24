
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians=5, hidden=[100, 100, 100, 100, 100], epsilon=1e-3):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon  # 防止数值不稳定
        
        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())  
            prev_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        
        self.pi = nn.Linear(prev_dim, num_gaussians)
        self.mu = nn.Linear(prev_dim, num_gaussians * output_dim)
        self.l = nn.Linear(prev_dim, num_gaussians * (output_dim * (output_dim + 1)) // 2)

    def forward(self, x):
        x = self.hidden_layers(x)
        
        pi = F.softmax(self.pi(x), dim=-1) + 1e-9  # 避免 log(0)
        mu = self.mu(x).view(-1, self.num_gaussians, self.output_dim)

        l_elements = self.l(x)
        batch_size = x.size(0)

        tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0)
        L = torch.zeros(batch_size, self.num_gaussians, self.output_dim, self.output_dim).to(device)
        l_elements = l_elements.view(batch_size, self.num_gaussians, -1)

        L[:, :, tril_indices[0], tril_indices[1]] = l_elements

        diag_indices = torch.arange(self.output_dim, device=device)
        L[:, :, diag_indices, diag_indices] = F.softplus(L[:, :, diag_indices, diag_indices]) + 1e-2  # 增加稳定性

        sigma = L @ L.transpose(-1, -2)
        sigma += torch.eye(self.output_dim, device=device) * self.epsilon  # 加 jitter

        try:
            L = torch.linalg.cholesky(sigma)  # 计算 Cholesky 分解
        except RuntimeError:
            print("Warning: Cholesky decomposition failed due to Σ not being positive definite.")
            print(sigma)
            raise

        return pi, mu, L

def mdn_loss(pi, mu, L, y, model, l2=1e-3):
    batch_size, num_gaussians, output_dim = mu.shape
    
    mvn = torch.distributions.MultivariateNormal(mu, scale_tril=L)

    y = y.unsqueeze(1).expand_as(mu)

    log_probs = mvn.log_prob(y)
    log_pi = torch.log(pi + 1e-6)  
    weighted_log_probs = log_probs + log_pi

    l2_loss = sum(torch.sum(p**2) for p in model.parameters())  
    loss = -torch.logsumexp(weighted_log_probs, dim=1).mean() + l2 * l2_loss
    return loss

def train_mdn(model, train_dl, epochs=200, lr=1e-3, l2=1e-3):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)

    model.train()
    min_total_loss = float('inf')
    best_model_total_path = 'F:\\aphy-chla-predictions\\Model\\mdn_model_best_total.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pi, mu, L = model(x)
            loss = mdn_loss(pi, mu, L, y, model, l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'Epoch {epoch + 1}, Loss: {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\aphy-chla-predictions\\Model\\mdn_model_best_total.pth')

def evaluate_mdn(model, test_dl):
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pi, mu, L = model(x)
            
            y_pred = torch.sum(pi.unsqueeze(-1) * mu, dim=1)

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    return np.vstack(predictions), np.vstack(actuals)


def load_real_data(aphy_file_path, rrs_file_path):

    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
    Rrs_real = array2
    a_phy_real = array1

    # Rrs_real=np.log(Rrs_real + 1)*10
    # a_phy_real=np.log(a_phy_real + 1)*10

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 2.4

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    #a_phy_real_normalized = np.array([minmax_scale(row, min_val, max_val, feature_range=(1, 10)) for row in a_phy_real])

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)

    train_size = int(0.8 * len(dataset_real))
    test_size = len(dataset_real) - train_size
    train_dataset_real, test_dataset_real = random_split(dataset_real, [train_size, test_size])

    train_real_dl = DataLoader(train_dataset_real, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val

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
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=12)

    return test_real_dl, input_dim, output_dim, min_val, max_val



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val  = load_real_data('F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\aphy_PACE.csv','F:\\aphy-chla-predictions\\Data\\Clean\\PACE\\Rrs_PACE.csv')
    test_real_Sep, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Sep.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Sep.csv')
    test_real_Oct, _, _,_,_  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\aphy_RA_PACE_Oct.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RA_PACE_Oct.csv')

    save_dir = "F:\\aphy-chla-predictions\\plots\\MDN_aph_PACE"
    os.makedirs(save_dir, exist_ok=True)

    model = MDN(input_dim, output_dim).to(device)
    #opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_mdn(model, train_real_dl, epochs=200)

    model.load_state_dict(torch.load('F:\\aphy-chla-predictions\\Model\\mdn_model_best_total.pth', map_location=device))

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