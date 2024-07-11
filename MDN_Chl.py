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


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians=10):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        self.pi = nn.Linear(32, num_gaussians)
        self.sigma = nn.Linear(32, num_gaussians * output_dim)
        self.mu = nn.Linear(32, num_gaussians * output_dim)

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

def train_mdn(model, train_dl, epochs=200):
    model.train()
    min_total_loss = float('inf')
    best_model_total_path = 'F:\\Geo\\Model\\mdn_model_chl_best_total.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)
            loss = mdn_loss(pi, sigma, mu, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\Geo\\Model\\mdn_model.pth')

def evaluate_mdn(model, test_dl):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)
            m = torch.distributions.Normal(loc=mu, scale=sigma)
            y_samples = m.sample()  # Shape: [batch_size, num_gaussians, output_dim]
            pi_expanded = pi.unsqueeze(-1)  # Shape: [batch_size, num_gaussians, 1]
            weighted_samples = pi_expanded * y_samples  # Broadcasting to match shapes
            y_pred = torch.sum(weighted_samples, dim=1)  # Shape: [batch_size, output_dim]
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)


def load_real_data(aphy_file_path, rrs_file_path):
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    array1=array1.reshape(-1,1)

    Rrs_real = array2
    Chl_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]
    scaler_Chl_real = MinMaxScaler(feature_range=(0, 10))

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    Chl_real_normalized = scaler_Chl_real.fit_transform(Chl_real)

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    Chl_real_tensor = torch.tensor(Chl_real_normalized, dtype=torch.float32)


    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)

    train_size = int(0.7 * len(dataset_real))
    test_size = len(dataset_real) - train_size
    train_dataset_real, test_dataset_real = random_split(dataset_real, [train_size, test_size])


    train_real_dl = DataLoader(train_dataset_real, batch_size=1024, shuffle=True, num_workers=12)
    test_real_dl = DataLoader(test_dataset_real, batch_size=1024, shuffle=False, num_workers=12)

    return train_real_dl, test_real_dl, scaler_Chl_real, input_dim, output_dim


def load_real_test(aphy_file_path, rrs_file_path):
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    array1=array1.reshape(-1,1)

    Rrs_real = array2
    Chl_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]
    scaler_Chl_real = MinMaxScaler(feature_range=(0, 10))

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    Chl_real_normalized = scaler_Chl_real.fit_transform(Chl_real)

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    Chl_real_tensor = torch.tensor(Chl_real_normalized, dtype=torch.float32)


    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=12)

    return test_real_dl, scaler_Chl_real, input_dim, output_dim

def calculate_metrics(predictions, actuals, threshold=0.8):
    """
    Calculate epsilon, beta and additional metrics (RMSE, RMSLE, MAPE, Bias, MAE).
    
    :param predictions: array-like, predicted values
    :param actuals: array-like, actual values
    :param threshold: float, relative error threshold
    :return: epsilon, beta, rmse, rmsle, mape, bias, mae
    """
    # Apply the threshold to filter out predictions with large relative error
    mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]
    
    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / (filtered_actuals+1e-10))
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
    beta = 100 * np.sign(Z) * (10**np.abs(Z) - 1)
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((filtered_predictions - filtered_actuals) ** 2))
    rmsle = np.sqrt(np.mean((np.log10(filtered_predictions + 1) - np.log10(filtered_actuals + 1)) ** 2))
    mape = 100 * np.median(np.abs((filtered_predictions - filtered_actuals) / (filtered_actuals+1e-10)))
    bias = 10 ** (np.mean(np.log10(filtered_predictions) - np.log10(filtered_actuals+1e-10)))
    mae = np.mean(np.abs(np.log10(filtered_predictions) - np.log10(filtered_actuals+1e-10)))
    
    return epsilon, beta, rmse, rmsle, mape, bias, mae



def plot_results(predictions_rescaled, actuals_rescaled, save_dir, threshold=0.8, mode='test'):

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    log_actual = np.log10(np.where(actuals == 0, 1e-10, actuals))
    log_prediction = np.log10(np.where(predictions == 0, 1e-10, predictions))

    filtered_log_actual = np.log10(np.where(filtered_actuals == 0, 1e-10, filtered_actuals))
    filtered_log_prediction = np.log10(np.where(filtered_predictions == 0, 1e-10, filtered_predictions))

    
    epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(filtered_predictions, filtered_actuals,threshold)

    valid_mask = np.isfinite(filtered_log_actual) & np.isfinite(filtered_log_prediction)
    slope, intercept = np.polyfit(filtered_log_actual[valid_mask], filtered_log_prediction[valid_mask], 1)
    x = np.array([-2, 4])
    y = slope * x + intercept

    plt.figure(figsize=(6, 6))


    plt.plot(x, y, linestyle='--', color='blue', linewidth=0.8)
    lims = [-2, 4]
    plt.plot(lims, lims, linestyle='-',color='black', linewidth=0.8)

    sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.5)
    
    sns.kdeplot(x=filtered_log_actual, y=filtered_log_prediction, levels=3, color="black", fill=False, linewidths=0.8)

    plt.xlabel('Actual $Chla$ Values', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Predicted $Chla$ Values', fontsize=16, fontname='Times New Roman')
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.grid(True, which="both", ls="--")

    plt.legend(title=(f'MAE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n'
                    f'Bias = {bias:.2f}, Slope = {slope:.2f} \n'
                    f'MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%'),
            fontsize=16, title_fontsize=12, prop={'family': 'Times New Roman'})

    plt.xticks(fontsize=20, fontname='Times New Roman')
    plt.yticks(fontsize=20, fontname='Times New Roman')

    plt.savefig(os.path.join(save_dir, f'{mode}_plot.pdf'), bbox_inches='tight')
    plt.close()



def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def inverse_transform_by_row(scalers, data):
    return np.array([scalers[i].inverse_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(data)])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_real_dl, test_real_dl, scaler_Chl_real, input_dim, output_dim  = load_real_data('F:\\Geo\\Data\\Real\\Chl_RC_HICO.csv','F:\\Geo\\Data\\Real\\Rrs_RC_HICO.csv')
    test_real_Sep, scaler_Chl_real_Sep, _, _  = load_real_test('F:\\Geo\\Data\\Real\\Chl_RC_HICO_Sep.csv','F:\\Geo\\Data\\Real\\Rrs_RC_HICO_Sep.csv')
    test_real_Oct, scaler_Chl_real_Oct, _, _  = load_real_test('F:\\Geo\\Data\\Real\\Chl_RC_HICO_Oct.csv','F:\\Geo\\Data\\Real\\Rrs_RC_HICO_Oct.csv')

    save_dir = "F:\\Geo\\plots\\MDN_chla_HICO"
    os.makedirs(save_dir, exist_ok=True)

    model = MDN(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)

    train_mdn(model, train_real_dl, epochs=200)

    model.load_state_dict(torch.load('F:\\Geo\\Model\\mdn_model_chl_best_total.pth', map_location=device))

    predictions, actuals = evaluate_mdn(model, test_real_dl)
    predictions_rescaled = scaler_Chl_real.inverse_transform(predictions)
    actuals_rescaled = scaler_Chl_real.inverse_transform(actuals)

    predictions_Sep, actuals_Sep = evaluate_mdn(model, test_real_Sep)
    predictions_rescaled_Sep = scaler_Chl_real.inverse_transform(predictions_Sep)
    actuals_rescaled_Sep = scaler_Chl_real.inverse_transform(actuals_Sep)

    predictions_Oct, actuals_Oct = evaluate_mdn(model, test_real_Oct)
    predictions_rescaled_Oct = scaler_Chl_real.inverse_transform(predictions_Oct)
    actuals_rescaled_Oct = scaler_Chl_real.inverse_transform(actuals_Oct)


    save_to_csv(predictions_rescaled, os.path.join(save_dir, 'predictions_rescaled.csv'))
    save_to_csv(actuals_rescaled, os.path.join(save_dir, 'actuals_rescaled.csv'))

    plot_results(predictions_rescaled, actuals_rescaled, save_dir, mode='test')

    plot_results(predictions_rescaled_Sep, actuals_rescaled_Sep, save_dir, mode='Sep')

    plot_results(predictions_rescaled_Oct, actuals_rescaled_Oct, save_dir, mode='Oct')
