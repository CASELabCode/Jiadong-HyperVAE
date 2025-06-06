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

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # encoder
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.fc1 = nn.Linear(128, 64) 
        self.fc2 = nn.Linear(128, 64)  

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),
            #nn.Softplus()
        )

    def encode(self, x):
        x = self.encoder_layer(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)*0.001
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    L1 = F.l1_loss(recon_x, x, reduction='mean')
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    return MSE




def train(model, train_dl, epochs=200):
    model.train()

    min_total_loss = float('inf')

    best_model_total_path = 'F:\\aphy-chla-predictions\\Model\\vae_trans_model_best_Chl.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            y_pred, mu, log_var = model(x)
            loss = loss_function(y_pred, y, mu, log_var) 
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
            y_pred, _, _ = model(x)

            y_pred = 10 ** y_pred  # 还原回 Chl
            y_actual = 10 ** y  # 还原实际值


            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_actual.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)


def load_real_data(aphy_file_path, rrs_file_path):
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    array1=array1.reshape(-1,1)

    Rrs_real = array2
    Chl_real = np.log10(array1+ 1e-10)
  

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32, device=device)
    Chl_real_tensor = torch.tensor(Chl_real, dtype=torch.float32, device=device)

    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)

    train_size = int(0.7 * len(dataset_real))
    test_size = len(dataset_real) - train_size
    train_dataset_real, test_dataset_real = random_split(dataset_real, [train_size, test_size])


    train_real_dl = DataLoader(train_dataset_real, batch_size=train_size, shuffle=True, num_workers=0)
    test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=0)

    return train_real_dl, test_real_dl, input_dim, output_dim


def load_real_test(aphy_file_path, rrs_file_path):
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    array1=array1.reshape(-1,1)

    Rrs_real = array2
    Chl_real = np.log10(array1+ 1e-6)
    

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    Chl_real_tensor = torch.tensor(Chl_real, dtype=torch.float32)


    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=0)

    return test_real_dl, input_dim, output_dim

def calculate_metrics(predictions, actuals):
    """
    Calculate epsilon, beta and additional metrics (RMSE, RMSLE, MAPE, Bias, MAE).
    
    :param predictions: array-like, predicted values
    :param actuals: array-like, actual values
    :param threshold: float, relative error threshold
    :return: epsilon, beta, rmse, rmsle, mape, bias, mae
    """
    
    # Calculate epsilon and beta
    log_ratios = np.log10(predictions/ (actuals+1e-7))
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
    beta = 100 * np.sign(Z) * (10**np.abs(Z) - 1)
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))/2
    rmsle = np.sqrt(np.mean((np.log10(predictions + 1) - np.log10(actuals + 1)) ** 2))
    mape = 100 * np.median(np.abs((predictions - actuals+1e-7) / (actuals+1e-7)))
    bias = 10 ** (np.mean(np.log10(predictions+1e-7) - np.log10(actuals+1e-7)))
    mae = 10** np.mean(np.abs(np.log10(predictions+1e-7) - np.log10(actuals+1e-7)))
    
    return epsilon, beta, rmse, rmsle, mape, bias, mae



def plot_results(predictions_rescaled, actuals_rescaled, save_dir, mode='test'):

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    log_actual = np.log10(np.where(actuals == 0, 1e-6, actuals))
    log_prediction = np.log10(np.where(predictions == 0, 1e-6, predictions))

    epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(predictions, actuals)

    valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)

    slope, intercept = np.polyfit(log_actual[valid_mask], log_prediction[valid_mask], 1)
    x = np.array([-2, 4])
    y = slope * x + intercept

    plt.figure(figsize=(6, 6))


    plt.plot(x, y, linestyle='--', color='blue', linewidth=0.8)
    lims = [-2, 4]
    plt.plot(lims, lims, linestyle='-',color='black', linewidth=0.8)

    sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.5)
    
    sns.kdeplot(x=log_actual, y=log_prediction, levels=3, color="black", fill=False, linewidths=0.8)

    plt.xlabel('Actual $Chla$ Values', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Predicted $Chla$ Values', fontsize=16, fontname='Times New Roman')
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.grid(True, which="both", ls="--")

    plt.legend(title=(f'MALE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n'
                    f'Log-Bias = {bias:.2f}, Slope = {slope:.2f} \n'
                    f'MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%'),
            fontsize=16, title_fontsize=12, prop={'family': 'Times New Roman'})

    plt.xticks(fontsize=20, fontname='Times New Roman')
    plt.yticks(fontsize=20, fontname='Times New Roman')

    plt.savefig(os.path.join(save_dir, f'{mode}_plot.pdf'), bbox_inches='tight')
    plt.close()



def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_real_dl, test_real_dl, input_dim, output_dim  = load_real_data('F:\\aphy-chla-predictions\\Data\\Clean\\EMIT\\Chla_EMIT.csv','F:\\aphy-chla-predictions\\Data\\Clean\\EMIT\\Rrs_EMIT_chla.csv')
    test_real_Sep, _, _  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\Chl_RC_EMIT_Sep.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RC_EMIT_Sep.csv')
    test_real_Oct, _, _  = load_real_test('F:\\aphy-chla-predictions\\Data\\Real\\Chl_RC_EMIT_Oct.csv','F:\\aphy-chla-predictions\\Data\\Real\\Rrs_RC_EMIT_Oct.csv')


    save_dir = "F:\\aphy-chla-predictions\\plots\\VAE_Chla_EMIT_2"
    os.makedirs(save_dir, exist_ok=True)

    # 创建VAE模型及优化器
    model = VAE(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-10)

    train(model, train_real_dl, epochs=3000)

    model.load_state_dict(torch.load('F:\\aphy-chla-predictions\\Model\\vae_trans_model_best_Chl.pth', map_location=device))

    predictions, actuals = evaluate(model, test_real_dl)
    predictions_Sep, actuals_Sep = evaluate(model, test_real_Sep)
    predictions_Oct, actuals_Oct = evaluate(model, test_real_Oct)

 
    plot_results(predictions, actuals, save_dir, mode='test')
    plot_results(predictions_Sep, actuals_Sep, save_dir, mode='Sep')
    plot_results(predictions_Oct, actuals_Oct, save_dir, mode='Oct')


