import random
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from tsai.basics import *

import argparse
from exp.exp_main import Exp_Main
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST

# from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

parser = argparse.ArgumentParser(description="PatchTST for Time Series Forecasting")
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--seq_len", type=int, default=90, help="input sequence length")
parser.add_argument(
    "--pred_len", type=int, default=21, help="prediction sequence length"
)
parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers")
parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
parser.add_argument("--d_model", type=int, default=16, help="dimension of model")
parser.add_argument("--d_ff", type=int, default=128, help="dimension of fcn")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")

# PatchTST
parser.add_argument(
    "--fc_dropout", type=float, default=0.3, help="fully connected dropout"
)
parser.add_argument("--head_dropout", type=float, default=0, help="head dropout")
parser.add_argument("--patch_len", type=int, default=16, help="patch length")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument(
    "--padding_patch", default="end", help="None: None; end: padding on the end"
)
parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")
parser.add_argument(
    "--affine", type=int, default=0, help="RevIN-affine; True 1 False 0"
)
parser.add_argument(
    "--subtract_last",
    type=int,
    default=0,
    help="0: subtract mean; 1: subtract last",
)
parser.add_argument(
    "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
)
parser.add_argument("--kernel_size", type=int, default=25, help="decomposition-kernel")
parser.add_argument(
    "--individual", type=int, default=0, help="individual head; True 1 False 0"
)

args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CFG = {
    "TRAIN_WINDOW_SIZE": 90,  # 90일치로 학습
    "PREDICT_SIZE": 21,  # 21일치 예측
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 4096,
    "SEED": 41,
    "LAMBDA1": 0.1,
    "LAMBDA2": 10,
    "LR_LAMBDA": 0.80,  # ddd
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])  # Seed 고정

train_data = pd.read_csv("./train.csv").drop(columns=["ID", "제품"])
output_kuka = train_data.iloc[:, -CFG["PREDICT_SIZE"] :].to_numpy()
product_features = pd.read_csv("./features.csv").drop(
    columns=["Unnamed: 0", "Unnamed: 0.1", "ID", "제품", "대분류", "중분류", "소분류", "브랜드"]
)
time_series_features = pd.read_csv("./total_dates_scaling.csv").drop(
    columns=["Unnamed: 0", "Date"]
)
# (15890, 459)
# price = pd.read_csv('./price.csv').iloc[:,7:]
# sale_event = pd.read_csv('./sale_event.csv').drop(columns=['Unnamed: 0'])

# preprocessing
numeric_cols = train_data.columns[4:]
min_values = train_data[numeric_cols].min(axis=1)
max_values = train_data[numeric_cols].max(axis=1)
ranges = max_values - min_values
ranges[ranges == 0] = 1
train_data[numeric_cols] = (train_data[numeric_cols].subtract(min_values, axis=0)).div(
    ranges, axis=0
)
scale_min_dict = min_values.to_dict()
scale_max_dict = max_values.to_dict()

# Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ["대분류", "중분류", "소분류", "브랜드"]
categorical_nums = [5, 11, 53, 3170]

for i, col in enumerate(categorical_columns):
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
    train_data[col] = train_data[col].div(categorical_nums[i] - 1)  # minmax scaling

# (15890, 1)로 만들기
scaler1 = MinMaxScaler()
scaler1.fit(product_features[["판매량평균"]])
sales_mean_scaled = scaler1.transform(product_features[["판매량평균"]])

scaler2 = MinMaxScaler()
scaler2.fit(product_features[["제품수"]])
prod_num_scaled = scaler2.transform(product_features[["제품수"]])

nprice_mid = product_features[["NormalizedPrice_중분류"]].to_numpy()

product = np.column_stack((sales_mean_scaled, prod_num_scaled, nprice_mid))

# (459, 1)
# weekend_salary = (time_series_features[['Weekend']].to_numpy() + 0.5*time_series_features[['Salary']].to_numpy() + 0.5*np.where(time_series_features[['DayofWeek']]==4/6, 1, 0)).squeeze()
sale_info = time_series_features["SaleInfo"].to_numpy()
holiday = time_series_features["Holiday"].to_numpy()
salary = time_series_features["Salary"].to_numpy()
month = time_series_features["Month"].to_numpy()
dayofweek = time_series_features["DayofWeek"].to_numpy()

time_series = np.stack((sale_info, salary, month, dayofweek))


def make_train_data(
    data, train_size=CFG["TRAIN_WINDOW_SIZE"], predict_size=CFG["PREDICT_SIZE"]
):
    """
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    """
    num_rows = len(data)
    window_size = train_size + predict_size

    input_data = np.empty(
        (
            num_rows * (len(data.columns) - window_size + 1),
            train_size,
            1 + product.shape[1] + time_series.shape[0] + 1,
        )
    )
    target_data = np.empty(
        (
            num_rows * (len(data.columns) - window_size + 1),
            predict_size,
            1 + product.shape[1] + time_series.shape[0] + 1,
        )
    )
    # target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    for i in tqdm(range(num_rows), dynamic_ncols=True):
        encode_info = np.array(data.iloc[i, 0])  # 대분류 소분류
        product_data = product[i, :]
        # time_series
        sales_data = np.array(data.iloc[i, 4:])
        # price_data = np.array(price.iloc[i, :])
        # sale_event_data = np.array(sale_event.iloc[i, :])

        for j in range(len(sales_data) - window_size + 1):
            time_series_window = time_series[:, j : j + window_size]
            # price_window = price_data[j : j + window_size]
            # sale_event_window = sale_event_data[j : j + window_size]
            window = sales_data[j : j + window_size]

            temp_data = np.concatenate(
                (
                    np.tile(encode_info, (train_size, 1)),
                    np.tile(product_data, (train_size, 1)),
                    time_series_window[:, :train_size].T,
                    window[:train_size].reshape((-1, 1)),
                ),
                axis=1,
            )
            temp_target = np.concatenate(
                (
                    np.tile(encode_info, (predict_size, 1)),
                    np.tile(product_data, (predict_size, 1)),
                    time_series_window[:, train_size:].T,
                    window[train_size:].reshape((-1, 1)),
                ),
                axis=1,
            )
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = temp_target

    return input_data, target_data


def make_predict_data(data, train_size=CFG["TRAIN_WINDOW_SIZE"]):
    """
    평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
    data : 일별 판매량
    train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
    """
    num_rows = len(data)

    input_data = np.empty(
        (num_rows, train_size, 1 + product.shape[1] + time_series.shape[0] + 1)
    )

    for i in tqdm(range(num_rows), dynamic_ncols=True):
        encode_info = np.array(data.iloc[i, 0])
        product_data = product[i, :]
        sales_data = np.array(data.iloc[i, -train_size:])
        # price_data = np.array(price.iloc[i, -train_size:])
        # sale_event_data = np.array(sale_event.iloc[i, -train_size:])

        time_series_window = time_series[:, -train_size:]
        # price_window = price_data[-train_size:]
        # sale_event_window = sale_event_data[-train_size:]
        window = sales_data[-train_size:]

        temp_data = np.concatenate(
            (
                np.tile(encode_info, (train_size, 1)),
                np.tile(product_data, (train_size, 1)),
                time_series_window[:, :train_size].T,
                window[:train_size].reshape((-1, 1)),
            ),
            axis=1,
        )
        input_data[i] = temp_data

    return input_data


# train_input, train_target = make_train_data(train_data)  # train val total
# train_input, train_target = make_train_data(
#     train_data.iloc[:, : -CFG["PREDICT_SIZE"]]
# )  # train val
# test_input_kuka = make_predict_data(
#     train_data.iloc[:, : -CFG["PREDICT_SIZE"]]
# )  # test(kuka pre-submission)
test_input = make_predict_data(train_data)  # test(submission)


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)


# dataset = CustomDataset(train_input, train_target)

# data_len = len(dataset)
# train_size = int(data_len * 0.8)
# val_size = int(data_len * 0.2)

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(
#     train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=0
# )
# val_loader = DataLoader(
#     val_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0
# )

test_dataset = CustomDataset(test_input, None)
test_loader = DataLoader(
    test_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0
)


class PSFALoss(nn.Module):
    def __init__(self):
        super(PSFALoss, self).__init__()

    def forward(self, inputs, targets):
        share_denominator = torch.sum(targets, axis=0)
        share_denominator[share_denominator == float(0)] = 1  # 나눠지게 만들자
        share = targets / share_denominator
        error_demoninator = torch.max(inputs, targets)
        error_demoninator[error_demoninator == float(0)] = 1  # 나눠지게 만들자
        error = torch.abs(inputs - targets) / error_demoninator
        metric = error * share
        loss = torch.mean(torch.sum(metric, axis=1))
        return loss


def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    psfa = PSFALoss().to(device)
    best_loss = 9999999
    best_model = None

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_loader), dynamic_ncols=True):
            X = X.to(device)
            Y = Y.to(device)
            Y_sales = Y[:, :, -1]
            optimizer.zero_grad()

            output = model(X)
            output_sales = output[:, :, -1]
            loss = CFG["LAMBDA1"] * criterion(output, Y) + CFG["LAMBDA2"] * psfa(
                output_sales, Y_sales
            )

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        scheduler.step()  # added

        val_loss = validation(model, val_loader, criterion, psfa, device)
        print(
            f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]"
        )

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print("Model Saved")
    return best_model


def validation(model, val_loader, criterion, psfa, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader), dynamic_ncols=True):
            X = X.to(device)
            Y = Y.to(device)
            Y_sales = Y[:, :, -1]
            output = model(X)
            output_sales = output[:, :, -1]
            loss = CFG["LAMBDA1"] * criterion(output, Y) + CFG["LAMBDA2"] * psfa(
                output_sales, Y_sales
            )

            val_loss.append(loss.item())
    return np.mean(val_loss)


# model = PatchTST.Model(args)
# print(torch.cuda.device_count())
# model = nn.DataParallel(model)  # parallel mode
# optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
# scheduler = optim.lr_scheduler.LambdaLR(
#     optimizer=optimizer, lr_lambda=lambda epoch: CFG["LR_LAMBDA"] ** epoch
# )
# infer_model = train(model, optimizer, scheduler, train_loader, val_loader, device)

# torch.save(infer_model.state_dict(), "./patchtst_submit_0822_ver3.pth")


# 저장된 파일 불러오기

infer_model = PatchTST.Model(args).to(device)
infer_model = nn.DataParallel(infer_model)  # parallel mode

infer_model.load_state_dict(torch.load("./patchtst_submit_0822_ver3.pth"))


def inference(model, test_loader, device):
    predictions = []

    with torch.no_grad():
        for X in tqdm(iter(test_loader), dynamic_ncols=True):
            X = X.to(device)

            output = model(X)

            # 모델 출력인 output을 CPU로 이동하고 numpy 배열로 변환
            output = output.cpu().numpy()

            predictions.extend(output)

    return np.array(predictions)


pred = inference(infer_model, test_loader, device)
# 추론 결과를 inverse scaling
for idx in range(len(pred)):
    pred[idx, :] = (
        pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    )

# 결과 후처리
np.savetxt("save.txt", pred[:, :, -1])
pred = np.round(pred[:, :, -1], 0).astype(int)
submit = pd.read_csv("./sample_submission.csv")
submit.iloc[:, 1:] = pred
submit.to_csv("./patchtst_submit_0822_ver3.csv", index=False)
