import pandas as pd
import numpy as np

df = pd.read_csv('CaliforniaHousingPrices.csv')

df = pd.concat([df[df['ocean_proximity'] == '<1H OCEAN'],df[df['ocean_proximity'] == 'INLAND']])
df = df[df.columns[:-1]]

#Q1
print(" Q1")
nulls = df.isnull().sum()
print(nulls[nulls > 0])

#Q2
print(" Q2")
print(df['population'].median())

#Q3
print(" Q3")
df['median_house_value'] = np.log1p(df['median_house_value'])
##shuffle
n = len(df)
perc20 = int(n*0.2)

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
##split
df_train = df.iloc[idx[ : n-perc20*2]]
df_valid = df.iloc[idx[n-perc20*2 : n-perc20]]
df_test = df.iloc[idx[n-perc20 : ]]

df_train.reset_index(drop=True)
df_valid.reset_index(drop=True)
df_test.reset_index(drop=True)
##prepare y values
y_train = df_train['median_house_value'].values
del df_train['median_house_value']
y_valid = df_valid['median_house_value'].values
del df_valid['median_house_value']
y_test = df_test['median_house_value'].values
del df_test['median_house_value']
##fill nulls
train_bedrooms_mean = df_train['total_bedrooms'].mean()

def train_linear_regression(x,y):
    ones = np.ones(x.shape[0])
    x = np.column_stack([ones, x])

    XTX = x.T.dot(x)
    inv_XTX = np.linalg.inv(XTX)
    w_full = inv_XTX.dot(x.T).dot(y)

    return w_full[0], w_full[1:]
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
def prepare_x_f0(df):
    df_tmp = df
    df_tmp = df_tmp.fillna(0)
    x=df_tmp.values
    return x
def prepare_x_fmed(df):
    df_tmp = df
    df_tmp = df_tmp.fillna(train_bedrooms_mean)
    x=df_tmp.values
    return x

##train and validate fill with 0
x_train_f0 = prepare_x_f0(df_train)
w0_f0,w_f0 = train_linear_regression(x_train_f0,y_train)

x_val_f0 = prepare_x_f0(df_valid)
y_pred_f0 = w0_f0 + x_val_f0.dot(w_f0)
rmse_f0 = round(rmse(y_valid,y_pred_f0),2)
print("f0 rmse -> ", rmse_f0)

##train and validate fill with medium
x_train_fmed = prepare_x_fmed(df_train)
w0_fmed,w_fmed = train_linear_regression(x_train_fmed,y_train)

x_val_fmed = prepare_x_fmed(df_valid)
y_pred_fmed = w0_fmed + x_val_fmed.dot(w_fmed)
rmse_fmed = round(rmse(y_valid,y_pred_fmed),2)
print("fmed rmse -> ", rmse_fmed)

##Q4
print(" Q4")
## regularization
def train_linear_regression_reg(x, y, r=0.001):
    ones = np.ones(x.shape[0])
    x = np.column_stack([ones, x])

    XTX = x.T.dot(x)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(x.T).dot(y)
    
    return w_full[0], w_full[1:]

r_values = np.array([0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10])

for i in r_values:

    print(i)
    x_train_r = prepare_x_f0(df_train)
    w0_r,w_r = train_linear_regression_reg(x_train_r,y_train,i)

    x_val_r = prepare_x_f0(df_valid)
    y_pred_r = w0_r + x_val_r.dot(w_r)
    rmse_r = round(rmse(y_valid,y_pred_r),2)
    print("rmse with r=",i," -> ", rmse_r)

##Q5
print(" Q5")
seeds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
rmse_scores = np.zeros(len(seeds))

for i in range(len(seeds)):
    idx = np.arange(n)
    np.random.seed(seeds[i])
    np.random.shuffle(idx)

    df_train = df.iloc[idx[ : n-perc20*2]]
    df_valid = df.iloc[idx[n-perc20*2 : n-perc20]]
    df_test = df.iloc[idx[n-perc20 : ]]

    df_train.reset_index(drop=True)
    df_valid.reset_index(drop=True)
    df_test.reset_index(drop=True)

    y_train = df_train['median_house_value'].values
    del df_train['median_house_value']
    y_valid = df_valid['median_house_value'].values
    del df_valid['median_house_value']
    y_test = df_test['median_house_value'].values
    del df_test['median_house_value']

    x_train = prepare_x_f0(df_train)
    w0,w = train_linear_regression(x_train,y_train)

    x_val = prepare_x_f0(df_valid)
    y_pred = w0 + x_val.dot(w)
    rmse_score = round(rmse(y_valid,y_pred),2)
    rmse_scores[i] = rmse_score

print('rmse std = ',round(np.std(rmse_scores),3))

#Q6
print(" Q6")
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

df_train = df.iloc[idx[ : n-perc20*2]]
df_valid = df.iloc[idx[n-perc20*2 : n-perc20]]
df_test = df.iloc[idx[n-perc20 : ]]

df_train.reset_index(drop=True)
df_valid.reset_index(drop=True)
df_test.reset_index(drop=True)

y_train = df_train['median_house_value'].values
del df_train['median_house_value']
y_valid = df_valid['median_house_value'].values
del df_valid['median_house_value']
y_test = df_test['median_house_value'].values
del df_test['median_house_value']

x_train = prepare_x_f0(df_train)
w0,w = train_linear_regression_reg(x_train,y_train,0.001)

x_val = prepare_x_f0(df_valid)
y_pred = w0 + x_val.dot(w)
rmse_score = round(rmse(y_valid,y_pred),2)
print("rmse -> ", rmse_score)
