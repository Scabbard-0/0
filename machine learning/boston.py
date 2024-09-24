import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data_describe(df):
    # 查看数据的描述信息，在描述信息里可以看到每个特征的均值，最大值，最小值等信息
    df.describe()

def read_data(file_path):
    return pd.read_csv(file_path, sep=',')

def get_training_data(df, num):
    # 取测试集的前450项
    df_training = df.head(num)
    # print(df_training)
    return df_training

def get_test_data(df, num):
    # 取数据集的后50项
    df_test = df.tail(num)
    # print(df_test)
    return df_test

def get_last_column(df):
    return df.iloc[:, -1]

def get_not_last_column(df):
    return df.iloc[:, :(df.shape[1] - 1)]

def dataframe_to_matrix(df):
    return df.to_numpy()

def model_calculation(df_x, df_y):

    #计算模型
    one_matrix = np.full(len(df_x), 1, dtype=int)
    x_matrix = np.insert(dataframe_to_matrix(df_x), 0, values=one_matrix, axis=1)
    y_matrix = dataframe_to_matrix(df_y)
    beta = np.dot(np.linalg.inv(np.dot(x_matrix.T, x_matrix)), np.dot(x_matrix.T, y_matrix))
    return beta

def get_estimate(beta, df_test):

    estimate = np.full(len(df_test), 0, dtype=int)
    one_matrix = np.full(len(df_test), 1, dtype=int)
    test_matrix = dataframe_to_matrix(get_not_last_column(df_test))
    test_matrix = np.insert(test_matrix, 0, values=one_matrix, axis=1)
    for i in range(np.size(test_matrix, 0)):
        estimate[i] = np.dot(beta, test_matrix[i])
    return estimate


if __name__ == '__main__':
    path = 'boston.csv'
    # 读取并分配集
    df_housing_data = read_data(path)
    df_training_data = get_training_data(df_housing_data, 450)
    df_test_data = get_test_data(df_housing_data, 56)

    # 训练
    beta_matrix = model_calculation(get_not_last_column(df_training_data), get_last_column(df_training_data))
    estimate_value_price = get_estimate(beta_matrix, df_test_data)
    true_value_price = dataframe_to_matrix(get_last_column(df_test_data))

    # 绘图
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(true_value_price, color="r", label="实际价格")
    plt.plot(estimate_value_price, color=(0, 0, 0), label="预测价格")
    plt.xlabel("测试序号")
    plt.ylabel("价格")
    plt.title("实际值与预测值折线图")
    plt.legend()
    plt.show()