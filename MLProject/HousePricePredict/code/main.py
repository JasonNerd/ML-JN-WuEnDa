"""
in_project:     HousePricePredict
file_name:      main.py
create_by:      mrrai
create_time:    2022/10/29 20:33
description:    
"""
from datapreprocess import read_data


if __name__ == '__main__':
    train_features, train_labels, test_features = read_data("../data/train.csv", "../data/test.csv")

