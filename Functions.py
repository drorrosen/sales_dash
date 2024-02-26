import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.data_cleaned = None


    def translate_column_names(self):
        column_translations = {
            '序号': 'Serial Number',
            '日期': 'Date',
            '类别': 'Category',
            '编号': 'ID',
            '收货退货': 'Delivery Returns',
            '型号': 'Model',
            '快递': 'Courier',
            '快递单号': 'Courier Tracking Number',
            '客户电话': 'Customer Phone',
            '数量': 'Quantity',
            '单价': 'Unit Price',
            '总价': 'Total Price',
            '快递费': 'Delivery Fee',
            '毛利': 'Gross Profit',
            '客服': 'Customer Service',
            'Commission(AED)': 'Commission (AED)'
        }

        self.data.rename(columns=column_translations, inplace=True)


    def clean_data(self):
        #columns_to_remove = ['Courier Tracking Number', 'Customer Phone', 'Customer Service']
        self.data_cleaned = self.data


        if 'Delivery Returns' in self.data_cleaned.columns:
            self.data_cleaned['Delivery Returns'].fillna('No Return', inplace=True)
            self.data_cleaned['Delivery Returns'] = self.data_cleaned['Delivery Returns'].replace({'rtn': 'Returned'})

        else:
            self.data_cleaned['Delivery Returns'] = 'No Return'

        self.data_cleaned = self.data_cleaned.dropna(axis=1, how='all')
        self.data_cleaned = self.data_cleaned.dropna(axis=0, how='all')

        self.data_cleaned = self.data_cleaned.loc[:, self.data_cleaned.nunique() > 1]

        self.data_cleaned = convert_date_column(self.data_cleaned, 'Date')
        #self.data_cleaned['Date'] = pd.to_datetime(self.data_cleaned['Date'], origin='1899-12-30', unit='D')


        if  'Delivery Fee' not in self.data_cleaned:
            self.data_cleaned['Delivery Fee'] = 0

    def calculate_net_profit(self):
        # Check if 'Commission (AED)' column exists
        if 'Commission (AED)' in self.data_cleaned.columns:
            self.data_cleaned['Net Profit'] = self.data_cleaned['Gross Profit'] + self.data_cleaned['Commission (AED)'] - self.data_cleaned['Delivery Fee']
        else:
            # If there's no 'Commission (AED)' column, proceed without it
            self.data_cleaned['Net Profit'] = self.data_cleaned['Gross Profit'] - self.data_cleaned['Delivery Fee']


def convert_date_column(df, date_column_name):
    # Check if the date column is already in datetime format
    if pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
        print("Date column is already in datetime format.")
    else:
        # Assume the column is in Excel's serial date format and convert
        print("Converting date column from Excel's serial date format to datetime.")
        df[date_column_name] = pd.to_datetime(df[date_column_name], origin='1899-12-30', unit='D')

    return df

def add_net_profit_lags(data, n_lags):
    for lag in range(1, n_lags + 1):
        data[f'Net_Profit_lag_{lag}'] = data['Net Profit'].shift(lag)
    return data


def file_upload():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        return df
    return None


def format_number(number):
    return f"{number:,.2f}"