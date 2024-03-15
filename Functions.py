import numpy as np
import streamlit as st
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError
from io import BytesIO
import pandas as pd
import io







class DataFrameProcessor:
    def __init__(self, df):
        self.df = df

    def remove_unnamed_columns(self):
        columns_to_drop = [str(col) for col in self.df.columns if str(col).startswith('Unnamed:')]
        self.df = self.df.drop(columns=columns_to_drop)
        return self

    def clean_dataframe(self):
        self.df = self.df.dropna(axis=0, how='all')
        self.df = self.df.dropna(axis=1, how='all')
        return self

    def translate_columns(self):
        translation_map = {
            '序号': 'Serial Number', '日期': 'Date', '类别': 'Category', '编号': 'Number',
            '收货退货': 'Received/Returned', '型号': 'Model', '快递': 'Courier',
            '快递单号': 'Courier Tracking Number', '客户电话': 'Customer Phone',
            '数量': 'Quantity', '单价': 'Unit Price', '总价': 'Total Price',
            '快递费': 'Shipping Fee', '毛利': 'Gross Profit', '客服': 'Customer Service',
            '收款': 'Payment Received', '收款员': 'Cashier', '备注': 'Remarks',
            '快递.1': 'Courier Additional Info', '收货/退货': 'Received/Returned Alt',
            '是否结账': 'Payment Settled', '结账人员': 'Settlement Clerk',
            '退货/作废取消单': 'Return/Cancellation Order Voided', '客户地址': 'Customer Address',
            '销售统计表': 'Sales Summary Table', '提成': 'commission'
        }
        self.df.columns = [translation_map.get(col, col) for col in self.df.columns]
        return self

    def convert_date_column(self, date_column_name):
        # Check if the date column is already in datetime format
        if pd.api.types.is_datetime64_any_dtype(self.df[date_column_name]):
            print("Date column is already in datetime format.")
        else:
            print("Checking date column format...")
            # Temporarily convert column to numeric, with non-numeric as NaN (without altering original DataFrame)
            temp_numeric_series = pd.to_numeric(self.df[date_column_name], errors='coerce')

            # If the conversion results entirely in NaN, it means original data might already be in a correct format or non-numeric
            if temp_numeric_series.isna().all():
                print("Column appears to be non-numeric or already formatted. Skipping conversion.")
            else:
                # Assume the column contains Excel's serial date format and convert
                print("Converting date column from Excel's serial date format to datetime.")
                self.df[date_column_name] = pd.to_datetime(temp_numeric_series, origin='1899-12-30', unit='D', errors='coerce')

        return self.df

    def process(self):
        return self.remove_unnamed_columns() \
            .clean_dataframe() \
            .translate_columns() \
            .convert_date_column('Date')




def calculate_net_profit(df):
        # Check if 'Commission (AED)' column exists
        if 'Commission (AED)' in df.columns:
            df['Net Profit'] = df['Gross Profit'] + df['Commission (AED)'] - df['Shipping Fee']
        else:
            # If there's no 'Commission (AED)' column, proceed without it
            df['Net Profit'] = df['Gross Profit'] - df['Shipping Fee']

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


###connecting to dropbox
APP_KEY = '3638tax2xlh5v7z'
APP_SECRET = 'vbh0tikcuq4vn21'
REFRESH_TOKEN = 'MK_Mzc-1gfAAAAAAAAAAATYGHPB8MeL8Kswb3jfeTpHotBlWfcV0rS4ZfrGjHgwV'




dbx = dropbox.Dropbox(
    oauth2_refresh_token=REFRESH_TOKEN,
    app_key=APP_KEY,
    app_secret=APP_SECRET
)


def upload_dataframe_to_dropbox(df, target_path):
    """
    Uploads a DataFrame to Dropbox as a CSV.
    :param df: DataFrame to upload.
    :param target_path: Target path in Dropbox, including the file name (e.g., '/myfolder/myfile.csv').
    """
    # Convert DataFrame to CSV and then to bytes
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    data = buffer.getvalue().encode()


    # Upload the byte stream to Dropbox
    try:
        dbx.files_upload(data, target_path, mode=WriteMode('overwrite'))
        print(f"DataFrame uploaded to '{target_path}'.")
    except ApiError as e:
        print(f"Error uploading DataFrame: {e}")



def download_file_from_dropbox(dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    return res.content


# Mapping function
def map_to_returned(category):
    returned_keywords = ['rtn', 'retn', 'cancel', 'exchange', 'missing', 'excharge']  # Assuming 'excharge' is a typo for 'exchange'
    if category in returned_keywords:
        return 'returned'
    else:
        return 'not_returned'

