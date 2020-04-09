import time
import numpy as np
import csv
from itertools import zip_longest
from itertools import chain
import pandas as pd


# # convert json file to csv
# data = []
# with open('data/endomondoHR_proper.json') as f:
#     for l in f:
#         data.append(eval(l))
#
# df = pd.DataFrame(data)
# df.to_csv('mycsvfile.csv')


# # process of filtered data
# def split_dataframe_rows(df, column_selectors, row_delimiter):
#     # we need to keep track of the ordering of the columns
#     def _split_list_to_rows(row, row_accumulator, column_selector, row_delimiter):
#         split_rows = {}
#         max_split = 0
#         for column_selector in column_selectors:
#
#             split_row = str(row[column_selector]).strip('[').strip(']').split(row_delimiter)
#             split_rows[column_selector] = split_row
#             if len(split_row) > max_split:
#                 max_split = len(split_row)
#
#         for i in range(max_split):
#             new_row = row.to_dict()
#             for column_selector in column_selectors:
#                 try:
#                     new_row[column_selector] = split_rows[column_selector].pop(0)
#                 except IndexError:
#                     new_row[column_selector] = ''
#             row_accumulator.append(new_row)
#
#     new_rows = []
#     df.apply(_split_list_to_rows, axis=1, args=(new_rows, column_selectors, row_delimiter))
#     new_df = pd.DataFrame(new_rows, columns=df.columns)
#     return new_df
#
# data = pd.read_csv('mycsvfile.csv',nrows=40)
#
# new_data = split_dataframe_rows(data,['altitude','heart_rate','latitude','longitude','speed','timestamp'],',')
#
# new_data = new_data.drop(['Unnamed: 0'], axis=1)
# new_data.to_csv('sample2.csv')


# # process of cleaned data
# data = np.load("data/processed_endomondoHR_proper_interpolate.npy")
# MyEmptydf = pd.DataFrame()
# for i in range(5000,10001):
#     if i%100 == 0:
#         print(i)
#     df = pd.DataFrame(data[0][i])
#     MyEmptydf = pd.concat([MyEmptydf,df])
#
#
# MyEmptydf.to_csv('mycsvfile_10000.csv',index=False)


# data1 = pd.read_csv('data/mycsvfile_5000.csv')
# data2 = pd.read_csv('data/mycsvfile_10000.csv')
# data = pd.concat([data1,data2])
#
# data = data[['altitude','derived_speed','distance','gender','heart_rate','id','latitude','longitude','since_begin','since_last','sport','time_elapsed','timestamp','userId','tar_derived_speed','tar_heart_rate']]
# data.to_csv('data/processed_data.csv',index=False)

df = pd.read_csv('data/processed_data.csv')
print(df.userId.unique())
print(len(df.userId.unique()))
# new_df = df.groupby('userId')
# d = dict(tuple(new_df))
# i=0
# for i, g in new_df:
#     print('current user is :',i)
#     globals()['df_' + str(i)] =  g
#     globals()['df_' + str(i)].to_csv('user'+str(i)+'.csv',index=False)



