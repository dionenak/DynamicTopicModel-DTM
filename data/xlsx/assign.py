# -*- coding: utf-8 -*-
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('march2.xlsx')

df.drop(df.columns[[20,21,22,23,24,25,26]], axis=1, inplace=True)
print(df.columns)
