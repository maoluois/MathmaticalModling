import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题path1 = r'附件1.xlsx'
path1 = r'附件1.xlsx'
path2 = r'附件2.xlsx'

df_agriarea = pd.read_excel(path1, sheet_name='乡村的现有耕地')
df_agrikind = pd.read_excel(path1, sheet_name='乡村种植的农作物')
df_agrinum = pd.read_excel(path2, sheet_name='2023年的农作物种植情况') 
df_agrisell = pd.read_excel(path2, sheet_name='2023年统计的相关数据')
# sheet_names = pd.ExcelFile(path2).sheet_names
# print(sheet_names)

df_agriarea = df_agriarea.drop(columns=['说明 '])
df_agrikind = df_agrikind.drop(columns=['说明'])
df_agrikind = df_agrikind.drop(columns=['种植耕地'])
df_agrinum.ffill(inplace=True)
df_agriarea.columns = ['种植地块', '地块类型', '地块面积/亩']

df_merge1 = pd.merge(df_agrinum, df_agriarea, on='种植地块', how='left')

df_merge = pd.merge(df_merge1, df_agrisell, on=['作物编号', '地块类型'], how='left') 
df_merge.drop(columns=['种植季次_y','作物名称_y','序号'], inplace=True)
price = df_merge['销售单价/(元/斤)']
beans = df_merge['作物类型']
is_beans = []
price_min = []
price_max = []
# 找到价格区间
for i in price:
    min = float(i.split('-')[0])
    max = float(i.split('-')[1])
    price_min.append(min)
    price_max.append(max)
# 找到豆类
for i in beans:
    if len(i) == 6:
        is_beans.append(1)
    else:
        is_beans.append(0)

df_merge['销售单价_min'] = price_min
df_merge['销售单价_max'] = price_max
df_merge['是否豆类'] = is_beans
df_merge['总产量'] = df_merge['亩产量/斤'] * df_merge['地块面积/亩']
df_merge['总成本'] = df_merge['地块面积/亩'] * df_merge['种植成本/(元/亩)']
df_merge['总收入'] = df_merge['总产量'] * (df_merge['销售单价_min'] + df_merge['销售单价_max']) / 2
df_merge['总利润'] = df_merge['总收入'] - df_merge['总成本']
df_merge1 = df_merge
df_merge['亩利润'] = df_merge['总利润'] / df_merge['地块面积/亩']
df_merge['总利润率'] = df_merge['总利润'] / df_merge['总成本']
# df_merge.to_excel('merge.xlsx', index=False)
# 平旱地、梯田种植作物的种类
K_a = df_agrikind.iloc[0:15, 1]
# 水浇地第一季种植作物的种类
K_b = df_agrikind.iloc[15:34, 1]
# 水浇地第二季种植作物的种类
K_bb = df_agrikind.iloc[34:37, 1] 
# 普通大棚第一季种植作物的种类
K_c = df_agrikind.iloc[16:34, 1]
df_agrikind.iloc[29, 1] = '生菜'
# 普通大棚第二季种植作物的种类
K_cc = df_agrikind.iloc[37:41, 1]
# 智慧大棚第一季种植作物的种类
K_d = df_agrikind.iloc[16:34, 1]
# 智慧大棚第二季种植作物的种类
K_dd = df_agrikind.iloc[16:34, 1]
# 平旱地、梯田的编号
P_a = df_agriarea.iloc[0:26, 0]
# 水浇地的编号
P_b = df_agriarea.iloc[26:34, 0]
# 普通大棚的编号
P_c = df_agriarea.iloc[34:50, 0]
# 智慧大棚的编号
P_d = df_agriarea.iloc[50:54, 0]


# 平旱地、梯田种植情况
a = np.zeros(shape=(26, 15, 8))
# 水浇地第一季种植情况
b = np.zeros(shape=(8, 19, 8))
# 水浇地第二季种植情况
bb = np.zeros(shape=(8, 3, 8))
# 普通大棚第一季种植情况
c = np.zeros(shape=(16, 18, 8))
# 普通大棚第二季种植情况
cc = np.zeros(shape=(16, 4, 8))
# 智慧大棚第一季种植情况
d = np.zeros(shape=(4, 18, 8))
# 智慧大棚第二季种植情况
dd = np.zeros(shape=(4, 18, 8))



for i in range(4):
    df_range = df_merge[(df_merge['地块类型'] == '智慧大棚') & (df_merge['种植季次_x'] == '第一季')]
    for k in range(len(df_range)):
        if df_range.iloc[k, 0] == P_d.iloc[i]:
            print(f"这{df_range.iloc[k, 0]}种了")
            for j in range(18): 
                if df_range.iloc[k, 2] == K_d.iloc[j]:
                    print(f"在这{df_range.iloc[k, 0]}种了{K_d.iloc[j]}")
                    d[i, j, 0] = 1
             
        
                
print(len(df_range))
print(d[:, :, 0])
