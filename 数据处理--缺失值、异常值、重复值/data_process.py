#1\数据缺失值处理：Numpy\Pandas\sklearn.preprocessing中的Imputer模块
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

df=pd.DataFrame(np.random.randn(6,4),columns=['col1','col2','col3','col4'])
print(df)
df.iloc[1,1]=np.nan    #增加缺失值
df.iloc[2,1]=np.nan
df.iloc[4,3]=np.nan
print(df)

#查看缺失值
nan_all=df.isnull()  #获取数据框中的N值
print(nan_all)

#查看那些列缺失
nan_col1=df.isnull().any()      #获得含有na的列
nan_col2=df.isnull().all()      #获得全部为na的列
print(nan_col1)
print(nan_col2)

#丢弃含有nan的记录
df2=df.dropna()    #丢弃含有na的记录
print(df2)

#使用sklearn将缺失值替换为特定值
nan_model=Imputer(missing_values='NaN',strategy='mean',axis=0)  #使用Imputer函数
nan_result=nan_model.fit_transform(df)    #应用规则
print(nan_result)

#用pandas将nan值转化为特定值
nan_result_pd1=df.fillna(method='backfill')  #用后面的值替换缺失值
nan_result_pd2=df.fillna(method='bfill',limit=1)  #用后面的值替换nan值，每列只能替换一个
nan_result_pd3=df.fillna(method='pad')  #用前面的值替换nan值
nan_result_pd4=df.fillna(0)    #用0插补
nan_result_pd5=df.fillna({'col2':'1.1','col4':'1.2'})   #指定列用指定数字插入
nan_result_pd6=df.fillna(df.mean()['col2':'col4'])
print(nan_result)
print(nan_result_pd1)
print(nan_result_pd2)
print(nan_result_pd3)
print(nan_result_pd4)
print(nan_result_pd5)
print(nan_result_pd6)

#简单粗暴，直接使用pandas中replace函数
df_replace=df.replace(np.nan,0)
print(df_replace)


#2、异常值处理：使用Z标准化后得到的阈值作为判断标准：标准化后的得分超过阈值则为异常
import pandas as pd

YC_df=pd.DataFrame({'col1':[1,120,3,5,2,12,13],'col2':[12,17,31,53,22,32,43]})
print(YC_df)

#通过Z标准化来识别
df_zscore=YC_df.copy()       #复制一个数据框用来存储得分
cols=YC_df.columns
for col in cols:
    df_col=YC_df[col]
    z_score=(df_col-df_col.mean())/df_col.std()  #计算每列的zscore
    df_zscore[col]=z_score.abs()>2.2             #一般情况下，阈值大于2时已经很异常了
print(df_zscore)


#3、重复值处理
import pandas as pd

data1=['a',3]
data2=['b',2]C
data3=['a',3]
data4=['c',2]
CF_df=pd.DataFrame([data1,data2,data3,data4],columns=['col1','col2'])
print(CF_df)

isDuplicated=CF_df.duplicated()      #判断重复记录数
print(isDuplicated)

new_df1=CF_df.drop_duplicates()
new_df2=CF_df.drop_duplicates(['col1'])      #删除同一列中值相同的记录
new_df3=CF_df.drop_duplicates(['col2'])      
new_df4=CF_df.drop_duplicates(['col1','col2']) #删除指定列值形同的数据
print(new_df1)
print(new_df2)
print(new_df3)
print(new_df4)

#除了使用pandas中的duplicates()外也可以使用numpy中的unique方法