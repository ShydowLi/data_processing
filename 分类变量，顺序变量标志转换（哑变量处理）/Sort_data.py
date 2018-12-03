#将分类数据和顺序数据转换为标志变量
#1、分类数据的处理
#2、顺序数据的处理

#运用标志方法处理分类和顺序数据

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df=pd.DataFrame({
    'id':[3566841,6541227,3512441],
    'sex':['male','female','female'],
    'level':['high','low','middle']
})
print(df)

#自定义转换过程
df_new=df.copy()
for col_num,col_name in enumerate(df):
    col_data=df[col_name]
    col_type=col_data.dtype
    if col_type=='object':
        df_new=df_new.drop(col_name,1)
        value_sets=col_data.unique()
        for value_unique in value_sets:
            col_name_new=col_name+'-'+value_unique
            col_tmp=df.iloc[:,col_num]
            new_col=(col_tmp==value_unique)
            df_new[col_name_new]=new_col
print(df_new)

#使用sklearn进行标志转换(哑变量编码)

df2=pd.DataFrame({
    'id':[3566841,6541227,3512441],
    'sex':[1,2,2],
    'level':[3,1,2]
})
id_data=df2.values[:,:1]
print(id_data)
transform_data=df2.values[:,1:]
enc=OneHotEncoder()
df2_new=enc._fit_transform(transform_data).toarray()
df2_all=pd.concat((pd.DataFrame(id_data),pd.DataFrame(df2_new)),axis=1)
print(df2_all)