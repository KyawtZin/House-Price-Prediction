# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:01:21 2023

@author: KKZ
"""

#https://www.kaggle.com/code/sharath2310/housing-price-prediction/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
cur_dir = 'D:\supervised learning'
os.chdir(cur_dir)
root = os.getcwd()


file_path = os.path.join(root,"D:/supervised learning")
file_name = 'Chennai houseing sale.csv'
file_dir = os.path.join(file_path,file_name)
raw_data = pd.read_csv(file_dir)

print(raw_data.head(5))
print(type(raw_data))
print(raw_data.shape[0])
print(raw_data.shape[1])
print("This Data Frame includes "+str(raw_data.shape[0])+" row and "+ str(raw_data.shape[1]) + " columns."
      
raw_data.nunique()

raw_data.info()

def null_finding(raw_data):
    count = raw_data.count()
    count_null = raw_data.isnull().sum()
    percentage = (count_null/count)*100
    return(percentage)

raw_data.apply(null_finding,axis=0)

# wil remove the null count since the percentage is very low

data_without_null = raw_data.dropna()
print(data_without_null.shape)
data_without_null.isnull().sum()
# null value are excluded

#removing space between Area
print(raw_data['AREA'].unique())

#lower the character
data_without_null.columns= data_without_null.columns.str.lower()

data_without_null['area'] =  data_without_null['area'].str.lower()

data_without_null['area'].replace({'velchery':'velachery', 
                       'kknagar':'kk nagar', 
                       'tnagar':'t nagar', 
                       'chormpet':'chrompet', 
                       'chrompt':'chrompet', 
                       'chrmpet':'chrompet', 
                       'ana nagar':'anna nagar', 
                       'ann nagar':'anna nagar',
                       'karapakam':'karapakkam', 
                       'adyr':'adyar'},inplace=True)


data_without_null['sale_cond'] = data_without_null['sale_cond'].str.lower()
data_without_null['sale_cond'].replace({
                                        'normal sale':'normalsale',
                                        'ab normal':'abnormal', 
                                        'partiall':'partial',
                                        'adj land':'adjland'},inplace=True)



data_without_null['buildtype'].replace({'Commercial':'commercial',
                                        'Others':'other',
                                        'Other':'other',
                                        'House':'house',
                                        'Comercial':'commercial'
                                        },inplace = True)

data_without_null['park_facil'].replace({'Noo':'No'},inplace=True)


data_without_null['park_facil'] = data_without_null['park_facil'].str.lower()

#cleaning street
data_without_null['street'].replace({'Pavd':'Paved',
                          'NoAccess':'No Access'},inplace=True)
data_without_null['street'] = data_without_null['street'].str.lower()


#cleaning mzzone
data_without_null['mzzone'] = data_without_null['mzzone'].str.lower()


data_without_null['date_sale'] = pd.to_datetime(data_without_null['date_sale'], format='%d-%m-%Y')

data_without_null['date_build'] = pd.to_datetime(data_without_null['date_build'], format='%d-%m-%Y')

data_without_null[['date_sale','date_build']]

data_without_null['house_age'] = pd.DatetimeIndex(data_without_null['date_sale']).year-pd.DatetimeIndex(data_without_null['date_build']).year

data_without_null['total_price'] = data_without_null['reg_fee'] + data_without_null['commis']  + data_without_null['sales_price']
                                    
data_cleansed = data_without_null.reindex(columns = ['total_price','reg_fee','commis','sales_price','area','sale_cond', 'park_facil',
       'buildtype', 'utility_avail', 'street', 'mzzone', 
       'date_build', 'date_sale', 'house_age', 
       'int_sqft', 'dist_mainroad', 'n_bedroom','n_bathroom', 'n_room', 
       'qs_rooms', 'qs_bathroom', 'qs_bedroom', 'qs_overall'])         

data_cleansed.head(10)    

data_num = data_cleansed[['total_price','reg_fee','commis','sales_price','int_sqft','house_age','qs_overall']]
data_num.head(5)

data_category = data_cleansed[['area','sale_cond','park_facil','buildtype','utility_avail','street','mzzone','dist_mainroad','n_bedroom','n_bathroom','n_room','qs_rooms','qs_bathroom','qs_bedroom']]                      
data_category.head(5)

data_num.hist(bins=100,figsize=(20,15))
plt.show()

data_category.hist(bins=50,figsize=(20,15))

'''Some insights that we can see are

1.Distance from the main road is realtivaly equally distributed
2. There are less that 500 properties with 4 bedrooms
3. Majority of the properties recorded have one bathroom
Looking at the total number of rooms there is a very small amount with 6 total rooms 
relative to the population sample'''

plt.figure(figsize=(18,10), dpi=150)
sns.heatmap(data_cleansed.corr(method='pearson'), cbar=False, annot=True, fmt='.1f', linewidth=0.2, cmap='coolwarm');

sns.pairplot(data=data_num)


data_cleansed.info()


plt.figure(figsize=(25,18),dpi=400)

plt.subplot(4,4,1)
sns.regplot(data=data_cleansed,x='reg_fee',y='total_price',line_kws={"color" :'red'},ci=95)
plt.subplot(4,4,2)
sns.regplot(data=data_cleansed,x='commis',y='total_price',line_kws={"color" :'red'},ci=95)
plt.subplot(4,4,3)
sns.regplot(data=data_cleansed,x='int_sqft',y='total_price',line_kws={"color" :'red'},ci=95)
plt.subplot(4,4,4)
sns.regplot(data=data_cleansed,x='house_age',y='total_price',line_kws={"color" :'red'},ci=95)
plt.subplot(4,4,5)
sns.regplot(data=data_cleansed,x='qs_overall',y='total_price',line_kws={"color" :'red'},ci=95)
plt.subplot(4,4,6)
sns.regplot(data=data_cleansed,x='dist_mainroad',y='total_price',line_kws={"color" :'red'},ci=95)

plt.show()

''' from the plots, the following facts are found.
1. Reg_fee and commis are directly related with total price and we can remove to determine the price.
2. Size and age of the house is well related with total price and it is distributed.
3. qs_overall and dist_mainroad are not related with total price.'''

fig , (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(15,5)) #sharey -> share 'Price' as y
ax1.scatter(data_cleansed['int_sqft'],data_cleansed['total_price'])
ax1.set_title('Price and Size')
ax2.scatter(data_cleansed['house_age'],data_cleansed['total_price'])
ax2.set_title('Price and Age')

log_price = np.log(data_cleansed['total_price'])

data_cleansed['log_price'] = log_price
data_cleansed.head(10)

data_log = data_cleansed.drop(['total_price'],axis = 1)
data_log.head(5)


fig , (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(15,5)) #sharey -> share 'Price' as y
ax1.scatter(data_log['int_sqft'],data_log['log_price'])
ax1.set_title('Price and Size')
ax1.set_xlabel('int_sqft')
ax1.set_ylabel('price')
ax2.scatter(data_log['house_age'],data_log['log_price'])
ax2.set_xlabel('house_age')
ax2.set_ylabel('price')
ax2.set_title('Price and Age')

data_vif = data_log.drop(['log_price'],axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor


variables = data_vif[['house_age','int_sqft',
                      'dist_mainroad','n_bedroom',
                      'n_bathroom','n_room','qs_rooms',
                      'qs_bathroom','qs_overall']]
variables.shape
vif = pd.DataFrame()
temp = []

variables.shape[1]

for i in range(variables.shape[1]):
    temp.append(variance_inflation_factor(variables.values, i))

vif['VIF'] = temp

print(vif)

vif['Features'] = variables.columns

print(vif)

data_no_multi = data_log.drop(['n_room','qs_rooms','qs_bathroom','qs_overall'],axis = 1)

data_no_multi.shape
data_no_multi.drop(['date_build','date_sale',],axis=1,inplace=True)



data_with_dummies = pd.get_dummies(data_no_multi, drop_first=True)

data_with_dummies.head(2)
data_with_dummies.columns

data_preprocessed = data_with_dummies.copy()

target_prices = data_preprocessed['log_price']

inputs = data_preprocessed.drop(['log_price'],axis=1)

from sklearn.preprocessing import  StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,target_prices,test_size=0.2, random_state=365)

reg = LinearRegression()

reg.fit(x_train,y_train)

y_hat = reg.predict(x_train)
y_hat

plt.figure(figsize=(25,18),dpi=400)
sns.regplot(x= y_train,y= y_hat,line_kws={"color" :'red'},ci=95)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=10)
plt.ylabel('Predictions (y_hat)',size=10)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(14,17)
plt.ylim(14,17)
plt.show()

plt.figure(figsize=(25,18),dpi=400)
sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)
plt.show()

reg.score(x_train,y_train)


reg.intercept_

reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

y_hat_test = reg.predict(x_test)
y_hat_test

plt.figure(figsize=(25,18),dpi=400)
sns.scatterplot(x = y_test, y = y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(14,17)
plt.ylim(14,17)
plt.show()

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()
y_test = y_test.reset_index(drop=True)

df_pf['Target'] = np.exp(y_test)
df_pf

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
df_pf.sort_values(by=['Difference%'])
