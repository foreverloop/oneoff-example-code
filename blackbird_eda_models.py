#!/usr/bin/env python
# coding: utf-8

# In[2]:


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

df_birds = pd.read_csv("blackbird.csv")

print("setup and imports complete")


# In[3]:


print("data frame shape: ",np.shape(df_birds))
df_birds.head()


# In[14]:


df_freq_count = df_birds.groupby(['Ring number']).size().reset_index(name='count')
df_freq_count[df_freq_count['count'] > 1].sort_values(by=['count'], ascending=False).head(10)


# In[3]:


#method to list out missing values, and a percentage of the dataframe as a whole which is missing
def missingDataCount(df):
    missing_val_count = df.isnull().sum()
    missing_val_count.sort_values(ascending=False,inplace=True)
    total_cells = np.product(df.shape)
    total_missing = missing_val_count.sum()
    print (missing_val_count[0:10])
    print ("percent missing overall (out of 100) = ",(total_missing/total_cells) * 100)

missingDataCount(df_birds)


# In[4]:


def categoryCheck(df,label):
    value_count = df[label].value_counts()
    df_values_counts = value_count.rename_axis('unique_values').reset_index(name='counts')
    categ_sums = df_values_counts.counts.groupby(df_values_counts.unique_values).sum()
    plt.pie(categ_sums,labels=categ_sums.index, autopct='%1.0f%%')
    plt.title("showing category breakdown for {0}".format(label))
    plt.show()

categoryCheck(df_birds,"Sex")
categoryCheck(df_birds,"Age")
categoryCheck(df_birds,"Scheme")


# In[5]:


#imputed the missing values with the mean...maybe too much of an assumption
df_birds['Weight'].fillna(df_birds['Weight'].mean(), inplace=True)
df_birds['Wing'].fillna(df_birds['Wing'].mean(),inplace=True)

missingDataCount(df_birds)


# In[6]:


all_blackbird_features = ['Scheme','Ring number','Age','Sex','Wing','Weight','Day','Month','Year','Time']
blackbird_numeric_features = ['Wing','Weight']

for feature in blackbird_numeric_features:
    x = df_birds[feature]
    plt.figure(1); plt.title('Normal')
    sns.distplot(x, kde=True, fit=stats.norm)
    plt.show()


# In[7]:


ax = sns.scatterplot(x="Weight", y="Wing", hue="Age",data=df_birds)
plt.show()
ax = sns.scatterplot(x="Sex", y="Wing", hue="Age",data=df_birds)
plt.show()
ax = sns.scatterplot(x="Age", y="Wing", hue="Sex",data=df_birds)


# In[8]:


#visually we can see a lot of variance for the weight var so...measures of spread
df_birds['Weight'].describe()


# In[9]:


#change the categorical letters to discrete nominal and ordinal values respectively
#this is so they will work properly with models and such later on
le = LabelEncoder()
le.fit(df_birds['Sex'])
df_birds['Sex'] = le.transform(df_birds['Sex'])
print("Sex labels: ",le.classes_)
le = LabelEncoder()
le.fit(df_birds['Age'])
df_birds['Age'] = le.transform(df_birds['Age'])
print("Age labels: ",le.classes_)


# In[10]:


def build_correlate_graph(df_to_corr, features,correlate_for,corr_type):
    mydf = pd.DataFrame()
    mydf['feature'] = features
    mydf[corr_type] = [df_to_corr[f].corr(df_to_corr[correlate_for], corr_type) for f in features]
    mydf = mydf.sort_values(corr_type)
    print(mydf)
    plt.figure(figsize=(6, 0.25*len(features)))
    ax = sns.barplot(data=mydf, y='feature', x=corr_type, orient='h')
    ax.set_title("blackbird {0} correlations".format(corr_type))
    plt.ylabel('')
    plt.show()


# In[11]:


#do spearman, pearsons and kendall correlations
#No correlations for scheme because there is one scheme which account for {0} of the schemes involved"6/len(df_birds)
#No for Ring number, since it is just an identifier for the bird and not actually useful for prediction
blackbird_corr_features = ['Wing','Weight','Sex','Time','Age','Day','Month','Year']

build_correlate_graph(df_birds,blackbird_corr_features,'Wing','spearman')
build_correlate_graph(df_birds,blackbird_corr_features,'Wing','pearson')
build_correlate_graph(df_birds,blackbird_corr_features,'Wing','kendall')

#The most correlated features, in order: Weight, Age and Sex
#descision not to scale values since Sex cannot be scaled and only Weight and Age could be
#which is inconsistent overall


# In[12]:


#here is the same code as below, but run only for the weight feature
#this has been done to allow for a comparison of having multiple features versus a single feature
#and weatehr or not the loss in accuracy is worth the extra transparency which this method provides
features = ['Weight']

X_train, X_valid, y_train, y_valid = train_test_split(df_birds[features], 
                                                      df_birds['Wing'], 
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

linear_reg_model = LinearRegression().fit(X_train, y_train)
lrg_predictions = linear_reg_model.predict(X_valid)

linear_reg_mse = int(mean_squared_error(y_valid, lrg_predictions))
linear_reg_r2 = r2_score(y_valid, lrg_predictions)

print("feature coefficients: ",linear_reg_model.coef_)
print("Regression y intercept: ",linear_reg_model.intercept_)
print("linear regression mean squared error: {0}, linear regression r squared: {1}"      .format(linear_reg_mse,linear_reg_r2))

plt.scatter(X_valid,y_valid)
plt.plot(X_valid,lrg_predictions,c='orangered')
plt.show()

print(np.shape(X_valid),np.shape(y_valid))
ax = sns.regplot(x=X_valid, y=y_valid,scatter_kws={"color": "royalblue"}, line_kws={"color": "orangered"})
ax.set(xlabel='Weight', ylabel='Wing')
plt.show()
#the regression line IS accurate, but the x values only start at ~80 as they are weights


# In[13]:


#only features with correlations of +-0.1 are included, the other features are noise
features = ['Weight','Sex','Age','Month']

X_train, X_valid, y_train, y_valid = train_test_split(df_birds[features], 
                                                      df_birds['Wing'], 
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

linear_reg_model = LinearRegression().fit(X_train, y_train)
lrg_predictions = linear_reg_model.predict(X_valid)

linear_reg_mse = int(mean_squared_error(y_valid, lrg_predictions))
linear_reg_r2 = r2_score(y_valid, lrg_predictions)

print("multiple feature coefficients: ",linear_reg_model.coef_)
print("multiple Regression y intercept: ",linear_reg_model.intercept_)
print("multiple linear regression mean squared error: {0}, linear regression r squared: {1}"      .format(linear_reg_mse,linear_reg_r2))

#the feature coefficients would usually explain the expected change in Y for a change in value of 1 for X
#these are not so good for explaining the dependent variable, I think mostly because the variables are not scaled
#and cannot be scaled

#because I have used multiple variables in the prediction, it is now not possible to draw a single regression line
#this is because there are several feature coefficients and it would not be good to choose just one of them


# In[14]:


random_forest_reg = RandomForestRegressor(n_estimators=350, random_state=0)
random_forest_reg.fit(X_train,y_train)
random_forest_reg_pred = random_forest_reg.predict(X_valid)

random_forest_mse = int(mean_squared_error(y_valid, random_forest_reg_pred))
random_forest_r2 = r2_score(y_valid, random_forest_reg_pred)
print("Feature importances:",random_forest_reg.feature_importances_)
print("Random Forest regression mean squared error: {0}, Random forest regression r squared: {1}".format(random_forest_mse,random_forest_r2))


# In[ ]:




