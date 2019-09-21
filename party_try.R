library(tidyverse)
library(party)
list.files(path = "../input")

list.files(path = "../input/titanic/")
df_titanic_train = read_csv('../input/titanic/train.csv')
df_titanic_test = read_csv('../input/titanic/test.csv')


forest_check <- cforest(formula = 
                          factor(df_titanic_train$Survived) ~ df_titanic_train$Age + 
                          factor(df_titanic_train$Pclass), 
                        data = df_titanic_train, control = cforest_unbiased())