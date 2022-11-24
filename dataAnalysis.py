#%%
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
titanic_df = pd.read_csv(url)

# drop columns
titanic_df.drop('Cabin', inplace=True, axis=1)
titanic_df.drop('Name', inplace=True, axis=1)
titanic_df.drop('Ticket', inplace=True, axis=1)
titanic_df.drop('PassengerId', inplace=True, axis=1)

# Drop nans
titanic_df = titanic_df[pd.notnull(titanic_df['Embarked'])]

# Fix Age, nans to average mean value
mean = titanic_df["Age"].mean()
titanic_df["Age"] = titanic_df["Age"].fillna(mean)

# Fix gender to 0,1
titanic_df.loc[titanic_df["Sex"] == 'male', 'Sex'] = 0
titanic_df.loc[titanic_df["Sex"] == 'female', 'Sex'] = 1

# Fix Embarked 0,1,2
titanic_df.loc[titanic_df["Embarked"] == 'S', 'Embarked'] = 0
titanic_df.loc[titanic_df["Embarked"] == 'C', 'Embarked'] = 1
titanic_df.loc[titanic_df["Embarked"] == 'Q', 'Embarked'] = 2

# Fix col as type int
titanic_df["Embarked"] = titanic_df["Embarked"].astype('int64')
titanic_df["Sex"] = titanic_df["Sex"].astype('int64')
#%%
survived_df = titanic_df[titanic_df['Survived'] == 1]
died_df = titanic_df[titanic_df['Survived'] == 0]
total_survived = len(survived_df)
total_died = len(died_df)
#%% Male Female survival/death rate
female_survived = len(died_df[died_df["Sex"] == 1])
male_survived = total_died-female_survived
print(female_survived / total_died)
print(male_survived / total_died)
#%% Pclass survival/death rate
class1_survived = len(died_df[died_df["Pclass"] == 1])
class2_survived = len(died_df[died_df["Pclass"] == 2])
class3_survived = len(died_df[died_df["Pclass"] == 3])
print(class1_survived / total_died)
print(class2_survived / total_died)
print(class3_survived / total_died)
#%% age mean and std
print(survived_df['Age'].mean())
print(survived_df['Age'].std())
print(died_df['Age'].mean())
print(died_df['Age'].std())
#%% SibSp class distrobution survived
for i in range(9):
    print(len(died_df[died_df["SibSp"] == i])/total_died)
#%% parch class distrobution
for i in range(7):
    print(len(died_df[died_df["Parch"] == i])/total_died)
#%% fare mean and std
print(survived_df['Fare'].mean())
print(survived_df['Fare'].std())
print(died_df['Fare'].mean())
print(died_df['Fare'].std())
#%% Embarked class distrobution
for i in range(3):
    print(len(died_df[died_df["Embarked"] == i])/total_died)
#%%
np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
np.random.normal(1, 0.5)