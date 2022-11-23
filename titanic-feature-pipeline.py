import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd
from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate
import numpy as np

if __name__ == '__main__':

    project = hopsworks.login()
    fs = project.get_feature_store()

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


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"],
    description="titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})


#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")