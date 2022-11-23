import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_titanic(survived, pclass_max, pclass_min, sex_max, age_max, age_min, sibsp_max, parch_max,
                    fare_max, embarked_max):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [random.randrange(pclass_min, pclass_max+1)],
                       "sex": [random.randrange(sex_max+1)],
                       "age": [random.uniform(age_max, age_min)],
                       "sibsp": [random.randrange(sibsp_max+1)],
                       "parch": [random.randrange(parch_max+1)],
                       "fare": [(random.randrange(fare_max+1))/1.0],
                       "embarked": [random.randrange(embarked_max+1)]
                      })
    df['survived'] = survived
    # df["sex"] = df["sex"].astype('int64')
    # df["embarked"] = df["embarked"].astype('int64')
    return df


def get_random_titanic_survivors():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    died_df = generate_titanic(0, 3, 1, 1, 80, 0, 2, 6, 512, 2)
    survived_df = generate_titanic(1, 3, 1, 1, 80, 0, 2, 6, 512, 2)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        titanic_df = died_df
        print("Dead person added")
    else:
        titanic_df = survived_df
        print("Alive person added")

    return titanic_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_titanic_survivors()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    print(titanic_df.dtypes)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
