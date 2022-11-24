import os
import modal
import numpy as np
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def new_generate_titanic(survived, pclassDist, sexDist, ageMeanStd,sibspDist,parchDist,fareMeanStd,embarkedDist):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [np.random.choice(np.arange(1, 4), p=pclassDist)],
                       "sex": [np.random.choice(np.arange(0, 2), p=sexDist)],
                       "age": [np.random.normal(ageMeanStd[0], ageMeanStd[1])],
                       "sibsp": [np.random.choice(np.arange(0, 9), p=sibspDist)],
                       "parch": [np.random.choice(np.arange(0, 7), p=parchDist)],
                       "fare": [np.random.normal(fareMeanStd[0], fareMeanStd[1])/1.0],
                       "embarked": [np.random.choice(np.arange(0, 3), p=embarkedDist)]
                      })
    df['survived'] = survived
    # df["sex"] = df["sex"].astype('int64')
    # df["embarked"] = df["embarked"].astype('int64')
    return df

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
    pclassDist = [0.3941176470588235, 0.25588235294117645, 0.35]
    pclassDistD = [0.14571948998178508, 0.1766848816029144, 0.6775956284153005]
    sexDist = [0.6794117647058824,0.3205882352941177]
    sexDistD = [0.14754098360655737,0.8524590163934426]
    ageS = [28.41487888301388,13.682061823039358]
    ageD = [30.40211582345838,12.45814089636586]
    sibDist = [0.611764705882353,0.32941176470588235,0.03823529411764706,0.011764705882352941,0.008823529411764706,0.0, 0.0, 0.0, 0.0]
    sibDistD = [0.7249544626593807,0.1766848816029144,0.0273224043715847,0.02185792349726776,0.0273224043715847,0.009107468123861567,0.0,0.0,0.012750455373406194]
    parchDist = [0.6794117647058824,0.19117647058823528,0.11764705882352941,0.008823529411764706,0.0,0.0029411764705882353,0.0]
    parchDistD = [0.8105646630236795,0.0965391621129326,0.07285974499089254,0.0036429872495446266,0.007285974499089253,0.007285974499089253,0.0018214936247723133]
    fareS = [48.20949823529412,66.74877308878189]
    fareD = [22.117886885245902,31.388206530563984]
    embDist = [0.638235294117647,0.2735294117647059,0.08823529411764706]
    embDistD = [0.7777777777777778,0.1366120218579235,0.08561020036429873]
    died_df = new_generate_titanic(0,pclassDistD,sexDistD,ageD,sibDistD,parchDistD,fareD,embDistD)
    survived_df = new_generate_titanic(1,pclassDist,sexDist,ageS,sibDist,parchDist,fareS,embDist)

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
