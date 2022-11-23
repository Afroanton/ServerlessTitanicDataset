import gradio as gr
import numpy as np
from PIL import Image
import requests
import io
import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def sexToInt(sex):
    if sex.lower() == "male":
        sex = int(0)
    elif sex.lower() == "female":
        sex = int(1)

def embarkedToInt(input):
    if input.lower() == "s":
        input = int(0)
    elif input.lower() == "c":
        input = int(1)
    elif input.lower() == "q":
        input = int(3)

def titanic(pclass, sex, age, sibsp, parch, fare, embarked):

    # sex = sexToInt(sex)
    # embarked = embarkedToInt(embarked)
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    input_list.append(embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))

    titanic_url = "https://raw.githubusercontent.com/DavidKrugerT/images/main/" + str(res[0]) + ".png"
    img = Image.open(requests.get(titanic_url, stream=True).raw)
    return img
    # if res[0] == 1:
    #     return "Survived"
    # return "Died"

demo = gr.Interface(
    fn=titanic,
    title="Titanic Predictive Analytics",
    description="Experiment with Predictive Survival",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="pclass (1, 2, 3)"),
        gr.inputs.Number(default=1.0, label="sex (male = 0), (female = 1)"),
        gr.inputs.Number(default=25.0, label="age (Number)"),
        gr.inputs.Number(default=1.0, label="sibsp (int)"),
        gr.inputs.Number(default=0.0, label="parch (0, 1, 2)"),
        gr.inputs.Number(default=15.0, label="fare (Price)"),
        gr.inputs.Number(default=1.0, label="embarked (S = 0, C = 1, Q = 2)"),
    ],
    outputs=gr.Image(type="pil"))

demo.launch()

