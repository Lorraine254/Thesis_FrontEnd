import streamlit as st
import pandas as pd
from joblib import load 
import os
import numpy as np

from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from sklearn.model_selection import train_test_split
from lime import lime_tabular

# loading the data and caching
@st.cache_data
def read_data():
    return pd.read_csv('https://raw.githubusercontent.com/NUELBUNDI/Machine-Learning-Data-Set/main/diabetes.csv')

# loading the model pickle file
@st.cache_resource
def model_load():
    loaded_models, loaded_model_results = load('lightgmb_model.pkl')
    return loaded_models, loaded_model_results

