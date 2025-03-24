import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import streamlit.components.v1 as components

# Page configuration

st.set_page_config(page_title="Nairobi PM2.5 Prediction Tool", page_icon="🪂", layout="wide")


note_book_path = 'Notebook.html'
model_path     = 'models_metadata.pkl'


with st.sidebar:
    selected = option_menu(None, ["Home","EDA", "Models", "Prediction",'Interpretation'], 
    icons     =['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon ="cast", default_index=0, orientation="vertical",
    
    styles={
    "icon"              : {"color": "orange", "font-size": "16px"}, 
    "nav-link"          : {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected" : {"background-color": "green"},
    })

if selected == "EDA":
    st.header(":orange[Diabetes Prediction] Tool Exploratory Analysis",divider=True,)
    st.markdown("---")

    renderer = get_pyg_renderer()
    renderer.render_explore()
