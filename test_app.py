import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Test", layout="wide")

st.title("Test App")
st.write("This is a test to see if Streamlit works")

st.metric("Test Metric", "123.45")
st.line_chart([1, 2, 3, 4, 5])

st.write("If you can see this, Streamlit is working!")
