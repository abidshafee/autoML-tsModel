import streamlit as st
# import auto_ts as ts
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime

st.title("Throughput Metrics")

@st.cache(allow_output_mutation=True)
def load_dataset():
    data = pd.read_csv("https://raw.githubusercontent.com/abidshafee/autoML-tsModel/main/throughput_metrics.csv", parse_dates=['Time'], index_col='Time')
    return data

df = load_dataset()
st.subheader("Throughput Dataset For different Sites:")
st.dataframe(df)
st.subheader("Data Description")
st.write(df.describe(include='all'))
# st.bar_chart(df)
df.hist()
st.pyplot()
# st.line_chart(df)
st.subheader("Plot Site Data")
sites = st.selectbox("Select Site:",("SiteA", "SiteB", "SiteC", "SiteD","SiteE", "SiteF"))

@st.cache(allow_output_mutation=True)
def plot_sites(cols):
    plot = px.line(df, x=df.index, y=df[cols])
    return plot
fig1 = plot_sites(sites)
st.plotly_chart(fig1, use_container_width=False)

st.subheader("Data Distribution")

scaler = StandardScaler()
tdf = scaler.fit_transform(df)

attrs = [tdf[0], tdf[1], tdf[2], tdf[3], tdf[4], tdf[5]]
labels = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
colors = ['#393E46', '#2BCDC1', '#F66095', '#835AF1', '#7FA6EE', '#B8F7D4']
bins = [0.27, 0.25, 0.2, 0.15, 0.1, 0.07]

@st.cache(allow_output_mutation=True)
def data_distribution(data, label, col, bin):
    disp = ff.create_distplot(data, label, colors = col, bin_size=bin, show_curve=False)
    return disp

fig2 = data_distribution(attrs, labels, colors, bins)
st.plotly_chart(fig2, use_container_width=True)





