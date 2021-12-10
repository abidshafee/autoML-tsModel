import streamlit as st
# import auto_ts as ts
import pandas as pd
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
#fig = ff.Figure([ff.Scatter(x=df['Time'], y=df['SiteA', 'SiteB'])])
st.plotly_chart(fig1, use_container_width=False)

attrs = [df.columns[0], df.columns[1], df.columns[2]]
labels = ['SiteA', 'SiteB', 'SiteC']
fig2 = ff.create_distplot(
         attrs, labels)
st.plotly_chart(fig2, use_container_width=True)





