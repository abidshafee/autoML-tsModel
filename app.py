# from os import name
import streamlit as st
# import auto_ts as ts
import pandas as pd
import scipy.stats as sst
import numpy as np
# import statsmodels
from sklearn.preprocessing import StandardScaler
# import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

st.title("Throughput Metrics Forcasting")

# Read the wine-quality csv file from the URL
csv_url = (
        "https://raw.githubusercontent.com/abidshafee/autoML-tsModel/main/throughput_metrics.csv"
    )

@st.cache(allow_output_mutation=True)
def load_dataset(csv):
    try:
        data = pd.read_csv(csv, parse_dates=['Time'], index_col='Time')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
    return data

#@st.cache(allow_output_mutation=True)
#def load_dataset():
    # data = pd.read_csv("https://raw.githubusercontent.com/abidshafee/autoML-tsModel/main/throughput_metrics.csv", parse_dates=['Time'], index_col='Time')
    ## data = data.dropna(inplace=True)
    # return data

df = load_dataset(csv_url)
st.subheader("Throughput Dataset For different Sites:")

# st.bar_chart(df)
# df.hist()
# st.pyplot()
# st.line_chart(df)

# Removing Outliers
@st.cache(allow_output_mutation=True)
def remove_outliers(data):
    z_score = sst.zscore(data)
    abs_z_scores = np.abs(z_score)
    data_filter = (abs_z_scores<3).all(axis=1)
    df = data[data_filter]
    return df

df = remove_outliers(df)
st.dataframe(df)
st.subheader("Data Description")
st.write(df.describe(include='all'))

fig = make_subplots(rows=6, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03)

# for i, col in enumerate(df.columns[0:], 1):
# for i, col in enumerate(df.columns[0:]):
    # fig.add_trace(go.Scatter(x=df.index, y=df[col], name=df.columns[i]), row=6, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['SiteA'], name="SiteA"), row=6, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteB'], name="SiteB"), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteC'], name="SiteC"), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteD'], name="SiteD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteE'], name="SiteE"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteF'], name="SiteF"), row=1, col=1)

#fig = stack_plot()
fig.update_layout(height=680, width=800, xaxis6_rangeslider_visible=True,
                  title_text="Stacked Sites Data", showlegend=True,
                  xaxis6_rangeslider_thickness=0.05)
st.plotly_chart(fig, use_container_width=True)

# Plotting all features together
# df[['SiteF','SiteE','SiteD','SiteC','SiteB','SiteA']].plot(subplots=True)
# st.pyplot()

st.subheader("Plot Site Data")
sites = st.selectbox("Select Site:",("SiteA", "SiteB", "SiteC", "SiteD","SiteE", "SiteF"))

@st.cache(allow_output_mutation=True)
def plot_sites(cols):
    # plot = px.line(df, x=df.index, y=df[cols])
    plot = go.Figure([go.Scatter(x=df.index, y=df[cols])])
    return plot
fig1 = plot_sites(sites)
fig1.update_layout(title_text='Individual Site Data')
# fig1.update_xaxes(rangeslider_visible=True)
fig1.update_xaxes(
    rangeslider_visible=True,
    rangeslider_thickness=0.05,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(fig1, use_container_width=True)


st.subheader("Data Distribution")

scaler = StandardScaler()
tdf = scaler.fit_transform(df)

attrs = [tdf[0], tdf[1], tdf[2], tdf[3], tdf[4], tdf[5]]
labels = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
colors = ['#393E46', '#2BCDC1', '#F66095', '#835AF1', '#7FA6EE', '#B8F7D4']
bins = [0.27, 0.25, 0.2, 0.15, 0.1, 0.07]

@st.cache(allow_output_mutation=True)
def data_distribution(data, label, col, bin):
    disp = ff.create_distplot(data, label, colors = col, bin_size=bin, show_curve=True)
    return disp

fig2 = data_distribution(attrs, labels, colors, bins)
fig2.update_layout(title_text='Throughput Data Distribution')
st.plotly_chart(fig2, use_container_width=True)