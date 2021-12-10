from os import name
import streamlit as st
# import auto_ts as ts
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
# df.hist()
# st.pyplot()
# st.line_chart(df)

fig = make_subplots(rows=6, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03)

fig.add_trace(go.Scatter(x=df.index, y=df['SiteA'], name="SiteA"),
              row=6, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteB'], name="SiteB"),
              row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteC'], name="SiteC"),
              row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteD'], name="SiteD"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteE'], name="SiteE"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SiteF'], name="SiteF"),
              row=1, col=1)
fig.update_layout(height=600, width=800,
                  title_text="Stacked All Sites Data", showlegend=True)
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