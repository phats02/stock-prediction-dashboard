import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import xgboost
import pickle
import yfinance as yf
import ta
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from dash.exceptions import PreventUpdate


app = dash.Dash()
server = app.server


def rename_column(df):

    df.rename(
        columns={
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "Adj Close",
            "Volume": "volume",
            "SMA": "SMA",
            "ROC": "ROC",
            "RSI": "RSI",
            "BBANDS_upper": "Real Upper Band",
            "BBANDS_lower": "Real Lower Band",
        },
        inplace=True,
    )

    return df


def update_data(companyName):

    enddate = datetime.now()
    startdate = enddate - timedelta(days=60)
    df = yf.download(companyName, start=startdate, end=enddate, interval="15m")
    df.reset_index(inplace=True)
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=14)

    # Calculate ROC after dropping NaN values
    df["ROC"] = ta.momentum.roc(df["Close"], window=12)

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BBANDS_upper"] = bollinger.bollinger_hband()
    df["BBANDS_lower"] = bollinger.bollinger_lband()
    df["Real Middle Band"] = df["SMA"]

    df.dropna(inplace=True)

    df = rename_column(df)
    df.to_csv("../DATA/" + companyName + ".csv")


def replace_bbands(the_list):
    for item in the_list:
        if item == "KBANDS":
            yield "Real Lower Band"
            yield "Real Middle Band"
            yield "Real Upper Band"
        else:
            yield item


def lstm_predict_future(data, model, indicatorArr, period):
    # data
    data = data[indicatorArr].values
    data = data[-60:]

    # scaled data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data)

    # model input
    modelInput = scaledData.reshape(-1, scaledData.shape[0], scaledData.shape[1])

    # predicted scaled value
    predictedScaledValue = model.predict(modelInput)

    # predicted value
    predictedValue = scaler.inverse_transform(
        np.tile(predictedScaledValue, (1, scaledData.shape[1]))
    )[:, 0]

    return predictedValue


def xgboost_predict_future(data, model, indicatorArr, period):
    # indicator
    indicatorArr.insert(1, "volume")

    # data
    data = data[indicatorArr]
    data = data[-2:]

    # model input
    X = pd.DataFrame({})
    n = len(data)
    for i in range(1, n + 1):
        for column in data.columns:
            X[column + "_date_" + str(i)] = [data.iloc[n - i][column]]

    # predicted value
    predictedValue = model.predict(X)

    return predictedValue


app.layout = html.Div(
    [
        html.Div(
            "Stock Price Analysis",
            style={
                "padding": "2vh 0vw",
                "margin-bottom": "5px",
                "textAlign": "center",
                "background-color": "#1E0601",
                "color": "white",
                "font-size": "42px",
                "font-weight": "bold",
            },
        ),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Stock Data",
                    className="hoverable-tab hoverable-title",
                    selected_className="selected-tab",
                    style={"font-size": "24px", "color": "#1E90FF"},
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="company-dropdown",
                                            options=[
                                                {"label": "Microsoft", "value": "MSFT"},
                                                {"label": "Apple", "value": "AAPL"},
                                                {"label": "Facebook", "value": "META"},
                                                {"label": "Tesla", "value": "TSLA"},
                                                {"label": "Google", "value": "GOOGL"},
                                            ],
                                            multi=False,
                                            placeholder="Choose company",
                                            value="MSFT",
                                            style={
                                                "width": "60%",
                                                "margin": "0 auto 20px auto",
                                            },
                                        ),
                                        html.Button(
                                            "Update",
                                            id="update_button",
                                            style={
                                                "background-color": "#5DADE2",
                                                "border": "none",
                                                "color": "white",
                                                "padding": "15px 32px",
                                                "text-align": "center",
                                                "text-decoration": "none",
                                                "display": "inline-block",
                                                "font-size": "20px",
                                                "font-weight": "bold",
                                                "margin-left": "auto",
                                                "margin-top": "10px",
                                                "margin-bottom": "10px",
                                                "margin-right": "auto",
                                                "width": "20%",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                    style={"text-align": "center", "font-size": "24px"},
                                ),
                                html.Div(id="something", children=""),
                                html.H1("Stock Price", style={"textAlign": "center"}),
                                dcc.Dropdown(
                                    id="my-dropdown",
                                    options=[
                                        {"label": "Microsoft", "value": "MSFT"},
                                        {"label": "Apple", "value": "AAPL"},
                                        {"label": "Facebook", "value": "META"},
                                        {"label": "Tesla", "value": "TSLA"},
                                        {"label": "Google", "value": "GOOGL"},
                                    ],
                                    multi=True,
                                    value=["MSFT"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="stockprice"),
                                html.H1(
                                    "Stock Market Volume",
                                    style={
                                        "textAlign": "center",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="my-dropdown2",
                                    options=[
                                        {"label": "Microsoft", "value": "MSFT"},
                                        {"label": "Apple", "value": "AAPL"},
                                        {"label": "Facebook", "value": "META"},
                                        {"label": "Tesla", "value": "TSLA"},
                                        {"label": "Google", "value": "GOOGL"},
                                    ],
                                    multi=True,
                                    value=["MSFT"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="volume"),
                            ],
                            className="container",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Stock Prediction",
                    style={"font-size": "24px", "color": "#EA0707"},
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dropdown-company",
                                    options=[
                                        {"label": "Microsoft", "value": "MSFT"},
                                        {"label": "Apple", "value": "AAPL"},
                                        {"label": "Facebook", "value": "META"},
                                        {"label": "Tesla", "value": "TSLA"},
                                        {"label": "Google", "value": "GOOGL"},
                                    ],
                                    multi=False,
                                    placeholder="Choose company",
                                    value="MSFT",
                                    style={
                                        "margin-left": "auto",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px",
                                        "margin-right": "auto",
                                        "width": "80%",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="dropdown-model",
                                    options=[
                                        {
                                            "label": "Extreme Gradient Boosting (XGBOOST)",
                                            "value": "XGBOOST",
                                        },
                                        {
                                            "label": "Recurrent Neural Network (RNN)",
                                            "value": "RNN",
                                        },
                                        {
                                            "label": "Long Short Term Memory (LSTM)",
                                            "value": "LSTM",
                                        },
                                    ],
                                    multi=False,
                                    placeholder="Choose model",
                                    value="LSTM",
                                    style={
                                        "margin-left": "auto",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px",
                                        "margin-right": "auto",
                                        "width": "80%",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="dropdown-period",
                                    options=[{"label": "15 minutes", "value": 15}],
                                    multi=False,
                                    placeholder="Choose time period",
                                    value=15,
                                    style={
                                        "margin-left": "auto",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px",
                                        "margin-right": "auto",
                                        "width": "80%",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="dropdown-indicator",
                                    options=[
                                        {"label": "Close Price", "value": "close"},
                                        {
                                            "label": "Price Rate of Change (ROC)",
                                            "value": "ROC",
                                        },
                                        {
                                            "label": "Relative Strength Index (RSI)",
                                            "value": "RSI",
                                        },
                                        {
                                            "label": "Simple Moving Averages (SMA)",
                                            "value": "SMA",
                                        },
                                        {"label": "Bolling Bands", "value": "KBANDS"},
                                    ],
                                    multi=True,
                                    placeholder="Choose indicators",
                                    value=["close"],
                                    style={
                                        "margin-left": "auto",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px",
                                        "margin-right": "auto",
                                        "width": "80%",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "Predict",
                                            id="predict_button",
                                            style={
                                                "background-color": "#5DADE2",
                                                "border": "none",
                                                "color": "white",
                                                "padding": "15px 32px",
                                                "text-align": "center",
                                                "text-decoration": "none",
                                                "display": "inline-block",
                                                "font-size": "20px",
                                                "font-weight": "bold",
                                                "margin-left": "auto",
                                                "margin-top": "10px",
                                                "margin-bottom": "10px",
                                                "margin-right": "auto",
                                                "width": "20%",
                                                "cursor": "pointer",
                                            },
                                        )
                                    ],
                                    style={"text-align": "center"},
                                ),
                                dcc.Graph(id="predicted_graph"),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ],
    style={
        "font-family": "Arial, sans-serif",
        "margin": "0",
        "padding": "0",
        "box-sizing": "border-box",
        "width": "101%",
        "font-size": "16px",
    },
)

app.css.append_css(
    {
        "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css",
        "internal_stylesheets": [
            {
                "css": """
                #update_button:hover, #update_button2:hover, #predict_button:hover {
                    background-color: #1ABC9C !important;
                }
                .hoverable-title:hover {
                    background-color: #909A9B !important;
                }
                """
            }
        ],
    }
)

# df = pd.read_csv("../DATA/AAPL.csv")


@app.callback(Output("stockprice", "figure"), [Input("my-dropdown", "value")])
def update_graph(selected_dropdown):
    if not selected_dropdown:
        return go.Figure()

    dropdown = {
        "MSFT": "Microsoft",
        "AAPL": "Apple",
        "META": "Facebook",
        "TSLA": "Tesla",
        "GOOGL": "Google",
    }
    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    trace5 = []
    trace6 = []
    trace7 = []
    trace8 = []
    trace9 = []
    for stock in selected_dropdown:
        df = pd.read_csv(f"../DATA/{stock}.csv")
        # print(stock)
        trace1.append(
            go.Scatter(
                x=df["date"],
                y=df["open"],
                mode="lines",
                opacity=0.8,
                name=f"Open {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>"
                "Open: %{y:.3f}<br>"
                "High: %{customdata[0]:.3f}<br>"
                "Low: %{customdata[1]:.3f}<br>"
                "Close: %{customdata[2]:.3f}<extra></extra>",
                customdata=np.stack((df["high"], df["low"], df["close"]), axis=-1),
                hoverlabel=dict(bgcolor="#5AEA07", font_size=16, font_family="Arial"),
            )
        )
        trace2.append(
            go.Scatter(
                x=df["date"],
                y=df["high"],
                mode="lines",
                opacity=0.7,
                name=f"High {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>"
                "Open: %{customdata[0]:.3f}<br>"
                "High: %{y:.3f}<br>"
                "Low: %{customdata[1]:.3f}<br>"
                "Close: %{customdata[2]:.3f}<extra></extra>",
                customdata=np.stack((df["open"], df["low"], df["close"]), axis=-1),
                hoverlabel=dict(bgcolor="#5AEA07", font_size=16, font_family="Arial"),
            )
        )
        trace3.append(
            go.Scatter(
                x=df["date"],
                y=df["low"],
                mode="lines",
                opacity=0.6,
                name=f"Low {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>"
                "Open: %{customdata[0]:.3f}<br>"
                "High: %{customdata[1]:.3f}<br>"
                "Low: %{y:.3f}<br>"
                "Close: %{customdata[2]:.3f}<extra></extra>",
                customdata=np.stack((df["open"], df["high"], df["close"]), axis=-1),
                hoverlabel=dict(bgcolor="#5AEA07", font_size=16, font_family="Arial"),
            )
        )
        trace4.append(
            go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                opacity=0.5,
                name=f"Close {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>"
                "Open: %{customdata[0]:.3f}<br>"
                "High: %{customdata[1]:.3f}<br>"
                "Low: %{customdata[2]:.3f}<br>"
                "Close: %{y}<extra></extra>",
                customdata=np.stack((df["open"], df["high"], df["low"]), axis=-1),
                hoverlabel=dict(bgcolor="#5AEA07", font_size=16, font_family="Arial"),
            )
        )
        trace5.append(
            go.Scatter(
                x=df["date"],
                y=df["SMA"],
                mode="lines",
                opacity=0.8,
                name=f"SMA {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>" "SMA: %{y:.3f}<br>",
                visible="legendonly",
            )
        )
        trace6.append(
            go.Scatter(
                x=df["date"],
                y=df["ROC"],
                mode="lines",
                opacity=0.8,
                name=f"ROC {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>" "ROC: %{y:.3f}<br>",
                visible="legendonly",
            )
        )
        trace7.append(
            go.Scatter(
                x=df["date"],
                y=df["RSI"],
                mode="lines",
                opacity=0.8,
                name=f"RSI {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>" "RSI: %{y:.3f}<br>",
                visible="legendonly",
            )
        )
        trace8.append(
            go.Scatter(
                x=df["date"],
                y=df["Real Lower Band"],
                mode="lines",
                opacity=0.8,
                name=f"Lower Band {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>" "Lower Band: %{y:.3f}<br>",
                visible="legendonly",
            )
        )
        trace9.append(
            go.Scatter(
                x=df["date"],
                y=df["Real Upper Band"],
                mode="lines",
                opacity=0.8,
                name=f"Upper Band: {dropdown[stock]}",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>"
                "Upper Band: %{y:.3f}<br>",
                visible="legendonly",
            )
        )
    traces = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"Stock Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            titlefont={"size": 24, "color": "Blue", "family": "Arial, sans-serif"},
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {
                                "count": 1,
                                "label": "1M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {
                                "count": 6,
                                "label": "6M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
                "titlefont": {"size": 18, "color": "black"},
            },
            yaxis={
                "title": "Price (USD)",
                "titlefont": {"size": 18, "color": "Blue"},
            },
        ),
    }
    return figure


@app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
def update_graph(selected_dropdown_value):

    dropdown = {
        "MSFT": "Microsoft",
        "AAPL": "Apple",
        "META": "Facebook",
        "TSLA": "Tesla",
        "GOOGL": "Google",
    }
    trace1 = []
    for stock in selected_dropdown_value:
        df = pd.read_csv(f"../DATA/{stock}.csv")
        trace1.append(
            go.Scatter(
                x=df["date"],
                y=df["volume"],
                mode="lines",
                opacity=0.7,
                name=f"",
                textposition="bottom center",
                hovertemplate=f"<b>{dropdown[stock]}</b><br>" "Volume: %{y:.3f}<br>",
                hoverlabel=dict(bgcolor="Purple", font_size=16, font_family="Arial"),
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            titlefont={"size": 24, "color": "Purple", "family": "Arial, sans-serif"},
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {
                                "count": 1,
                                "label": "1M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {
                                "count": 6,
                                "label": "6M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
                "titlefont": {"size": 18, "color": "black"},
            },
            yaxis={
                "title": "Transactions Volume",
                "titlefont": {"size": 18, "color": "purple"},
            },
            hovermode="closest",
        ),
    }
    return figure


@app.callback(
    Output("predicted_graph", "figure"),
    [Input("predict_button", "n_clicks")],
    [
        State("dropdown-company", "value"),
        State("dropdown-model", "value"),
        State("dropdown-indicator", "value"),
        State("dropdown-period", "value"),
    ],
)
def update_graph(n_clicks, companyName, modelName, indicatorArr, period):
    data = pd.read_csv("../DATA/" + companyName + ".csv")

    # model
    modelFileName = "../MODEL/" + modelName

    indicatorArr.sort(key=str.lower)

    for indicator in indicatorArr:
        if indicator == "close":
            continue
        if indicator == "KBANDS":
            indicator = "BBANDS"
        modelFileName = modelFileName + "_" + indicator

    indicatorArr = list(replace_bbands(indicatorArr))

    print(indicatorArr)

    predictions = None
    if modelName == "LSTM" or modelName == "RNN":
        modelFileName = modelFileName + ".h5"
        model = load_model(modelFileName)
        futurePredictions = lstm_predict_future(data, model, indicatorArr, period)
        #
        dataset = data
        dataset = dataset[indicatorArr].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        X = []
        for i in range(60, len(dataset)):
            X.append(dataset[i - 60 : i][:])
        X = np.array(X[-100:])
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(
            np.tile(predictions, (1, dataset.shape[1]))
        )[:, 0]
        df = data.iloc[-len(predictions) :]
        df.loc[:, "predictions"] = predictions

        #
    elif modelName == "XGBOOST":
        modelFileName = modelFileName + ".dat"
        model = pickle.load(open(modelFileName, "rb"))
        futurePredictions = xgboost_predict_future(data, model, indicatorArr, period)
        #
        dataset = data
        temp = indicatorArr.copy()
        dataset = dataset[temp]
        for i in range(1, 3):
            for indicator in temp:
                dataset[indicator + "_date_" + str(i)] = dataset[indicator].shift(i)
        dataset.dropna(inplace=True)
        X = dataset.drop(temp, axis=1)
        X = X[-100:]
        predictions = model.predict(X)
        df = data.iloc[-len(predictions) :]
        df["predictions"] = predictions
        #

    prediction_df = pd.concat(
        [data["close"], pd.Series(futurePredictions)], ignore_index=True
    )

    print(modelFileName)

    figure = {
        "data": [
            go.Scatter(
                x=data.index[-300:],
                y=data.close[-300:],
                mode="lines",
                name="Real Price",
                hovertemplate="Real price: %{y}<extra></extra>",
            ),
            go.Scatter(
                x=df.index,
                y=df.predictions,
                mode="lines",
                name="Model Validation in 100 previous data points",
                hovertemplate="Predicted price: %{y}<extra></extra>",
            ),
            go.Scatter(
                x=prediction_df.index[-len(futurePredictions) :],
                y=prediction_df.values[-len(futurePredictions) :],
                mode="markers",
                name="Predicted Price",
                hovertemplate="Predicted price: %{y}<extra></extra>",
            ),
        ],
        "layout": go.Layout(
            title=f"Predicted stock price is {prediction_df.values[-len(futurePredictions)]} USD.",
            hovermode="closest",
            xaxis={"title": "Data Point"},
            yaxis={"title": "Close Price (USD)"},
        ),
    }
    return figure


@app.callback(
    Output("something", "children"),
    [Input("update_button", "n_clicks"), Input("company-dropdown", "value")],
)
def update_output(n_clicks, selected_company):

    if n_clicks:
        update_data(selected_company)
        return html.Div(
            [
                html.H3(
                    f"Data for {selected_company} has been updated!",
                    style={"color": "green"},
                )
            ]
        )
    else:
        return html.Div(
            [html.H3("Choose company to update data", style={"color": "red"})]
        )


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=9999)
