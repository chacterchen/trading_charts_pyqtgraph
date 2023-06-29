import json
from datetime import datetime

import charts
import numpy as np
import pyqtgraph as pg
import utility
from binance.client import Client
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from pyqtgraph import QtCore, QtGui


def interval_to_unix(interval):
    intervals = {
        "1m": 60000,
        "3m": 180000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "2h": 7200000,
        "4h": 14400000,
        "6h": 21600000,
        "8h": 28800000,
        "12h": 43200000,
        "1d": 86400000,
        "3d": 259200000,
        "1w": 604800000,
        "1M": 2592000000,
    }

    return intervals[interval]


def mouse_moved(evt):
    pos = evt[0]
    if pI.sceneBoundingRect().contains(pos):
        mouse_point = vb.mapSceneToView(pos)

        utility.price_text.setText(str(charts.format_value(mouse_point.y(), tick_size)))
        utility.price_text.setPos(win.width(), pos.y())

        if mouse_point.x() >= mouse_point.x() - mouse_point.x() % (
            interval_to_unix(interval) / 1000
        ) + (interval_to_unix(interval) / 1000 / 2):
            x = (
                mouse_point.x()
                - mouse_point.x() % (interval_to_unix(interval) / 1000)
                + (interval_to_unix(interval) / 1000)
            )

        else:
            x = mouse_point.x() - mouse_point.x() % (interval_to_unix(interval) / 1000)

        utility.time_text.setText(
            datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )
        utility.v_line.setPos(x)
        utility.time_text.setPos(
            vb.mapViewToScene(QtCore.QPointF(x, mouse_point.y())).x(), win.height()
        )
        utility.h_line.setPos(mouse_point.y())


with open("api.json") as f:
    api = json.load(f)

client = Client(api["api_key"], api["api_secret"])

pair = "BTCUSDT"
fetch_time = "365 day ago UTC"
interval = "1d"

klines = client.get_historical_klines(pair, interval, fetch_time)

dates = [i / 1000 for i in np.array(klines, dtype=float)[:, 0]]
opens = np.array(klines, dtype=float)[:, 1]
highs = np.array(klines, dtype=float)[:, 2]
lows = np.array(klines, dtype=float)[:, 3]
closes = np.array(klines, dtype=float)[:, 4]
volume = np.array(klines, dtype=float)[:, 5]

tick_size = float(client.get_symbol_info(pair)["filters"][0]["tickSize"])

app = QtWidgets.QApplication([])
app.setWindowIcon(QIcon("icon.png"))

win = pg.GraphicsLayoutWidget(title=f"{pair} {interval} Chart", size=(1600, 900))
win.setBackground("#000000")
win.show()

grad = QtGui.QLinearGradient(0, 0, 0, 1)
grad.setColorAt(0.1, pg.mkColor("#082131"))
grad.setColorAt(0.9, pg.mkColor("#022E41"))

l = pg.GraphicsLayout()
win.setCentralWidget(l)
date_axis = utility.DateAxisItem(orientation="bottom")
pI = pg.PlotItem(axisItems={"bottom": date_axis})
p1 = pI.vb

l.addItem(pI)

pI.showAxis("right")
pI.hideAxis("left")


p1.addItem(
    charts.TPO(
        plot=p1,
        interval=client.get_historical_klines(pair, "1d", fetch_time),
        period=client.get_historical_klines(pair, "1w", fetch_time),
        tick_size=100,
        VA_show=True,
        POC_show=True,
        alphabet=False,
        deployed=False,
        heatmap_gradient=[
            [0, "#ff0000"],
            [0.25, "#ffff00"],
            [0.5, "#00ff00"],
            [0.75, "#00ffff"],
            [1, "#0000ff"],
        ],
        heatmap_on=True,
        open_close_show=True,
        dynamic_VA=False,
    )
)


p1.addItem(utility.v_line, ignoreBounds=True)
p1.addItem(utility.h_line, ignoreBounds=True)

vb = pI.getViewBox()

p1.scene().addItem(utility.price_text)
p1.scene().addItem(utility.time_text)

proxy = pg.SignalProxy(pI.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved)
