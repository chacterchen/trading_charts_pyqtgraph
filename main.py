from datetime import datetime
import json

import numpy as np
import pyqtgraph as pg
from binance.client import Client
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from pyqtgraph import QtCore, QtGui

import charts
import utility


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


with open('api.json') as f:
    api = json.load(f)
client = Client(api['api_key'], api['api_secret'])

pair = "ETHUSDT"
fetch_time = "365 day ago UTC"
interval = "1d"

klines = client.get_historical_klines(pair, interval, fetch_time)

dates = [i / 1000 for i in np.array(klines, dtype=float)[:, 0]]
opens = np.array(klines, dtype=float)[:, 1]
highs = np.array(klines, dtype=float)[:, 2]
lows = np.array(klines, dtype=float)[:, 3]
closes = np.array(klines, dtype=float)[:, 4]
volume = np.array(klines, dtype=float)[:, 5]

step_size = float(client.get_symbol_info(pair)["filters"][0]["tickSize"])

del klines

# vama = VAMA(dates, highs, lows, 0.05, 100)

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

# p1.addItem(CandlestickHighLowItem(dates, highs, lows))

# p1.addItem(CandlestickItem(dates, opens, highs, lows, closes))

# p1.addItem(BarItem(dates, opens, highs, lows, closes))

p1.addItem(charts.BionicItem(dates, opens, highs, lows, closes))

# p1.addItem(KagiItem(dates, opens, highs, lows, closes, reversalType = 'FixedPercent', reversalValue = 0.05, initialValue = 'HighLow', atr=False, atrPeriod=14))

# p1.addItem(RenkoItem(dates, opens, highs, lows, closes, reversalType = 'FixedPrice', reversalValue = 20, initialValue = 'HighLow', atr=False, atrPeriod=14))

#p1.addItem(PointAndFigureItem(dates, opens, highs, lows, closes, 'FixedPrice', 20, 'HighLow', False, 14))

# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][0] for i in range(len(vama))], pen=pg.mkPen(color= pg.mkColor((255, 17, 0, 128)), width=1)))
# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][1] for i in range(len(vama))]))

# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][2] for i in range(len(vama))]))

# p1.addItem(pg.PlotDataItem([i/1000 for i in np.array(lineKlines, dtype = float) [:,0]], np.array(lineKlines, dtype = float) [:,4], pen=pg.mkPen(color= pg.mkColor((255, 255, 255, 255)), width=2)))

# p1.addItem(HeikinAshiItem(dates, opens, highs, lows, closes))

# p1.addItem(LinearBreakItem(dates, closes, 3))

"""

p1.addItem(
    TPO(
        plot=p1,
        pair=pair,
        interval=interval,
        period="1w",
        fetch_time=fetch_time,
        tickSize=10,
        VAShow=True,
        POCShow=True,
        alphabet=False,
        deployed=False,
        profile_right=True,
        profile_left=True,
        heatmapGradient=[
            [0, "#ff0000"],
            [0.25, "#ffff00"],
            [0.5, "#00ff00"],
            [0.75, "#00ffff"],
            [1, "#0000ff"],
        ],
        heatmapOn=True,
        OpenCloseShow=True,
        DynamicVA=False,
    )
)
"""



vLine = pg.InfiniteLine(
    angle=90,
    movable=False,
    pen=pg.mkPen(color=pg.mkColor((255, 255, 255, 100)), style=QtCore.Qt.DotLine),
)
hLine = pg.InfiniteLine(
    angle=0,
    movable=False,
    pen=pg.mkPen(color=pg.mkColor((255, 255, 255, 100)), style=QtCore.Qt.DotLine),
)
p1.addItem(vLine, ignoreBounds=True)
p1.addItem(hLine, ignoreBounds=True)

vb = pI.getViewBox()

priceText = pg.TextItem(
    anchor=(1, 0.5), fill="#0d0d0d", color=pg.mkColor((255, 255, 255, 100))
)
timeText = pg.TextItem(
    anchor=(0.5, 1), fill="#0d0d0d", color=pg.mkColor((255, 255, 255, 100))
)

p1.scene().addItem(priceText)
p1.scene().addItem(timeText)


def mouse_moved(evt):
    pos = evt[0]
    if pI.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)

        priceText.setText(str(charts.format_value(mousePoint.y(), step_size)))
        priceText.setPos(win.width(), pos.y())

        if mousePoint.x() >= mousePoint.x() - mousePoint.x() % (
            interval_to_unix(interval) / 1000
        ) + (interval_to_unix(interval) / 1000 / 2):
            x = (
                mousePoint.x()
                - mousePoint.x() % (interval_to_unix(interval) / 1000)
                + (interval_to_unix(interval) / 1000)
            )

        else:
            x = mousePoint.x() - mousePoint.x() % (interval_to_unix(interval) / 1000)

        timeText.setText(datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
        vLine.setPos(x)
        timeText.setPos(
            vb.mapViewToScene(QtCore.QPointF(x, mousePoint.y())).x(), win.height()
        )
        hLine.setPos(mousePoint.y())


proxy = pg.SignalProxy(pI.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved)
