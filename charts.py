import string
from decimal import Decimal

import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from pyqtgraph import QtCore, QtGui


def format_value(val, step_size):
    return float(Decimal(str(val)) - (Decimal(str(val)) % Decimal(str(step_size))))


def heatmap(
    value=50,
    start_value=13,
    finish_value=89,
    colormap=[
        [0, "#0b1924"],
        [0.25, "#1a4863"],
        [0.5, "#ffffff"],
        [0.75, "#ff9c00"],
        [1, "#ff0000"],
    ],
    to_hex=True,
):
    if value < start_value:
        return colormap[0][1]
    elif value > finish_value:
        return colormap[-1][1]

    for i in range(len(colormap) - 1):
        if (
            colormap[i][0]
            <= (value - start_value) / (finish_value - start_value)
            <= colormap[i + 1][0]
        ):
            h = colormap[i][1]
            rgb_1 = tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            h_2 = colormap[i + 1][1]
            rgb_2 = tuple(int(h_2.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            change = (
                (value - start_value) / (finish_value - start_value) - colormap[i][0]
            ) / (colormap[i + 1][0] - colormap[i][0])
            rgb = [0, 0, 0]

            for i in range(3):
                if rgb_1[i] > rgb_2[i]:
                    rgb[i] = int(rgb_1[i] - (rgb_1[i] - rgb_2[i]) * change)
                elif rgb_1[i] < rgb_2[i]:
                    rgb[i] = int(rgb_1[i] + (rgb_2[i] - rgb_1[i]) * change)
                else:
                    rgb[i] = rgb_1[i]

            return "#%02x%02x%02x" % tuple(rgb) if to_hex else tuple(rgb)


def POC_VAH_VAL(volume):
    volume.sort(key=lambda x: x[0])
    volume = np.array(volume)
    POC = np.amax(volume[:, 1])
    POC_index = 0
    total_volume = 0
    value_area = POC

    for i in range(len(volume)):
        total_volume += volume[i][1]
        if POC == volume[i][1]:
            POC_index = i

    VAH = POC_index + 1 if POC_index < len(volume) - 1 else POC_index
    VAL = POC_index - 1 if POC_index > 0 else POC_index

    value_area += volume[VAH][1] + volume[VAL][1]

    while value_area / total_volume < 0.68:
        if VAH < len(volume) - 1:
            VAH += 1
            value_area += volume[VAH][1]
        if VAL > 0:
            VAL -= 1
            value_area += volume[VAL][1]

    POC = volume[POC_index][0]
    VAH = volume[VAH][0]
    VAL = volume[VAL][0]

    return POC, VAH, VAL


def SMA(values, period):
    SMA_values = []

    for i in range(period - 1, len(values)):
        gap = 0
        for j in range(i, i - period, -1):
            gap += values[j]
        SMA_values.append(gap / period)

    return SMA_values


class TPO(pg.GraphicsObject):
    def __init__(
        self,
        plot,
        interval,
        period,
        tick_size,
        VA_show,
        POC_show,
        alphabet,
        deployed,
        heatmap_gradient,
        heatmap_on,
        open_close_show,
        dynamic_VA,
    ):
        pg.GraphicsObject.__init__(self)

        self.plot = plot
        self.interval = np.array(interval, dtype=float)
        self.period = np.array(period, dtype=float)
        self.tick_size = tick_size
        self.VA_show = VA_show
        self.POC_show = POC_show
        self.alphabet = alphabet
        self.deployed = deployed
        self.heatmap_gradient = heatmap_gradient
        self.heatmap_on = heatmap_on
        self.open_close_show = open_close_show
        self.dynamic_VA = dynamic_VA

        if self.open_close_show:
            self.period_open_close = [
                [
                    format_value(self.period[i][1], self.tick_size),
                    format_value(self.period[i][4], self.tick_size),
                ]
                for i in range(len(self.period))
            ]

        self.period = self.period[:, 0]

        self.max_letter = int(
            (self.period[1] - self.period[0])
            / (self.interval[:, 0][1] - self.interval[:, 0][0])
        )

        self.TPO = []

        if self.heatmap_on:
            self.intervals_period = 0

            for j in range(len(self.interval)):
                if self.interval[:, 0][j] >= self.period[0] + (
                    self.period[1] - self.period[0]
                ):
                    break

                if self.interval[:, 0][j] >= self.period[0]:
                    self.intervals_period += 1

        for i in range(len(self.period)):
            self.TPO.append([self.period[i], {}])

            if self.open_close_show:
                self.TPO[-1].append(self.period_open_close[i])

            n = 0

            for j in range(len(self.interval)):
                if self.interval[:, 0][j] >= self.period[i] + (
                    self.period[1] - self.period[0]
                ):
                    break

                if self.interval[:, 0][j] >= self.period[i]:
                    current_high = format_value(self.interval[:, 2][j], self.tick_size)
                    current_low = format_value(self.interval[:, 3][j], self.tick_size)

                    price = current_high

                    while price >= current_low:
                        if str(price) in self.TPO[-1][1]:
                            self.TPO[-1][1][str(price)][0] += 1
                            self.TPO[-1][1][str(price)][1] += (
                                string.ascii_uppercase[n]
                                if n <= 25
                                else string.ascii_lowercase[n]
                            )

                        else:
                            self.TPO[-1][1][str(price)] = [
                                1,
                                string.ascii_uppercase[n]
                                if n <= 25
                                else string.ascii_lowercase[n],
                            ]

                        price = float(
                            Decimal(str(price)) - Decimal(str(self.tick_size))
                        )

                    n += 1

        for i in range(len(self.TPO)):
            self.TPO[i][0] /= 1000
            self.TPO[i][1] = [[float(k), self.TPO[i][1][k]] for k in self.TPO[i][1]]

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        w = (self.interval[:, 0][1] - self.interval[:, 0][0]) / 3000.0

        p.setPen(pg.mkPen(None))

        for i in range(len(self.TPO)):
            square_start = self.TPO[i][0]

            z = 0

            POC, VAH, VAL = POC_VAH_VAL(
                [
                    [self.TPO[i][1][n][0], self.TPO[i][1][n][1][0]]
                    for n in range(len(self.TPO[i][1]))
                ]
            )

            if self.deployed:
                for j in range(len(self.interval)):
                    if self.interval[:, 0][j] / 1000 >= self.TPO[i][0] + (
                        self.TPO[1][0] - self.TPO[0][0]
                    ):
                        break

                    if self.interval[:, 0][j] / 1000 >= self.TPO[i][0]:
                        current_high = format_value(
                            self.interval[:, 2][j], self.tick_size
                        )
                        current_low = format_value(
                            self.interval[:, 3][j], self.tick_size
                        )
                        price = current_high

                        while price >= current_low:
                            if self.heatmap_on:
                                color = heatmap(
                                    z,
                                    0,
                                    self.intervals_period,
                                    self.heatmap_gradient,
                                    False,
                                )
                                p.setBrush(
                                    pg.mkBrush((color[0], color[1], color[2], 85))
                                )
                                color = (color[0], color[1], color[2], 85)

                                if self.VA_show:
                                    if VAL <= price <= VAH:
                                        p.setBrush(
                                            pg.mkBrush(
                                                (color[0], color[1], color[2], 255)
                                            )
                                        )
                                        color = (color[0], color[1], color[2], 255)

                            else:
                                p.setBrush(pg.mkBrush("#cccccc"))
                                color = "#cccccc"

                                if self.VA_show:
                                    if VAL <= price <= VAH:
                                        p.setBrush(pg.mkBrush("#0080C0"))
                                        color = "#0080C0"

                                if self.POC_show:
                                    if price == POC:
                                        p.setBrush(pg.mkBrush("#ff3333"))
                                        color = "#ff3333"

                            if self.alphabet:
                                text = pg.TextItem(
                                    text=string.ascii_uppercase[z]
                                    if z <= 25
                                    else string.ascii_lowercase[z],
                                    anchor=(0.5, 0.5),
                                    color=pg.mkColor(color),
                                )
                                self.plot.addItem(text)
                                text.setPos(self.interval[:, 0][j] / 1000, price)
                                text.setFont(
                                    QFont("Bahnschrift", 10, weight=QtGui.QFont.Bold)
                                )

                            else:
                                p.drawRect(
                                    QtCore.QRectF(
                                        self.interval[:, 0][j] / 1000 - w,
                                        price
                                        + self.tick_size * 0.5
                                        - self.tick_size * 0.1,
                                        w * 2,
                                        -self.tick_size * 0.8,
                                    )
                                )

                            price = float(
                                Decimal(str(price)) - Decimal(str(self.tick_size))
                            )

                        z += 1

            else:
                for j in range(len(self.TPO[i][1])):
                    square_start = self.TPO[i][0]

                    for n in range(len(self.TPO[i][1][j][1][1])):
                        if self.heatmap_on:
                            color = heatmap(
                                string.ascii_uppercase.index(
                                    self.TPO[i][1][j][1][1][n]
                                ),
                                0,
                                self.intervals_period,
                                self.heatmap_gradient,
                                False,
                            )

                            if (
                                self.TPO[i][1][j][1][1][n] == "A"
                                and self.TPO[i][1][j][0] == self.TPO[i][2][0]
                            ):
                                color = (255, 128, 0)

                            if (
                                self.TPO[i][1][j][1][1][n]
                                == string.ascii_uppercase[self.max_letter - 1]
                                and self.TPO[i][1][j][0] == self.TPO[i][2][1]
                            ):
                                color = (255, 255, 0)

                            p.setBrush(pg.mkBrush((color[0], color[1], color[2], 85)))
                            color = (color[0], color[1], color[2], 85)

                            if self.VA_show:
                                if VAL <= self.TPO[i][1][j][0] <= VAH:
                                    p.setBrush(
                                        pg.mkBrush((color[0], color[1], color[2], 255))
                                    )
                                    color = (color[0], color[1], color[2], 255)

                        else:
                            p.setBrush(pg.mkBrush("#cccccc"))
                            color = "#cccccc"

                            if self.VA_show:
                                if VAL <= self.TPO[i][1][j][0] <= VAH:
                                    p.setBrush(pg.mkBrush("#0080C0"))
                                    color = "#0080C0"

                            if self.POC_show:
                                if self.TPO[i][1][j][0] == POC:
                                    p.setBrush(pg.mkBrush("#ff3333"))
                                    color = "#ff3333"

                        if self.alphabet:
                            text = pg.TextItem(
                                text=self.TPO[i][1][j][1][1][n],
                                anchor=(0.5, 0.5),
                                color=pg.mkColor(color),
                            )
                            self.plot.addItem(text)
                            text.setPos(
                                (
                                    square_start
                                    + (
                                        self.interval[:, 0][1] / 1000
                                        - self.interval[:, 0][0] / 1000
                                    )
                                    * (n + 1)
                                ),
                                self.TPO[i][1][j][0],
                            )
                            text.setFont(
                                QFont("Bahnschrift", 10, weight=QtGui.QFont.Bold)
                            )

                        else:
                            p.drawRect(
                                QtCore.QRectF(
                                    (
                                        square_start
                                        + (
                                            self.interval[:, 0][1] / 1000
                                            - self.interval[:, 0][0] / 1000
                                        )
                                        * (n)
                                    )
                                    - w,
                                    self.TPO[i][1][j][0]
                                    + self.tick_size * 0.5
                                    - self.tick_size * 0.1,
                                    w * 2,
                                    -self.tick_size * 0.8,
                                )
                            )

            if self.heatmap_on and self.POC_show:
                p.setBrush(pg.mkBrush(None))
                p.setPen(pg.mkPen("#ffffff"))
                p.drawRect(
                    QtCore.QRectF(
                        square_start
                        - (self.interval[:, 0][1] - self.interval[:, 0][0]) / 2000,
                        POC + self.tick_size * 0.5,
                        self.TPO[1][0] - self.TPO[0][0],
                        -self.tick_size,
                    )
                )
                p.setPen(pg.mkPen(None))

        if self.dynamic_VA:
            self.dynamic_VP = []
            self.TPO = []

            for i in range(len(self.period)):
                self.TPO.append([self.period[i], {}])

                n = 0

                for j in range(len(self.interval)):
                    if self.interval[:, 0][j] >= self.period[i] + (
                        self.period[1] - self.period[0]
                    ):
                        break

                    if self.interval[:, 0][j] >= self.period[i]:
                        current_high = format_value(
                            self.interval[:, 2][j], self.tick_size
                        )

                        current_low = format_value(
                            self.interval[:, 3][j], self.tick_size
                        )
                        price = current_high

                        while price >= current_low:
                            if str(price) in self.TPO[-1][1]:
                                self.TPO[-1][1][str(price)] += 1
                            else:
                                self.TPO[-1][1][str(price)] = 1
                            price = float(
                                Decimal(str(price)) - Decimal(str(self.tick_size))
                            )

                        n += 1

                        self.dynamic_VP.append(
                            [
                                self.interval[:, 0][j] / 1000,
                                POC_VAH_VAL(
                                    [
                                        [float(k), self.TPO[-1][1][k]]
                                        for k in self.TPO[-1][1]
                                    ]
                                ),
                            ]
                        )

            self.plot.addItem(
                pg.PlotDataItem(
                    [self.dynamic_VP[i][0] for i in range(len(self.dynamic_VP))],
                    [self.dynamic_VP[i][1][0] for i in range(len(self.dynamic_VP))],
                    pen=pg.mkPen(color=pg.mkColor("#27a9e6"), width=1),
                )
            )

            self.plot.addItem(
                pg.PlotDataItem(
                    [self.dynamic_VP[i][0] for i in range(len(self.dynamic_VP))],
                    [self.dynamic_VP[i][1][1] for i in range(len(self.dynamic_VP))],
                    pen=pg.mkPen(color=pg.mkColor("#c9814d"), width=1),
                )
            )

            self.plot.addItem(
                pg.PlotDataItem(
                    [self.dynamic_VP[i][0] for i in range(len(self.dynamic_VP))],
                    [self.dynamic_VP[i][1][2] for i in range(len(self.dynamic_VP))],
                    pen=pg.mkPen(color=pg.mkColor("#c9814d"), width=1),
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CandlestickHighLowItem(pg.GraphicsObject):
    def __init__(self, dates, highs, lows):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.highs = highs
        self.lows = lows

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        p.setPen(pg.mkPen("#000000"))
        p.setBrush(pg.mkBrush("#cccccc"))

        for i in range(len(self.dates)):
            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.highs[i],
                    w * 2,
                    self.lows[i] - self.highs[i],
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, dates, opens, highs, lows, closes):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        for i in range(len(self.dates)):
            c = "#F70606" if self.opens[i] > self.closes[i] else "#0CF50D"
            p.setBrush(pg.mkBrush(c))
            p.setPen(pg.mkPen(c))

            p.drawLine(
                QtCore.QPointF(self.dates[i], self.lows[i]),
                QtCore.QPointF(self.dates[i], self.highs[i]),
            )

            p.setPen(pg.mkPen("#000000"))

            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.closes[i],
                    w * 2,
                    self.opens[i] - self.closes[i],
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class BarItem(pg.GraphicsObject):
    def __init__(self, dates, opens, highs, lows, closes):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        for i in range(len(self.dates)):
            c = "#F70606" if self.opens[i] > self.closes[i] else "#0CF50D"
            p.setBrush(pg.mkBrush(c))
            p.setPen(pg.mkPen(c, width=2))

            p.drawLine(
                QtCore.QPointF(self.dates[i], self.lows[i]),
                QtCore.QPointF(self.dates[i], self.highs[i]),
            )

            p.drawLine(
                QtCore.QPointF(self.dates[i], self.closes[i]),
                QtCore.QPointF(self.dates[i] + w, self.closes[i]),
            )

            p.drawLine(
                QtCore.QPointF(self.dates[i], self.opens[i]),
                QtCore.QPointF(self.dates[i] - w, self.opens[i]),
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class BionicItem(pg.GraphicsObject):
    def __init__(self, dates, opens, highs, lows, closes):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        for i in range(len(self.dates)):
            p.setPen(pg.mkPen(None))

            p.setBrush(pg.mkBrush("#0CF50D"))
            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.closes[i],
                    w * 2,
                    -(self.closes[i] - self.lows[i]),
                )
            )
            p.setBrush(pg.mkBrush("#F70606"))
            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.highs[i],
                    w * 2,
                    -(self.highs[i] - self.closes[i]),
                )
            )

            p.setPen(pg.mkPen("#000000"))
            p.setBrush(pg.mkBrush(None))
            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.highs[i],
                    w * 2,
                    self.highs[i] - self.lows[i],
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class HeikinAshiItem(pg.GraphicsObject):
    def __init__(self, dates, opens, highs, lows, closes):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.opens = [np.mean([opens[0], closes[0]])]
        self.highs = [highs[0]]
        self.lows = [lows[0]]
        self.closes = [np.mean([opens[0], highs[0], lows[0], closes[0]])]

        for i in range(1, len(dates)):
            self.opens.append(np.mean([self.opens[-1], self.closes[-1]]))
            self.highs.append(max(highs[i], self.opens[-1], self.closes[-1]))
            self.lows.append(min(lows[i], self.opens[-1], self.closes[-1]))
            self.closes.append(np.mean([opens[i], highs[i], lows[i], closes[i]]))

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        for i in range(len(self.dates)):
            c = "#F70606" if self.opens[i] > self.closes[i] else "#0CF50D"
            p.setBrush(pg.mkBrush(c))
            p.setPen(pg.mkPen(c))

            p.drawLine(
                QtCore.QPointF(self.dates[i], self.lows[i]),
                QtCore.QPointF(self.dates[i], self.highs[i]),
            )

            p.setPen(pg.mkPen("#000000"))

            p.drawRect(
                QtCore.QRectF(
                    self.dates[i] - w,
                    self.closes[i],
                    w * 2,
                    self.opens[i] - self.closes[i],
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class KagiItem(pg.GraphicsObject):
    def __init__(
        self,
        dates,
        opens,
        highs,
        lows,
        closes,
        reversal_type,
        reversal_value,
        initial_value,
        atr=False,
        atr_period=14,
    ):
        pg.GraphicsObject.__init__(self)

        self.dates = dates[atr_period + 1 :] if atr else dates

        self.time_range = self.dates[1] - self.dates[0]
        self.kagi = np.zeros(len(self.dates))

        if atr:
            atr_filter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atr_period,
            )

        match initial_value:
            case "HighLow":
                pass
            case "OHLC4":
                price_calc = np.array(
                    [
                        np.mean([opens[i], highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "HLC3":
                price_calc = np.array(
                    [
                        np.mean([highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "Open":
                price_calc = opens
            case "High":
                price_calc = highs
            case "Low":
                price_calc = lows
            case "Close":
                price_calc = closes
            case _:
                raise "No such klines data!"

        trend = 0
        last_high = 0
        last_low = 0

        if initial_value != "HighLow":
            highs, lows = price_calc, price_calc

        for i in range(len(self.dates)):
            if trend == 0:
                if initial_value == "HighLow":
                    if highs[i] >= lows[0] + (
                        (atr_filter[0] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else lows[0] * reversal_value
                    ):
                        trend = 1
                        last_high = i
                        self.kagi[0] = -1

                    if lows[i] <= highs[0] - (
                        (atr_filter[0] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else highs[0] * reversal_value
                    ):
                        trend = -1
                        last_low = i
                        self.kagi[0] = 1

            if trend == 1:
                if (
                    lows[i]
                    <= highs[last_high]
                    - (
                        (atr_filter[last_high] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else highs[last_high] * reversal_value
                    )
                    and highs[i] < highs[last_high]
                ):
                    trend = -1
                    self.kagi[last_high] = 1
                    last_low = i
                if highs[i] > highs[last_high]:
                    last_high = i

            if trend == -1:
                if (
                    highs[i]
                    >= lows[last_low]
                    + (
                        (atr_filter[last_low] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else lows[last_low] * reversal_value
                    )
                    and lows[i] > lows[last_low]
                ):
                    trend = 1
                    self.kagi[last_low] = -1
                    last_high = i
                if lows[i] < lows[last_low]:
                    last_low = i

        last_trend = 0
        last_trend_index = 0

        for i in range(len(self.kagi) - 1, -1, -1):
            if self.kagi[i] != 0:
                last_trend = self.kagi[i]
                last_trend_index = i
                break

        determinant = 0
        high_determinant = highs[last_trend_index]
        low_determinant = lows[last_trend_index]

        for i in range(last_trend_index + 1, len(self.kagi)):
            if last_trend == 1:
                if lows[i] < low_determinant:
                    low_determinant = lows[i]
                    determinant = i
            if last_trend == -1:
                if highs[i] > high_determinant:
                    high_determinant = highs[i]
                    determinant = i

        self.kagi[determinant] = -1 if last_trend == 1 else 1

        self.out = []

        for i in range(len(self.dates)):
            if self.kagi[i] in (1, -1):
                self.out.append(
                    [self.dates[i], highs[i] if self.kagi[i] == 1 else lows[i]]
                )

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        p.setBrush(pg.mkBrush(None))

        for i in range(1, len(self.out)):
            self.out[i][0] = self.out[0][0] + self.time_range * i

        for i in range(1, len(self.out)):
            p.setPen(
                pg.mkPen(
                    "#0CF50D" if self.out[i][1] > self.out[i - 1][1] else "#F70606",
                    width=4,
                )
            )
            p.drawLine(
                QtCore.QPointF(self.out[i][0], self.out[i][1]),
                QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
            )
            p.setPen(
                pg.mkPen(
                    "#F70606" if self.out[i][1] > self.out[i - 1][1] else "#0CF50D",
                    width=4,
                )
            )
            p.drawLine(
                QtCore.QPointF(self.out[i - 1][0], self.out[i - 1][1]),
                QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class RenkoItem(pg.GraphicsObject):
    def __init__(
        self,
        dates,
        opens,
        highs,
        lows,
        closes,
        reversal_type,
        reversal_value,
        initial_value,
        atr=False,
        atr_period=14,
    ):
        pg.GraphicsObject.__init__(self)

        self.dates = dates[atr_period + 1 :] if atr else dates

        self.time_range = self.dates[1] - self.dates[0]
        self.renko = np.zeros(len(self.dates))
        self.reversal_value = reversal_value

        if atr:
            atr_filter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atr_period,
            )

        match initial_value:
            case "HighLow":
                pass
            case "OHLC4":
                price_calc = np.array(
                    [
                        np.mean([opens[i], highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "HLC3":
                price_calc = np.array(
                    [
                        np.mean([highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "Open":
                price_calc = opens
            case "High":
                price_calc = highs
            case "Low":
                price_calc = lows
            case "Close":
                price_calc = closes
            case _:
                raise "No such klines data!"

        trend = 0
        last_high = 0
        last_low = 0

        if initial_value != "HighLow":
            highs, lows = price_calc, price_calc

        for i in range(len(self.dates)):
            if trend == 0:
                if highs[i] >= lows[0] + (
                    (atr_filter[0] if atr else reversal_value)
                    if reversal_type == "FixedPrice"
                    else lows[0] * reversal_value
                ):
                    trend = 1
                    last_high = i
                    self.renko[0] = -1
                if lows[i] <= highs[0] - (
                    (atr_filter[0] if atr else reversal_value)
                    if reversal_type == "FixedPrice"
                    else highs[0] * reversal_value
                ):
                    trend = -1
                    last_low = i
                    self.renko[0] = 1

            if trend == 1:
                if (
                    lows[i]
                    <= highs[last_high]
                    - (
                        (atr_filter[last_high] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else highs[last_high] * reversal_value
                    )
                    and highs[i] < highs[last_high]
                ):
                    trend = -1
                    self.renko[last_high] = 1
                    last_low = i
                if highs[i] > highs[last_high]:
                    last_high = i

            if trend == -1:
                if (
                    highs[i]
                    >= lows[last_low]
                    + (
                        (atr_filter[last_low] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else lows[last_low] * reversal_value
                    )
                    and lows[i] > lows[last_low]
                ):
                    trend = 1
                    self.renko[last_low] = -1
                    last_high = i
                if lows[i] < lows[last_low]:
                    last_low = i

        last_trend = 0
        last_trend_index = 0

        for i in range(len(self.renko) - 1, -1, -1):
            if self.renko[i] != 0:
                last_trend = self.renko[i]
                last_trend_index = i
                break

        determinant = 0
        high_determinant = highs[last_trend_index]
        low_determinant = lows[last_trend_index]

        for i in range(last_trend_index + 1, len(self.renko)):
            if last_trend == 1:
                if lows[i] < low_determinant:
                    low_determinant = lows[i]
                    determinant = i

            if last_trend == -1:
                if highs[i] > high_determinant:
                    high_determinant = highs[i]
                    determinant = i

        self.renko[determinant] = -1 if last_trend == 1 else 1

        self.out = []

        for i in range(len(self.dates)):
            if self.renko[i] in (1, -1):
                self.out.append(
                    [self.dates[i], highs[i] if self.renko[i] == 1 else lows[i]]
                )

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(None))
        j = 0

        for i in range(1, len(self.out)):
            if self.out[i][1] > self.out[i - 1][1]:
                p.setBrush(pg.mkBrush("#0CF50D"))
                current_high = format_value(self.out[i][1], self.reversal_value)
                current_low = format_value(self.out[i - 1][1], self.reversal_value)
                price = current_low
                while price < current_high:
                    p.drawRect(
                        QtCore.QRectF(
                            self.out[1][0] + self.time_range * j,
                            price + self.reversal_value,
                            self.time_range,
                            self.reversal_value,
                        )
                    )
                    price += self.reversal_value
                    j += 1

            if self.out[i][1] < self.out[i - 1][1]:
                p.setBrush(pg.mkBrush("#F70606"))
                current_high = format_value(self.out[i - 1][1], self.reversal_value)
                current_low = format_value(self.out[i][1], self.reversal_value)
                price = current_high
                while price > current_low:
                    p.drawRect(
                        QtCore.QRectF(
                            self.out[1][0] + self.time_range * j,
                            price + self.reversal_value,
                            self.time_range,
                            self.reversal_value,
                        )
                    )
                    price -= self.reversal_value
                    j += 1

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class PointAndFigureItem(pg.GraphicsObject):
    def __init__(
        self,
        dates,
        opens,
        highs,
        lows,
        closes,
        reversal_type,
        reversal_value,
        initial_value,
        atr=False,
        atr_period=14,
    ):
        pg.GraphicsObject.__init__(self)

        self.dates = dates[atr_period + 1 :] if atr else dates

        self.time_range = self.dates[1] - self.dates[0]
        self.point_and_figure = np.zeros(len(self.dates))
        self.reversal_value = reversal_value

        if atr:
            atr_filter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atr_period,
            )

        match initial_value:
            case "HighLow":
                pass
            case "OHLC4":
                price_calc = np.array(
                    [
                        np.mean([opens[i], highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "HLC3":
                price_calc = np.array(
                    [
                        np.mean([highs[i], lows[i], closes[i]])
                        for i in range(len(dates))
                    ],
                    dtype=float,
                )
            case "Open":
                price_calc = opens
            case "High":
                price_calc = highs
            case "Low":
                price_calc = lows
            case "Close":
                price_calc = closes
            case _:
                raise "No such klines data!"

        trend = 0
        last_high = 0
        last_low = 0

        if initial_value != "HighLow":
            highs, lows = price_calc, price_calc

        for i in range(len(self.dates)):
            if trend == 0:
                if highs[i] >= lows[0] + (
                    (atr_filter[0] if atr else reversal_value)
                    if reversal_type == "FixedPrice"
                    else lows[0] * reversal_value
                ):
                    trend = 1
                    last_high = i
                    self.point_and_figure[0] = -1
                if lows[i] <= highs[0] - (
                    (atr_filter[0] if atr else reversal_value)
                    if reversal_type == "FixedPrice"
                    else highs[0] * reversal_value
                ):
                    trend = -1
                    last_low = i
                    self.point_and_figure[0] = 1

            if trend == 1:
                if (
                    lows[i]
                    <= highs[last_high]
                    - (
                        (atr_filter[last_high] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else highs[last_high] * reversal_value
                    )
                    and highs[i] < highs[last_high]
                ):
                    trend = -1
                    self.point_and_figure[last_high] = 1
                    last_low = i
                if highs[i] > highs[last_high]:
                    last_high = i

            if trend == -1:
                if (
                    highs[i]
                    >= lows[last_low]
                    + (
                        (atr_filter[last_low] if atr else reversal_value)
                        if reversal_type == "FixedPrice"
                        else lows[last_low] * reversal_value
                    )
                    and lows[i] > lows[last_low]
                ):
                    trend = 1
                    self.point_and_figure[last_low] = -1
                    last_high = i
                if lows[i] < lows[last_low]:
                    last_low = i

        last_trend = 0
        last_trend_index = 0

        for i in range(len(self.point_and_figure) - 1, -1, -1):
            if self.point_and_figure[i] != 0:
                last_trend = self.point_and_figure[i]
                last_trend_index = i
                break

        determinant = 0
        high_determinant = highs[last_trend_index]
        low_determinant = lows[last_trend_index]

        for i in range(last_trend_index + 1, len(self.point_and_figure)):
            if last_trend == 1:
                if lows[i] < low_determinant:
                    low_determinant = lows[i]
                    determinant = i

            if last_trend == -1:
                if highs[i] > high_determinant:
                    high_determinant = highs[i]
                    determinant = i

        self.point_and_figure[determinant] = -1 if last_trend == 1 else 1

        self.out = []

        for i in range(len(self.dates)):
            if self.point_and_figure[i] in (1, -1):
                self.out.append(
                    [
                        self.dates[i],
                        highs[i] if self.point_and_figure[i] == 1 else lows[i],
                    ]
                )

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setBrush(pg.mkBrush(None))
        j = 0
        for i in range(1, len(self.out)):
            if self.out[i][1] > self.out[i - 1][1]:
                p.setPen(pg.mkPen("#0CF50D", width=1))
                currentHigh = format_value(self.out[i][1], self.reversal_value)
                currentLow = format_value(self.out[i - 1][1], self.reversal_value)
                price = currentLow
                while price < currentHigh:
                    p.drawLine(
                        QtCore.QPointF(
                            self.out[1][0]
                            + self.time_range * j
                            + self.time_range * 0.1,
                            price + self.reversal_value * 0.9,
                        ),
                        QtCore.QPointF(
                            self.out[1][0]
                            + self.time_range * j
                            + self.time_range * 0.9,
                            price + self.reversal_value * 0.1,
                        ),
                    )
                    p.drawLine(
                        QtCore.QPointF(
                            self.out[1][0]
                            + self.time_range * j
                            + self.time_range * 0.1,
                            price + self.reversal_value * 0.1,
                        ),
                        QtCore.QPointF(
                            self.out[1][0]
                            + self.time_range * j
                            + self.time_range * 0.9,
                            price + self.reversal_value * 0.9,
                        ),
                    )
                    price += self.reversal_value
                j += 1
            if self.out[i][1] < self.out[i - 1][1]:
                p.setPen(pg.mkPen("#F70606", width=1))
                currentHigh = format_value(self.out[i - 1][1], self.reversal_value)
                currentLow = format_value(self.out[i][1], self.reversal_value)
                price = currentHigh
                while price > currentLow:
                    p.drawEllipse(
                        QtCore.QRectF(
                            self.out[1][0]
                            + self.time_range * j
                            + self.time_range * 0.1,
                            price + self.reversal_value - self.reversal_value * 0.1,
                            self.time_range * 0.8,
                            -self.reversal_value * 0.8,
                        )
                    )
                    price -= self.reversal_value
                j += 1

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class LinearBreakItem(pg.GraphicsObject):
    def __init__(self, dates, closes, check):
        pg.GraphicsObject.__init__(self)

        self.dates = dates
        self.time_range = self.dates[1] - self.dates[0]
        self.dates = dates[0]
        self.linebreak = []
        self.linebreak.append([closes[0], closes[1]])

        for i in range(2, len(closes)):
            if len(self.linebreak) < check:
                if self.linebreak[-1][1] - self.linebreak[-1][0] > 0:
                    if closes[i] > self.linebreak[-1][1]:
                        self.linebreak.append([self.linebreak[-1][1], closes[i]])
                    if closes[i] < self.linebreak[-1][0]:
                        self.linebreak.append([self.linebreak[-1][0], closes[i]])
                else:
                    if closes[i] < self.linebreak[-1][1]:
                        self.linebreak.append([self.linebreak[-1][1], closes[i]])
                    if closes[i] > self.linebreak[-1][0]:
                        self.linebreak.append([self.linebreak[-1][0], closes[i]])
            else:
                strength = 0
                for j in range(-1, -check, -1):
                    if self.linebreak[j][1] - self.linebreak[j][0] > 0:
                        if self.linebreak[j - 1][1] - self.linebreak[j - 1][0] <= 0:
                            break
                    if self.linebreak[j][1] - self.linebreak[j][0] < 0:
                        if self.linebreak[j - 1][1] - self.linebreak[j - 1][0] >= 0:
                            break
                    if j - 1 == -check:
                        strength = 1
                if strength:
                    if self.linebreak[-1][1] - self.linebreak[-1][0] > 0:
                        if closes[i] > self.linebreak[-1][1]:
                            self.linebreak.append([self.linebreak[-1][1], closes[i]])
                        if closes[i] < self.linebreak[-check][0]:
                            self.linebreak.append([self.linebreak[-1][0], closes[i]])
                    else:
                        if closes[i] < self.linebreak[-1][1]:
                            self.linebreak.append([self.linebreak[-1][1], closes[i]])
                        if closes[i] > self.linebreak[-check][0]:
                            self.linebreak.append([self.linebreak[-1][0], closes[i]])
                else:
                    if self.linebreak[-1][1] - self.linebreak[-1][0] > 0:
                        if closes[i] > self.linebreak[-1][1]:
                            self.linebreak.append([self.linebreak[-1][1], closes[i]])
                        if closes[i] < self.linebreak[-1][0]:
                            self.linebreak.append([self.linebreak[-1][0], closes[i]])
                    else:
                        if closes[i] < self.linebreak[-1][1]:
                            self.linebreak.append([self.linebreak[-1][1], closes[i]])
                        if closes[i] > self.linebreak[-1][0]:
                            self.linebreak.append([self.linebreak[-1][0], closes[i]])

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(None))
        for i in range(len(self.linebreak)):
            p.setBrush(
                pg.mkBrush(
                    "#0CF50D"
                    if self.linebreak[i][1] - self.linebreak[i][0] > 0
                    else "#F70606"
                )
            )
            p.drawRect(
                QtCore.QRectF(
                    self.dates + self.time_range * i,
                    self.linebreak[i][0],
                    self.time_range,
                    self.linebreak[i][1] - self.linebreak[i][0],
                )
            )
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


def VAMA(dates, highs, lows, tickSize, period):

    VA = {}

    PVV = []

    for i in range(period - 1, len(dates)):
        for n in range(i, i - period, -1):
            currentHigh = format_value(highs[n], tickSize)
            currentLow = format_value(lows[n], tickSize)
            price = currentHigh

            while price >= currentLow:
                if str(price) in VA:
                    VA[str(price)] += 1
                else:
                    VA[str(price)] = 1
                price = float(Decimal(str(price)) - Decimal(str(tickSize)))

        VA = [[float(k), VA[k]] for k in VA]
        PVV.append([POC_VAH_VAL(VA), dates[i]])
        VA = {}

    return PVV