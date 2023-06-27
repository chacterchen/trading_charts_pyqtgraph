import string
import time
from datetime import datetime, timedelta
from decimal import Decimal
from time import mktime
import json

import numpy as np
import pyqtgraph as pg
from binance.client import Client
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont, QIcon
from pyqtgraph import AxisItem, QtCore, QtGui


def get_klines(pair="BTCUSDT", interval="1DAY", fetch_time="7 day ago UTC"):
    if interval == "1m":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_1MINUTE, fetch_time
        )

    if interval == "3m":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_3MINUTE, fetch_time
        )

    if interval == "5m":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_5MINUTE, fetch_time
        )

    if interval == "15m":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_15MINUTE, fetch_time
        )

    if interval == "30m":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_30MINUTE, fetch_time
        )

    if interval == "1h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_1HOUR, fetch_time
        )

    if interval == "2h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_2HOUR, fetch_time
        )

    if interval == "4h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_4HOUR, fetch_time
        )

    if interval == "6h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_6HOUR, fetch_time
        )

    if interval == "8h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_8HOUR, fetch_time
        )

    if interval == "12h":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_12HOUR, fetch_time
        )

    if interval == "1d":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_1DAY, fetch_time
        )

    if interval == "3d":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_3DAY, fetch_time
        )

    if interval == "1w":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_1WEEK, fetch_time
        )

    if interval == "1M":
        return client.get_historical_klines(
            pair, Client.KLINE_INTERVAL_1MONTH, fetch_time
        )


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


def POC_VAH_VAL(volume):
    volume.sort(key=lambda x: x[0])

    volume = np.array(volume)

    POC = np.amax(volume[:, 1])

    POC_index = None

    total_volume = 0

    value_area = POC

    for i in range(len(volume)):
        total_volume += volume[i][1]

        if POC == volume[i][1]:
            POC_index = i

    if POC_index < len(volume) - 1:
        VAH = POC_index + 1

    else:
        VAH = POC_index

    if POC_index > 0:
        VAL = POC_index - 1

    else:
        VAL = POC_index

    value_area += volume[VAH][1]

    value_area += volume[VAL][1]

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
    SMA_array = []

    for i in range(period - 1, len(values)):
        gap = 0

        for j in range(i, i - period, -1):
            gap += values[j]

        gap /= period

        SMA_array.append(gap)

    return SMA_array


f = open('api.json')
api = json.load(f)
client = Client(api['api_key'], api['api_secret'])

pair = "ETHUSDT"
fetch_time = "365 day ago UTC"
interval = "1d"

klines = get_klines(pair, interval, fetch_time)

dates = np.array(klines, dtype=float)[:, 0]  # open time
dates = [i / 1000 for i in dates]
opens = np.array(klines, dtype=float)[:, 1]  # open
highs = np.array(klines, dtype=float)[:, 2]  # high
lows = np.array(klines, dtype=float)[:, 3]  # low
closes = np.array(klines, dtype=float)[:, 4]  # close
volume = np.array(klines, dtype=float)[:, 5]

step_size = float(client.get_symbol_info(pair)["filters"][0]["tickSize"])

del klines

__all__ = ["DateAxisItem", "ZoomLevel"]

MS_SPACING = 1 / 1000.0
SECOND_SPACING = 1
MINUTE_SPACING = 60
HOUR_SPACING = 3600
DAY_SPACING = 24 * HOUR_SPACING
WEEK_SPACING = 7 * DAY_SPACING
MONTH_SPACING = 30 * DAY_SPACING
YEAR_SPACING = 365 * DAY_SPACING


def makeMSStepper(stepSize):
    def stepper(val, n):
        val *= 1000
        f = stepSize * 1000
        return (val // (n * f) + 1) * (n * f) / 1000.0

    return stepper


def makeSStepper(stepSize):
    def stepper(val, n):
        return (val // (n * stepSize) + 1) * (n * stepSize)

    return stepper


def makeMStepper(stepSize):
    def stepper(val, n):
        d = datetime.utcfromtimestamp(val)
        base0m = d.month + n * stepSize - 1
        d = datetime(d.year + base0m // 12, base0m % 12 + 1, 1)
        return (d - datetime(1970, 1, 1)).total_seconds()

    return stepper


def makeYStepper(stepSize):
    def stepper(val, n):
        d = datetime.utcfromtimestamp(val)
        next_date = datetime((d.year // (n * stepSize) + 1) * (n * stepSize), 1, 1)
        return (next_date - datetime(1970, 1, 1)).total_seconds()

    return stepper


class TickSpec:
    """Specifies the properties for a set of date ticks and computes ticks
    within a given utc timestamp range"""

    def __init__(self, spacing, stepper, format, autoSkip=None):
        """
        ============= =========================================================
        Arguments
        spacing       approximate (average) tick spacing
        stepper       a stepper function that takes a utc time stamp and a step
                      steps number n to compute the start of the next unit. You
                      can use the make_X_stepper functions to create common
                      steppers.
        format        a strftime compatible format string which will be used to
                      convert tick locations to date/time strings
        autoSkip      list of step size multipliers to be applied when the tick
                      density becomes too high. The tick spec automatically
                      applies additional powers of 10 (10, 100, ...) to the
                      list if necessary. Set to None to switch autoSkip off
        ============= =========================================================
        """
        self.spacing = spacing
        self.step = stepper
        self.format = format
        self.autoSkip = autoSkip

    def makeTicks(self, minVal, maxVal, minSpc):
        ticks = []
        n = self.skipFactor(minSpc)
        x = self.step(minVal, n)
        while x <= maxVal:
            ticks.append(x)
            x = self.step(x, n)
        return (np.array(ticks), n)

    def skipFactor(self, minSpc):
        if self.autoSkip is None or minSpc < self.spacing:
            return 1
        factors = np.array(self.autoSkip)
        while True:
            for f in factors:
                spc = self.spacing * f
                if spc > minSpc:
                    return f
            factors *= 10


class ZoomLevel:
    """Generates the ticks which appear in a specific zoom level"""

    def __init__(self, tickSpecs):
        """
        ============= =========================================================
        tickSpecs     a list of one or more TickSpec objects with decreasing
                      coarseness
        ============= =========================================================
        """
        self.tickSpecs = tickSpecs
        self.utcOffset = 0

    def tickValues(self, minVal, maxVal, minSpc):
        allTicks = []
        valueSpecs = []
        utcMin = minVal - self.utcOffset
        utcMax = maxVal - self.utcOffset
        for spec in self.tickSpecs:
            ticks, skipFactor = spec.makeTicks(utcMin, utcMax, minSpc)
            ticks += self.utcOffset
            tick_list = [x for x in ticks.tolist() if x not in allTicks]
            allTicks.extend(tick_list)
            valueSpecs.append((spec.spacing, tick_list))
            if skipFactor > 1:
                break
        return valueSpecs


YEAR_MONTH_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(YEAR_SPACING, makeYStepper(1), "%Y", autoSkip=[1, 5, 10, 25]),
        TickSpec(MONTH_SPACING, makeMStepper(1), "%b"),
    ]
)
MONTH_DAY_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(MONTH_SPACING, makeMStepper(1), "%b"),
        TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), "%d", autoSkip=[1, 5]),
    ]
)
DAY_HOUR_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), "%a %d"),
        TickSpec(HOUR_SPACING, makeSStepper(HOUR_SPACING), "%H:%M", autoSkip=[1, 6]),
    ]
)
HOUR_MINUTE_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), "%a %d"),
        TickSpec(
            MINUTE_SPACING, makeSStepper(MINUTE_SPACING), "%H:%M", autoSkip=[1, 5, 15]
        ),
    ]
)
HMS_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(
            SECOND_SPACING,
            makeSStepper(SECOND_SPACING),
            "%H:%M:%S",
            autoSkip=[1, 5, 15, 30],
        )
    ]
)
MS_ZOOM_LEVEL = ZoomLevel(
    [
        TickSpec(MINUTE_SPACING, makeSStepper(MINUTE_SPACING), "%H:%M:%S"),
        TickSpec(
            MS_SPACING, makeMSStepper(MS_SPACING), "%S.%f", autoSkip=[1, 5, 10, 25]
        ),
    ]
)


class DateAxisItem(AxisItem):
    def __init__(self, orientation, utcOffset=None, **kvargs):
        super(DateAxisItem, self).__init__(orientation, **kvargs)
        if utcOffset is None:
            self.utcOffset = time.timezone
        else:
            self.utcOffset = utcOffset
        self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
        self.maxTicksPerPt = 1 / 60.0
        self.zoomLevels = {
            self.maxTicksPerPt: MS_ZOOM_LEVEL,
            30 * self.maxTicksPerPt: HMS_ZOOM_LEVEL,
            15 * 60 * self.maxTicksPerPt: HOUR_MINUTE_ZOOM_LEVEL,
            6 * 3600 * self.maxTicksPerPt: DAY_HOUR_ZOOM_LEVEL,
            5 * 3600 * 24 * self.maxTicksPerPt: MONTH_DAY_ZOOM_LEVEL,
            3600 * 24 * 30 * self.maxTicksPerPt: YEAR_MONTH_ZOOM_LEVEL,
        }

    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        dates = [datetime.utcfromtimestamp(v - self.utcOffset) for v in values]
        formatStrings = []
        for x in dates:
            try:
                if "%f" in tickSpec.format:
                    # we only support ms precision
                    formatStrings.append(x.strftime(tickSpec.format)[:-3])
                else:
                    formatStrings.append(x.strftime(tickSpec.format))
            except ValueError:  # Windows can't handle dates before 1970
                formatStrings.append("")
        return formatStrings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self.setZoomLevelForDensity(density)
        minSpacing = density / self.maxTicksPerPt
        values = self.zoomLevel.tickValues(minVal, maxVal, minSpc=minSpacing)
        return values

    def setZoomLevelForDensity(self, density):
        keys = sorted(self.zoomLevels.keys())
        key = next((k for k in keys if density < k), keys[-1])
        self.zoomLevel = self.zoomLevels[key]
        self.zoomLevel.utcOffset = self.utcOffset


class TimeAxisItem(pg.AxisItem):
    """
    A tool that provides a date-time aware axis. It is implemented as an
    AxisItem that interpretes positions as unix timestamps (i.e. seconds
    since 1970).
    The labels and the tick positions are dynamically adjusted depending
    on the range.
    It provides a  :meth:`attachToPlotItem` method to add it to a given
    PlotItem
    """

    # Max width in pixels reserved for each label in axis
    _pxLabelWidth = 80

    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self._oldAxis = None

    def tickValues(self, minVal, maxVal, size):
        """
        Reimplemented from PlotItem to adjust to the range and to force
        the ticks at "round" positions in the context of time units instead of
        rounding in a decimal base
        """

        maxMajSteps = int(size / self._pxLabelWidth)

        dt1 = datetime.fromtimestamp(minVal)
        dt2 = datetime.fromtimestamp(maxVal)

        dx = maxVal - minVal
        majticks = []

        if dx > 63072001:  # 3600s*24*(365+366) = 2 years (count leap year)
            d = timedelta(days=366)
            for y in range(dt1.year + 1, dt2.year):
                dt = datetime(year=y, month=1, day=1)
                majticks.append(mktime(dt.timetuple()))

        elif dx > 5270400:  # 3600s*24*61 = 61 days
            d = timedelta(days=31)
            dt = dt1.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + d
            while dt < dt2:
                # make sure that we are on day 1 (even if always sum 31 days)
                dt = dt.replace(day=1)
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 172800:  # 3600s24*2 = 2 days
            d = timedelta(days=1)
            dt = dt1.replace(hour=0, minute=0, second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 7200:  # 3600s*2 = 2hours
            d = timedelta(hours=1)
            dt = dt1.replace(minute=0, second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 1200:  # 60s*20 = 20 minutes
            d = timedelta(minutes=10)
            dt = (
                dt1.replace(minute=(dt1.minute // 10) * 10, second=0, microsecond=0) + d
            )
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 120:  # 60s*2 = 2 minutes
            d = timedelta(minutes=1)
            dt = dt1.replace(second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 20:  # 20s
            d = timedelta(seconds=10)
            dt = dt1.replace(second=(dt1.second // 10) * 10, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 2:  # 2s
            d = timedelta(seconds=1)
            majticks = range(int(minVal), int(maxVal))

        else:  # <2s , use standard implementation from parent
            return pg.AxisItem.tickValues(self, minVal, maxVal, size)

        L = len(majticks)
        if L > maxMajSteps:
            majticks = majticks[:: int(np.ceil(float(L) / maxMajSteps))]

        return [(d.total_seconds(), majticks)]

    def tickStrings(self, values, scale, spacing):
        """Reimplemented from PlotItem to adjust to the range"""
        ret = []
        if not values:
            return []

        if spacing >= 31622400:  # 366 days
            fmt = "%Y"

        elif spacing >= 2678400:  # 31 days
            fmt = "%Y %b"

        elif spacing >= 86400:  # = 1 day
            fmt = "%b/%d"

        elif spacing >= 3600:  # 1 h
            fmt = "%b/%d-%Hh"

        elif spacing >= 60:  # 1 m
            fmt = "%H:%M"

        elif spacing >= 1:  # 1s
            fmt = "%H:%M:%S"

        else:
            # less than 2s (show microseconds)
            # fmt = '%S.%f"'
            fmt = "[+%fms]"  # explicitly relative to last second

        for x in values:
            try:
                t = datetime.fromtimestamp(x)
                ret.append(t.strftime(fmt))
            except ValueError:  # Windows can't handle dates before 1970
                ret.append("")

        return ret

    def attachToPlotItem(self, plotItem):
        """Add this axis to the given PlotItem
        :param plotItem: (PlotItem)
        """
        self.setParentItem(plotItem)
        viewBox = plotItem.getViewBox()
        self.linkToView(viewBox)
        self._oldAxis = plotItem.axes[self.orientation]["item"]
        self._oldAxis.hide()
        plotItem.axes[self.orientation]["item"] = self
        pos = plotItem.axes[self.orientation]["pos"]
        plotItem.layout.addItem(self, *pos)
        self.setZValue(-1000)

    def detachFromPlotItem(self):
        """Remove this axis from its attached PlotItem
        (not yet implemented)
        """
        raise NotImplementedError()  # TODO


class TPO(pg.GraphicsObject):
    def __init__(
        self,
        plot,
        pair,
        interval,
        period,
        fetch_time,
        tickSize,
        VAShow,
        POCShow,
        alphabet,
        deployed,
        profile_right,
        profile_left,
        heatmapGradient,
        heatmapOn,
        OpenCloseShow,
        DynamicVA,
    ):
        global TPO_GLOBAL

        pg.GraphicsObject.__init__(self)

        self.plot = plot
        self.pair = pair
        self.interval = interval
        self.period = period
        self.fetch_time = fetch_time
        self.tickSize = tickSize
        self.VAShow = VAShow
        self.POCShow = POCShow
        self.alphabet = alphabet
        self.deployed = deployed
        self.profile_right = profile_right
        self.profile_left = profile_left
        self.heatmapGradient = heatmapGradient
        self.heatmapOn = heatmapOn
        self.OpenCloseShow = OpenCloseShow
        self.DynamicVA = DynamicVA

        if self.OpenCloseShow:
            self.periodOpenClose = np.array(
                get_klines(self.pair, self.period, fetch_time), dtype=float
            )

            self.periodOpenClose = [
                [
                    format_value(self.periodOpenClose[i][1], self.tickSize),
                    format_value(self.periodOpenClose[i][4], self.tickSize),
                ]
                for i in range(len(self.periodOpenClose))
            ]

        self.maxLetter = int(
            interval_to_unix(self.period) / interval_to_unix(self.interval)
        )

        self.period = np.array(
            get_klines(self.pair, self.period, fetch_time), dtype=float
        )[:, 0]

        self.interval = np.array(
            get_klines(self.pair, self.interval, fetch_time), dtype=float
        )

        self.TPO = []

        if self.heatmapOn:
            self.intervalsInPeriod = 0

            for j in range(len(self.interval)):
                if self.interval[:, 0][j] >= self.period[0] + (
                    self.period[1] - self.period[0]
                ):
                    break

                if self.interval[:, 0][j] >= self.period[0]:
                    self.intervalsInPeriod += 1

        for i in range(len(self.period)):
            self.TPO.append([self.period[i], {}])

            if self.OpenCloseShow:
                self.TPO[-1].append(self.periodOpenClose[i])

            n = 0

            for j in range(len(self.interval)):
                if self.interval[:, 0][j] >= self.period[i] + (
                    self.period[1] - self.period[0]
                ):
                    break

                if self.interval[:, 0][j] >= self.period[i]:
                    currentHigh = format_value(self.interval[:, 2][j], self.tickSize)

                    currentLow = format_value(self.interval[:, 3][j], self.tickSize)

                    price = currentHigh

                    while price >= currentLow:
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

                        price = float(Decimal(str(price)) - Decimal(str(self.tickSize)))

                    n += 1

        for i in range(len(self.TPO)):
            self.TPO[i][0] /= 1000

            self.TPO[i][1] = [[float(k), self.TPO[i][1][k]] for k in self.TPO[i][1]]

        TPO_GLOBAL = self.TPO

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        w = (self.interval[:, 0][1] / 1000 - self.interval[:, 0][0] / 1000) / 3.0

        p.setPen(pg.mkPen(None))

        for i in range(len(self.TPO)):
            kvadratStart = self.TPO[i][0]

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
                        currentHigh = format_value(
                            self.interval[:, 2][j], self.tickSize
                        )

                        currentLow = format_value(self.interval[:, 3][j], self.tickSize)

                        price = currentHigh

                        while price >= currentLow:
                            if self.heatmapOn:
                                color = heatmap(
                                    z,
                                    0,
                                    self.intervalsInPeriod,
                                    self.heatmapGradient,
                                    False,
                                )
                                p.setBrush(
                                    pg.mkBrush((color[0], color[1], color[2], 85))
                                )
                                color = (color[0], color[1], color[2], 85)

                                if self.VAShow:
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

                                if self.VAShow:
                                    if VAL <= price <= VAH:
                                        p.setBrush(pg.mkBrush("#0080C0"))
                                        color = "#0080C0"

                                if self.POCShow:
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
                                        + self.tickSize * 0.5
                                        - self.tickSize * 0.1,
                                        w * 2,
                                        -self.tickSize * 0.8,
                                    )
                                )

                            price = float(
                                Decimal(str(price)) - Decimal(str(self.tickSize))
                            )

                        z += 1

                if self.heatmapOn:
                    if self.POCShow:
                        p.setBrush(pg.mkBrush(None))
                        p.setPen(pg.mkPen("#ffffff"))
                        p.drawRect(
                            QtCore.QRectF(
                                kvadratStart
                                - (
                                    self.interval[:, 0][1] / 1000
                                    - self.interval[:, 0][0] / 1000
                                )
                                / 2,
                                POC + self.tickSize * 0.5,
                                self.TPO[1][0] - self.TPO[0][0],
                                -self.tickSize,
                            )
                        )
                        p.setPen(pg.mkPen(None))

            else:
                for j in range(len(self.TPO[i][1])):
                    kvadratStart = self.TPO[i][0]

                    for n in range(len(self.TPO[i][1][j][1][1])):
                        if self.heatmapOn:
                            color = heatmap(
                                string.ascii_uppercase.index(
                                    self.TPO[i][1][j][1][1][n]
                                ),
                                0,
                                self.intervalsInPeriod,
                                self.heatmapGradient,
                                False,
                            )

                            if (
                                self.TPO[i][1][j][1][1][n] == "A"
                                and self.TPO[i][1][j][0] == self.TPO[i][2][0]
                            ):
                                color = (255, 128, 0)

                            if (
                                self.TPO[i][1][j][1][1][n]
                                == string.ascii_uppercase[self.maxLetter - 1]
                                and self.TPO[i][1][j][0] == self.TPO[i][2][1]
                            ):
                                color = (255, 255, 0)

                            p.setBrush(pg.mkBrush((color[0], color[1], color[2], 85)))
                            color = (color[0], color[1], color[2], 85)

                            if self.VAShow:
                                if VAL <= self.TPO[i][1][j][0] <= VAH:
                                    p.setBrush(
                                        pg.mkBrush((color[0], color[1], color[2], 255))
                                    )
                                    color = (color[0], color[1], color[2], 255)

                        else:
                            p.setBrush(pg.mkBrush("#cccccc"))
                            color = "#cccccc"

                            if self.VAShow:
                                if VAL <= self.TPO[i][1][j][0] <= VAH:
                                    p.setBrush(pg.mkBrush("#0080C0"))
                                    color = "#0080C0"

                            if self.POCShow:
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
                                    kvadratStart
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
                                        kvadratStart
                                        + (
                                            self.interval[:, 0][1] / 1000
                                            - self.interval[:, 0][0] / 1000
                                        )
                                        * (n)
                                    )
                                    - w,
                                    self.TPO[i][1][j][0]
                                    + self.tickSize * 0.5
                                    - self.tickSize * 0.1,
                                    w * 2,
                                    -self.tickSize * 0.8,
                                )
                            )

                if self.heatmapOn:
                    if self.POCShow:
                        p.setBrush(pg.mkBrush(None))
                        p.setPen(pg.mkPen("#ffffff"))
                        p.drawRect(
                            QtCore.QRectF(
                                kvadratStart
                                - (
                                    self.interval[:, 0][1] / 1000
                                    - self.interval[:, 0][0] / 1000
                                )
                                / 2,
                                POC + self.tickSize * 0.5,
                                self.TPO[1][0] - self.TPO[0][0],
                                -self.tickSize,
                            )
                        )
                        p.setPen(pg.mkPen(None))

        if self.DynamicVA:
            self.DynamicVPV = []

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
                        currentHigh = format_value(
                            self.interval[:, 2][j], self.tickSize
                        )

                        currentLow = format_value(self.interval[:, 3][j], self.tickSize)

                        price = currentHigh

                        while price >= currentLow:
                            if str(price) in self.TPO[-1][1]:
                                self.TPO[-1][1][str(price)] += 1

                            else:
                                self.TPO[-1][1][str(price)] = 1

                            price = float(
                                Decimal(str(price)) - Decimal(str(self.tickSize))
                            )

                        n += 1

                        self.DynamicVPV.append(
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
                    [self.DynamicVPV[i][0] for i in range(len(self.DynamicVPV))],
                    [self.DynamicVPV[i][1][0] for i in range(len(self.DynamicVPV))],
                    pen=pg.mkPen(color=pg.mkColor("#27a9e6"), width=1),
                )
            )

            self.plot.addItem(
                pg.PlotDataItem(
                    [self.DynamicVPV[i][0] for i in range(len(self.DynamicVPV))],
                    [self.DynamicVPV[i][1][1] for i in range(len(self.DynamicVPV))],
                    pen=pg.mkPen(color=pg.mkColor("#c9814d"), width=1),
                )
            )

            self.plot.addItem(
                pg.PlotDataItem(
                    [self.DynamicVPV[i][0] for i in range(len(self.DynamicVPV))],
                    [self.DynamicVPV[i][1][2] for i in range(len(self.DynamicVPV))],
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


def VAMA(dates, highs, lows, tickSize, period):
    start = time.time()

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

    print(time.time() - start)

    return PVV


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
            if self.opens[i] > self.closes[i]:
                p.setBrush(pg.mkBrush("#F70606"))
                p.setPen(pg.mkPen("#F70606"))
            else:
                p.setBrush(pg.mkBrush("#0CF50D"))
                p.setPen(pg.mkPen("#0CF50D"))

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
            if self.opens[i] > self.closes[i]:
                p.setBrush(pg.mkBrush("#F70606"))
                p.setPen(pg.mkPen("#F70606", width=2))
            else:
                p.setBrush(pg.mkBrush("#0CF50D"))
                p.setPen(pg.mkPen("#0CF50D", width=2))

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

            # p.setPen(pg.mkPen('#000000'))

            # p.drawRect(QtCore.QRectF(self.dates[i]-w, self.closes[i], w*2, self.opens[i]-self.closes[i]))

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
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []

        for i in range(len(dates)):
            if i == 0:
                self.opens.append((opens[i] + closes[i]) / 2)
                self.highs.append(highs[i])
                self.lows.append(lows[i])
                self.closes.append((opens[i] + highs[i] + lows[i] + closes[i]) / 4)

            else:
                self.opens.append((self.opens[-1] + self.closes[-1]) / 2)
                self.closes.append((opens[i] + highs[i] + lows[i] + closes[i]) / 4)
                self.highs.append(max(highs[i], self.opens[-1], self.closes[-1]))
                self.lows.append(min(lows[i], self.opens[-1], self.closes[-1]))

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.dates[1] - self.dates[0]) / 3.0

        for i in range(len(self.dates)):
            if self.opens[i] > self.closes[i]:
                p.setBrush(pg.mkBrush("#F70606"))
                p.setPen(pg.mkPen("#F70606"))
            else:
                p.setBrush(pg.mkBrush("#0CF50D"))
                p.setPen(pg.mkPen("#0CF50D"))

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
        reversalType,
        reversalValue,
        initialValue,
        atr=False,
        atrPeriod=14,
    ):
        # FixedPrice, FixedPercent
        # HighLow, OHLC4, HLC3, Open, High, Low, Close

        pg.GraphicsObject.__init__(self)

        if atr:
            dates = dates[atrPeriod + 1 :]

        self.dates = dates

        self.timeRange = self.dates[1] - self.dates[0]
        self.kagi = np.zeros(len(self.dates))

        if atr:
            atrFilter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atrPeriod,
            )

        if initialValue == "HighLow":
            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if highs[i] >= lows[0] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else lows[0] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.kagi[0] = -1

                    if lows[i] <= highs[0] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else highs[0] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.kagi[0] = 1

                if trend == "Bull":
                    if (
                        lows[i]
                        <= highs[lastHigh]
                        - (
                            (atrFilter[lastHigh] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else highs[lastHigh] * reversalValue
                        )
                        and highs[i] < highs[lastHigh]
                    ):
                        trend = "Bear"

                        self.kagi[lastHigh] = 1

                        lastLow = i

                    if highs[i] > highs[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if (
                        highs[i]
                        >= lows[lastLow]
                        + (
                            (atrFilter[lastLow] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else lows[lastLow] * reversalValue
                        )
                        and lows[i] > lows[lastLow]
                    ):
                        trend = "Bull"

                        self.kagi[lastLow] = -1

                        lastHigh = i

                    if lows[i] < lows[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.kagi) - 1, -1, -1):
                if self.kagi[i] != 0:
                    lastTrend = self.kagi[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.kagi)):
                if lastTrend == 1:
                    if lows[i] < lowDeterminant:
                        lowDeterminant = lows[i]

                        determinant = i

                if lastTrend == -1:
                    if highs[i] > highDeterminant:
                        highDeterminant = highs[i]

                        determinant = i

            if lastTrend == 1:
                self.kagi[determinant] = -1

            if lastTrend == -1:
                self.kagi[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.kagi[i] == 1:
                    out.append([dates[i], highs[i]])

                if self.kagi[i] == -1:
                    out.append([dates[i], lows[i]])

        else:
            if initialValue == "OHLC4":
                priceToCalc = np.array(
                    [
                        (opens[i] + highs[i] + lows[i] + closes[i]) / 4
                        for i in range(len(dates))
                    ]
                )

            if initialValue == "HLC3":
                priceToCalc = np.array(
                    [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(dates))]
                )

            if initialValue == "Open":
                priceToCalc = opens

            if initialValue == "High":
                priceToCalc = highs

            if initialValue == "Low":
                priceToCalc = lows

            if initialValue == "Close":
                priceToCalc = closes

            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if priceToCalc[i] >= priceToCalc[i] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.kagi[0] = -1

                    if priceToCalc[i] <= priceToCalc[i] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.kagi[0] = 1

                if trend == "Bull":
                    if priceToCalc[i] <= priceToCalc[lastHigh] - (
                        (atrFilter[lastHigh] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastHigh] * reversalValue
                    ):
                        trend = "Bear"

                        self.kagi[lastHigh] = 1

                        lastLow = i

                    if priceToCalc[i] > priceToCalc[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if priceToCalc[i] >= priceToCalc[lastLow] + (
                        (atrFilter[lastLow] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastLow] * reversalValue
                    ):
                        trend = "Bull"

                        self.kagi[lastLow] = -1

                        lastHigh = i

                    if priceToCalc[i] < priceToCalc[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.kagi) - 1, -1, -1):
                if self.kagi[i] != 0:
                    lastTrend = self.kagi[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.kagi)):
                if lastTrend == 1:
                    if priceToCalc[i] < lowDeterminant:
                        lowDeterminant = priceToCalc[i]

                        determinant = i

                if lastTrend == -1:
                    if priceToCalc[i] > highDeterminant:
                        highDeterminant = priceToCalc[i]

                        determinant = i

            if lastTrend == 1:
                self.kagi[determinant] = -1

            if lastTrend == -1:
                self.kagi[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.kagi[i] == 1:
                    out.append([dates[i], priceToCalc[i]])

                if self.kagi[i] == -1:
                    out.append([dates[i], priceToCalc[i]])

        self.out = out

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        p.setBrush(pg.mkBrush(None))

        for i in range(1, len(self.out)):
            self.out[i][0] = self.out[0][0] + self.timeRange * i

        for i in range(1, len(self.out)):
            if self.out[i][1] > self.out[i - 1][1]:
                p.setPen(pg.mkPen("#0CF50D", width=6))
                p.drawLine(
                    QtCore.QPointF(self.out[i][0], self.out[i][1]),
                    QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
                )
                p.setPen(pg.mkPen("#F70606", width=6))
                p.drawLine(
                    QtCore.QPointF(self.out[i - 1][0], self.out[i - 1][1]),
                    QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
                )
            else:
                p.setPen(pg.mkPen("#F70606", width=6))
                p.drawLine(
                    QtCore.QPointF(self.out[i][0], self.out[i][1]),
                    QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
                )
                p.setPen(pg.mkPen("#0CF50D", width=6))
                p.drawLine(
                    QtCore.QPointF(self.out[i - 1][0], self.out[i - 1][1]),
                    QtCore.QPointF(self.out[i][0], self.out[i - 1][1]),
                )

        """
		for i in range(1, len(self.out)):
			
			if self.out[i][1] > self.out[i-1][1]:
				p.setPen(pg.mkPen('#0CF50D', width=3))
				p.drawLine(QtCore.QPointF(self.out[i-1][0], self.out[i-1][1]), QtCore.QPointF(self.out[i][0], self.out[i][1]))
			else:
				p.setPen(pg.mkPen('#F70606', width=3))
				p.drawLine(QtCore.QPointF(self.out[i-1][0], self.out[i-1][1]), QtCore.QPointF(self.out[i][0], self.out[i][1]))
			#p.drawRect(QtCore.QRectF(self.dates[i]-w, self.closes[i], w*2, self.opens[i]-self.closes[i]))
		"""
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
        reversalType,
        reversalValue,
        initialValue,
        atr=False,
        atrPeriod=14,
    ):
        # FixedPrice, FixedPercent
        # HighLow, OHLC4, HLC3, Open, High, Low, Close

        pg.GraphicsObject.__init__(self)

        if atr:
            dates = dates[atrPeriod + 1 :]

        self.dates = dates

        self.timeRange = self.dates[1] - self.dates[0]
        self.renko = np.zeros(len(self.dates))
        self.reversalValue = reversalValue

        if atr:
            atrFilter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atrPeriod,
            )

        if initialValue == "HighLow":
            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if highs[i] >= lows[0] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else lows[0] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.renko[0] = -1

                    if lows[i] <= highs[0] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else highs[0] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.renko[0] = 1

                if trend == "Bull":
                    if (
                        lows[i]
                        <= highs[lastHigh]
                        - (
                            (atrFilter[lastHigh] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else highs[lastHigh] * reversalValue
                        )
                        and highs[i] < highs[lastHigh]
                    ):
                        trend = "Bear"

                        self.renko[lastHigh] = 1

                        lastLow = i

                    if highs[i] > highs[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if (
                        highs[i]
                        >= lows[lastLow]
                        + (
                            (atrFilter[lastLow] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else lows[lastLow] * reversalValue
                        )
                        and lows[i] > lows[lastLow]
                    ):
                        trend = "Bull"

                        self.renko[lastLow] = -1

                        lastHigh = i

                    if lows[i] < lows[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.renko) - 1, -1, -1):
                if self.renko[i] != 0:
                    lastTrend = self.renko[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.renko)):
                if lastTrend == 1:
                    if lows[i] < lowDeterminant:
                        lowDeterminant = lows[i]

                        determinant = i

                if lastTrend == -1:
                    if highs[i] > highDeterminant:
                        highDeterminant = highs[i]

                        determinant = i

            if lastTrend == 1:
                self.renko[determinant] = -1

            if lastTrend == -1:
                self.renko[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.renko[i] == 1:
                    out.append([dates[i], highs[i]])

                if self.renko[i] == -1:
                    out.append([dates[i], lows[i]])

        else:
            if initialValue == "OHLC4":
                priceToCalc = np.array(
                    [
                        (opens[i] + highs[i] + lows[i] + closes[i]) / 4
                        for i in range(len(dates))
                    ]
                )

            if initialValue == "HLC3":
                priceToCalc = np.array(
                    [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(dates))]
                )

            if initialValue == "Open":
                priceToCalc = opens

            if initialValue == "High":
                priceToCalc = highs

            if initialValue == "Low":
                priceToCalc = lows

            if initialValue == "Close":
                priceToCalc = closes

            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if priceToCalc[i] >= priceToCalc[i] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.renko[0] = -1

                    if priceToCalc[i] <= priceToCalc[i] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.renko[0] = 1

                if trend == "Bull":
                    if priceToCalc[i] <= priceToCalc[lastHigh] - (
                        (atrFilter[lastHigh] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastHigh] * reversalValue
                    ):
                        trend = "Bear"

                        self.renko[lastHigh] = 1

                        lastLow = i

                    if priceToCalc[i] > priceToCalc[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if priceToCalc[i] >= priceToCalc[lastLow] + (
                        (atrFilter[lastLow] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastLow] * reversalValue
                    ):
                        trend = "Bull"

                        self.renko[lastLow] = -1

                        lastHigh = i

                    if priceToCalc[i] < priceToCalc[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.renko) - 1, -1, -1):
                if self.renko[i] != 0:
                    lastTrend = self.renko[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.renko)):
                if lastTrend == 1:
                    if priceToCalc[i] < lowDeterminant:
                        lowDeterminant = priceToCalc[i]

                        determinant = i

                if lastTrend == -1:
                    if priceToCalc[i] > highDeterminant:
                        highDeterminant = priceToCalc[i]

                        determinant = i

            if lastTrend == 1:
                self.renko[determinant] = -1

            if lastTrend == -1:
                self.renko[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.renko[i] == 1:
                    out.append([dates[i], priceToCalc[i]])

                if self.renko[i] == -1:
                    out.append([dates[i], priceToCalc[i]])

        self.out = out
        print(self.out)
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        p.setPen(pg.mkPen(None))

        j = 0

        for i in range(1, len(self.out)):
            if self.out[i][1] > self.out[i - 1][1]:
                p.setBrush(pg.mkBrush("#0CF50D"))
                currentHigh = format_value(self.out[i][1], self.reversalValue)
                currentLow = format_value(self.out[i - 1][1], self.reversalValue)
                price = currentLow

                while price < currentHigh:
                    p.drawRect(
                        QtCore.QRectF(
                            self.out[1][0] + self.timeRange * j,
                            price + self.reversalValue,
                            self.timeRange,
                            self.reversalValue,
                        )
                    )

                    price += self.reversalValue

                    j += 1

            if self.out[i][1] < self.out[i - 1][1]:
                p.setBrush(pg.mkBrush("#F70606"))
                currentHigh = format_value(self.out[i - 1][1], self.reversalValue)
                currentLow = format_value(self.out[i][1], self.reversalValue)
                price = currentHigh

                while price > currentLow:
                    p.drawRect(
                        QtCore.QRectF(
                            self.out[1][0] + self.timeRange * j,
                            price + self.reversalValue,
                            self.timeRange,
                            self.reversalValue,
                        )
                    )

                    price -= self.reversalValue

                    j += 1

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class PointandFigureItem(pg.GraphicsObject):
    def __init__(
        self,
        dates,
        opens,
        highs,
        lows,
        closes,
        reversalType,
        reversalValue,
        initialValue,
        atr=False,
        atrPeriod=14,
    ):
        # FixedPrice, FixedPercent
        # HighLow, OHLC4, HLC3, Open, High, Low, Close

        pg.GraphicsObject.__init__(self)

        if atr:
            dates = dates[atrPeriod + 1 :]

        self.dates = dates

        self.timeRange = self.dates[1] - self.dates[0]
        self.PointandFigure = np.zeros(len(self.dates))
        self.reversalValue = reversalValue

        if atr:
            atrFilter = SMA(
                [
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    for i in range(1, len(highs))
                ],
                atrPeriod,
            )

        if initialValue == "HighLow":
            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if highs[i] >= lows[0] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else lows[0] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.PointandFigure[0] = -1

                    if lows[i] <= highs[0] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else highs[0] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.PointandFigure[0] = 1

                if trend == "Bull":
                    if (
                        lows[i]
                        <= highs[lastHigh]
                        - (
                            (atrFilter[lastHigh] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else highs[lastHigh] * reversalValue
                        )
                        and highs[i] < highs[lastHigh]
                    ):
                        trend = "Bear"

                        self.PointandFigure[lastHigh] = 1

                        lastLow = i

                    if highs[i] > highs[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if (
                        highs[i]
                        >= lows[lastLow]
                        + (
                            (atrFilter[lastLow] if atr else reversalValue)
                            if reversalType == "FixedPrice"
                            else lows[lastLow] * reversalValue
                        )
                        and lows[i] > lows[lastLow]
                    ):
                        trend = "Bull"

                        self.PointandFigure[lastLow] = -1

                        lastHigh = i

                    if lows[i] < lows[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.PointandFigure) - 1, -1, -1):
                if self.PointandFigure[i] != 0:
                    lastTrend = self.PointandFigure[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.PointandFigure)):
                if lastTrend == 1:
                    if lows[i] < lowDeterminant:
                        lowDeterminant = lows[i]

                        determinant = i

                if lastTrend == -1:
                    if highs[i] > highDeterminant:
                        highDeterminant = highs[i]

                        determinant = i

            if lastTrend == 1:
                self.PointandFigure[determinant] = -1

            if lastTrend == -1:
                self.PointandFigure[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.PointandFigure[i] == 1:
                    out.append([dates[i], highs[i]])

                if self.PointandFigure[i] == -1:
                    out.append([dates[i], lows[i]])

        else:
            if initialValue == "OHLC4":
                priceToCalc = np.array(
                    [
                        (opens[i] + highs[i] + lows[i] + closes[i]) / 4
                        for i in range(len(dates))
                    ]
                )

            if initialValue == "HLC3":
                priceToCalc = np.array(
                    [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(dates))]
                )

            if initialValue == "Open":
                priceToCalc = opens

            if initialValue == "High":
                priceToCalc = highs

            if initialValue == "Low":
                priceToCalc = lows

            if initialValue == "Close":
                priceToCalc = closes

            trend = None

            lastHigh = None

            lastLow = None

            for i in range(len(dates)):
                if trend == None:
                    if priceToCalc[i] >= priceToCalc[i] + (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bull"

                        lastHigh = i

                        self.PointandFigure[0] = -1

                    if priceToCalc[i] <= priceToCalc[i] - (
                        (atrFilter[0] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[i] * reversalValue
                    ):
                        trend = "Bear"

                        lastLow = i

                        self.PointandFigure[0] = 1

                if trend == "Bull":
                    if priceToCalc[i] <= priceToCalc[lastHigh] - (
                        (atrFilter[lastHigh] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastHigh] * reversalValue
                    ):
                        trend = "Bear"

                        self.PointandFigure[lastHigh] = 1

                        lastLow = i

                    if priceToCalc[i] > priceToCalc[lastHigh]:
                        lastHigh = i

                if trend == "Bear":
                    if priceToCalc[i] >= priceToCalc[lastLow] + (
                        (atrFilter[lastLow] if atr else reversalValue)
                        if reversalType == "FixedPrice"
                        else priceToCalc[lastLow] * reversalValue
                    ):
                        trend = "Bull"

                        self.PointandFigure[lastLow] = -1

                        lastHigh = i

                    if priceToCalc[i] < priceToCalc[lastLow]:
                        lastLow = i

            lastTrend = 0

            lastTrendIndex = 0

            for i in range(len(self.PointandFigure) - 1, -1, -1):
                if self.PointandFigure[i] != 0:
                    lastTrend = self.PointandFigure[i]

                    lastTrendIndex = i

                    break

            determinant = 0

            highDeterminant = highs[lastTrendIndex]

            lowDeterminant = lows[lastTrendIndex]

            for i in range(lastTrendIndex + 1, len(self.PointandFigure)):
                if lastTrend == 1:
                    if priceToCalc[i] < lowDeterminant:
                        lowDeterminant = priceToCalc[i]

                        determinant = i

                if lastTrend == -1:
                    if priceToCalc[i] > highDeterminant:
                        highDeterminant = priceToCalc[i]

                        determinant = i

            if lastTrend == 1:
                self.PointandFigure[determinant] = -1

            if lastTrend == -1:
                self.PointandFigure[determinant] = 1

            out = []

            for i in range(len(dates)):
                if self.PointandFigure[i] == 1:
                    out.append([dates[i], priceToCalc[i]])

                if self.PointandFigure[i] == -1:
                    out.append([dates[i], priceToCalc[i]])

        self.out = out

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        p.setBrush(pg.mkBrush(None))

        j = 0

        for i in range(1, len(self.out)):
            if self.out[i][1] > self.out[i - 1][1]:
                p.setPen(pg.mkPen("#0CF50D", width=1))
                currentHigh = format_value(self.out[i][1], self.reversalValue)
                currentLow = format_value(self.out[i - 1][1], self.reversalValue)
                price = currentLow

                while price < currentHigh:
                    p.drawLine(
                        QtCore.QPointF(
                            self.out[1][0] + self.timeRange * j + self.timeRange * 0.1,
                            price + self.reversalValue * 0.9,
                        ),
                        QtCore.QPointF(
                            self.out[1][0] + self.timeRange * j + self.timeRange * 0.9,
                            price + self.reversalValue * 0.1,
                        ),
                    )

                    p.drawLine(
                        QtCore.QPointF(
                            self.out[1][0] + self.timeRange * j + self.timeRange * 0.1,
                            price + self.reversalValue * 0.1,
                        ),
                        QtCore.QPointF(
                            self.out[1][0] + self.timeRange * j + self.timeRange * 0.9,
                            price + self.reversalValue * 0.9,
                        ),
                    )

                    price += self.reversalValue

                j += 1

            if self.out[i][1] < self.out[i - 1][1]:
                p.setPen(pg.mkPen("#F70606", width=1))
                currentHigh = format_value(self.out[i - 1][1], self.reversalValue)
                currentLow = format_value(self.out[i][1], self.reversalValue)
                price = currentHigh

                while price > currentLow:
                    p.drawEllipse(
                        QtCore.QRectF(
                            self.out[1][0] + self.timeRange * j + self.timeRange * 0.1,
                            price + self.reversalValue - self.reversalValue * 0.1,
                            self.timeRange * 0.8,
                            -self.reversalValue * 0.8,
                        )
                    )

                    price -= self.reversalValue

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
        self.timeRange = self.dates[1] - self.dates[0]
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
            if self.linebreak[i][1] - self.linebreak[i][0] > 0:
                p.setBrush(pg.mkBrush("#0CF50D"))

            else:
                p.setBrush(pg.mkBrush("#F70606"))

            p.drawRect(
                QtCore.QRectF(
                    self.dates + self.timeRange * i,
                    self.linebreak[i][0],
                    self.timeRange,
                    self.linebreak[i][1] - self.linebreak[i][0],
                )
            )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


# vama = VAMA(dates, highs, lows, 0.05, 100)


app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title=f"{pair} {interval} Chart", size=(1600, 900))

grad = QtGui.QLinearGradient(0, 0, 0, 1)
grad.setColorAt(0.1, pg.mkColor("#082131"))
grad.setColorAt(0.9, pg.mkColor("#022E41"))

win.setBackground("#000000")
app.setWindowIcon(QIcon("icon.png"))
win.show()

l = pg.GraphicsLayout()
win.setCentralWidget(l)
date_axis = DateAxisItem(orientation="bottom")
pI = pg.PlotItem(axisItems={"bottom": date_axis})
p1 = pI.vb

l.addItem(pI)

pI.showAxis("right")
pI.hideAxis("left")

# p1.addItem(CandlestickHighLowItem(dates, highs, lows))

# p1.addItem(CandlestickItem(dates, opens, highs, lows, closes))

# p1.addItem(BarItem(dates, opens, highs, lows, closes))

# p1.addItem(BionicItem(dates, opens, highs, lows, closes))

# p1.addItem(KagiItem(dates, opens, highs, lows, closes, reversalType = 'FixedPercent', reversalValue = 0.05, initialValue = 'HighLow', atr=False, atrPeriod=14))

# p1.addItem(RenkoItem(dates, opens, highs, lows, closes, reversalType = 'FixedPrice', reversalValue = 500, initialValue = 'HighLow', atr=False, atrPeriod=14))

# p1.addItem(PointandFigureItem(dates, opens, highs, lows, closes, 'FixedPrice', 500, 'HighLow', True, 14))

# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][0] for i in range(len(vama))], pen=pg.mkPen(color= pg.mkColor((255, 17, 0, 128)), width=1)))
# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][1] for i in range(len(vama))]))

# p1.addItem(pg.PlotCurveItem([vama[i][1] for i in range(len(vama))], [vama[i][0][2] for i in range(len(vama))]))

# lineKlines = get_klines(pair, '1h', fetch_time)

# p1.addItem(pg.PlotDataItem([i/1000 for i in np.array(lineKlines, dtype = float) [:,0]], np.array(lineKlines, dtype = float) [:,4], pen=pg.mkPen(color= pg.mkColor((255, 255, 255, 255)), width=2)))

# p1.addItem(HeikinAshiItem(dates, opens, highs, lows, closes))

# p1.addItem(LinearBreakItem(dates, closes, 3))


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


# cross hair
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


def mouseMoved(evt):
    pos = evt[0]
    if pI.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)

        priceText.setText(str(format_value(mousePoint.y(), step_size)))

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


proxy = pg.SignalProxy(pI.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
