from datetime import datetime, timedelta
import numpy as np
import time
import pyqtgraph as pg
from time import mktime
from pyqtgraph import AxisItem

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
    def __init__(self, spacing, stepper, format, autoSkip=None):
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
    def __init__(self, tickSpecs):
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
                    formatStrings.append(x.strftime(tickSpec.format)[:-3])
                else:
                    formatStrings.append(x.strftime(tickSpec.format))
            except ValueError:
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
    _pxLabelWidth = 80

    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self._oldAxis = None

    def tickValues(self, minVal, maxVal, size):
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
            fmt = "[+%fms]"  # explicitly relative to last second

        for x in values:
            try:
                t = datetime.fromtimestamp(x)
                ret.append(t.strftime(fmt))
            except ValueError:  # Windows can't handle dates before 1970
                ret.append("")

        return ret

    def attachToPlotItem(self, plotItem):
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
        raise NotImplementedError()  # TODO