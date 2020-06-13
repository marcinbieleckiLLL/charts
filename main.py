import sys
import matplotlib.pyplot as plt
import regex as re
from datetime import datetime as date


class ExchangesOrders:
    def __init__(self, line, date):
        self.dict = self.createDict(line)
        self.date = date

    def get(self, const):
        return self.dict[const]

    def createDict(self, line):
        dict = {}
        for item in line.split(";"):
            if isinstance(item.split(":"), list) and len(item.split(":")) > 1:
                key, value = item.split(":")
                dict[key] = value
        return dict


class Line:
    def __init__(self, line):
        self.time = self.getTime(line)
        self.exchangesOrders = self.getExchangeOrders(line)

    def getTime(self, line):
        return re.search("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line).group(0)

    def getExchangeOrders(self, line):
        return ExchangesOrders(line.split(' - ')[1], date.strptime(self.time, "%Y-%m-%d %H:%M:%S"))


def readFile(fileName, from_date, to_date):
    orders_dict = {}
    with open(fileName) as file:
        for line in file:
            line = Line(line)
            if from_date is None or to_date is None or from_date < line.exchangesOrders.date < to_date:
                orders_dict[line.exchangesOrders.date] = line.exchangesOrders
    return orders_dict


def getData(file, prefix, from_date, to_date):
    data = readFile(file, from_date, to_date)
    time = list(key for key in data.keys())
    values = list(float(value.get(prefix)) for value in data.values())
    return [file, time, values]


def draw(file, prefix, from_date, to_date):
    data = readFile(file, from_date, to_date)

    time = list(key for key in data.keys())
    binance = list(float(value.get("binance1" + prefix)) for value in data.values())
    bitmex = list(float(value.get("bitmex1" + prefix)) for value in data.values())
    ftx = list(float(value.get("ftx1" + prefix)) for value in data.values())
    okex = list(float(value.get("okex1" + prefix)) for value in data.values())

    plt.plot(time, binance)
    plt.plot(time, bitmex)
    plt.plot(time, ftx)
    plt.plot(time, okex)

    plt.legend(['binance', 'bitmex', 'ftx', 'okex'])
    plt.savefig(file.split(".")[0] + ".png")


def draw(data):
    for d in data:
        plt.plot(d[1], d[2])
    plt.legend(list(d[0].split(".")[0] for d in data))
    plt.savefig("abc.png")


def main(dateFrom, timeFrom, dateTo, timeTo):
    yFrom = int(dateFrom.split("-")[0])
    MFrom = int(dateFrom.split("-")[1])
    dFrom = int(dateFrom.split("-")[2])
    hFrom = int(timeFrom.split(":")[0])
    mFrom = int(timeFrom.split(":")[1])

    yTo = int(dateTo.split("-")[0])
    MTo = int(dateTo.split("-")[1])
    dTo = int(dateTo.split("-")[2])
    hTo = int(timeTo.split(":")[0])
    mTo = int(timeTo.split(":")[1])

    btc = getData("btc.log", "okex1bid", date(yFrom, MFrom, dFrom, hFrom, mFrom), date(yTo, MTo, dTo, hTo, mTo))
    btcFutureWeek = getData("btcFutureWeek.log", "okex1bid", date(yFrom, MFrom, dFrom, hFrom, mFrom),
                            date(yTo, MTo, dTo, hTo, mTo))
    btcFutureTwoWeeks = getData("btcFutureTwoWeeks.log", "okex1bid", date(yFrom, MFrom, dFrom, hFrom, mFrom),
                                date(yTo, MTo, dTo, hTo, mTo))
    btcFutureQuarter = getData("btcFutureQuarter.log", "okex1bid", date(yFrom, MFrom, dFrom, hFrom, mFrom),
                               date(yTo, MTo, dTo, hTo, mTo))
    btcFutureTwoQuarters = getData("btcFutureTwoQuarters.log", "okex1bid", date(yFrom, MFrom, dFrom, hFrom, mFrom),
                                   date(yTo, MTo, dTo, hTo, mTo))
    draw([btc, btcFutureWeek, btcFutureTwoWeeks, btcFutureQuarter, btcFutureTwoQuarters])


if __name__ == "__main__":
    dateFrom = sys.argv[1]
    timeFrom = sys.argv[2]
    dateTo = sys.argv[3]
    timeTo = sys.argv[4]

    main(dateFrom, timeFrom, dateTo, timeTo)