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


def readFile(fileName, hour):
    orders_dict = {}
    with open(fileName) as f:
        for line in f:
            line = Line(line)
            if (" " + hour + ":") in line.time:
                orders_dict[line.exchangesOrders.date] = line.exchangesOrders
    return orders_dict


def draw(file, prefix, hour):
    data = readFile(file, hour)

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


draw("eth.log", "bid", '04')