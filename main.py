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


def readFile(fileName):
    orders_dict = {}
    with open(fileName) as f:
        for line in f:
            line = Line(line)
            orders_dict[line.exchangesOrders.date] = line.exchangesOrders
    return orders_dict


data = readFile("eth.log")

time = list(key for key in data.keys())
binance = list(float(value.get("binance1ask")) for value in data.values())
bitmex = list(float(value.get("bitmex1ask")) for value in data.values())
ftx = list(float(value.get("ftx1ask")) for value in data.values())
okex = list(float(value.get("okex1ask")) for value in data.values())

#plt.plot(['21:42:52', '21:42:54', '21:42:56', '21:42:58'], [9109.36, 9109.05, 9109.04, 9109.56])
plt.plot(time, binance)
plt.plot(time, bitmex)
plt.plot(time, ftx)
plt.plot(time, okex)

plt.legend(['binance', 'bitmex', 'ftx', 'okex'])
plt.plot()