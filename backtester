from future import (absolute_import, division, print_function,
                        unicode_literals)
from matplotlib import warnings
from matplotlib.dates import (HOURS_PER_DAY, MIN_PER_HOUR, SEC_PER_MIN,
MONTHS_PER_YEAR, DAYS_PER_WEEK,
SEC_PER_HOUR, SEC_PER_DAY,
num2date, rrulewrapper, YearLocator,
MicrosecondLocator)
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
import numpy as np
import pandas as pd
from backtrader import order

# bearish = pd.read_csv('bearish_comments.csv')
# bullish = pd.read_csv('bullish_comments.csv')

class TestStrategy(bt.Strategy):
    params = dict(
        stop_loss=0.02,  # price is 2% less than the entry point
        trail=False,
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.bearish = pd.read_csv('bearish_comments.csv')
        self.bullish = pd.read_csv('bullish_comments.csv')
        self.bearish['dt'] = pd.to_datetime(self.bearish['dt'])
        self.bullish['dt'] = pd.to_datetime(self.bullish['dt'])
        self.bullish['bullish'] = np.where(self.bullish['TSLA'] > (2 * self.bearish['TSLA']) , True, False)
        self.bearish['bearish'] = np.where(self.bearish['TSLA'] > (2 * self.bullish['TSLA']), True, False)
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        self.order = None

    def next(self):
        dt = self.datas[0].datetime.date(0)
        # Simply log the closing price of the series from the reference
        buy = self.bullish[self.bullish['dt'].isin(
            [datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0, 0),
             datetime.datetime(dt.year, dt.month, dt.day, 23, 59, 59, 99)])].iloc[0].bullish
        sell = self.bearish[self.bearish['dt'].isin(
            [datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0, 0),
             datetime.datetime(dt.year, dt.month, dt.day, 23, 59, 59, 99)])].iloc[0].bearish
        self.log('Close, %.2f' % self.dataclose[0])
        print(buy)
        if self.order:
            return
        if not self.position:
            if buy:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.last_executed_price = self.dataclose[0]
                self.order = self.buy()
                # print(self.last_executed_price)
        else:
            if buy == True or buy == False:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                # self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # self.last_executed_price = self.dataclose[0]
                # self.order = self.sell()
                # print(self.last_executed_price)
                if self.dataclose[0] >= ((self.last_executed_price * 0.15) + self.last_executed_price):
                    # SELL, SELL, SELL!!! (with all possible default parameters)
                    self.log('TAKE PROFIT, %.2f' % self.dataclose[0])
                    self.order = self.sell()
                if self.dataclose[0] <= ((self.last_executed_price * 0.03) + self.last_executed_price):
                    # SELL, SELL, SELL!!! (with all possible default parameters)
                    self.log('STOP LOSS, %.2f' % self.dataclose[0])
                    self.order = self.sell()


if name == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))

datapath = os.path.join(modpath, '/home/yan/PycharmProjects/pythonProject/TSLA.csv')
    # Create a Data Feed

    data = bt.feeds.YahooFinanceData(
        dataname=datapath,
        # dtformat=('%Y-%m-%d %H:%M:%S'),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
