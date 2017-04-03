import numpy as np
import _pickle
import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime
from pandas.io.data import DataReader


def load_params(path):
    return np.load(path)
    #f = open(path, 'r', encoding='utf-8')
    #obj = _pickle.load(f)
    #f.close()
    #return obj


def save_params(obj, path):
    return np.save(path, obj)
    #f = open(path, 'w', encoding='utf-8')
    #_pickle.dump(obj, f, protocol=-1)
    #f.close()


def get_dow_constituents():
    page = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    r = requests.get(page)

    tickers = []
    if r.status_code == 200:
        soup = bs(r.content, 'html.parser')
        table = soup.find('table', class_="wikitable sortable")

        for row in table.find_all('tr'):
            cols = row.find_all('td')
            if cols:
                tickers.append(cols[2].string)

    return tickers


def save_data(start, end):
    tickers = get_dow_constituents()
    data = DataReader(tickers, 'yahoo', start, end)

    data['Adj Close'].to_csv('close.csv')
    data['Volume'].to_csv('volume.csv')


def load_data_from_yahoo(start, end, test_split, val_split):
    print('Pulling data from yahoo...')
    tickers = get_dow_constituents()
    data = DataReader(tickers, 'yahoo', start, end)

    pxs = data.ix['Adj Close'].as_matrix()
    vol = data.ix['Volume'].as_matrix()

    #r = (pxs[1:, :] - pxs[:-1, :]) / (pxs[:-1, :] * np.abs(np.max(pxs, axis=0)))
    #vol_diff = (vol[1:, :] - vol[:-1, :]) / (vol[:-1, :] * np.abs(np.max(vol, axis=0)))

    r = (pxs[1:, :] - pxs[:-1, :]) / pxs[:-1, :]
    r = (r - np.mean(r, axis=0)) / np.std(r, axis=0)

    vol_diff = (vol[1:, :] - vol[:-1, :]) / vol[:-1, :]
    vol_diff = (vol_diff - np.mean(vol_diff, axis=0)) / np.std(vol_diff, axis=0)

    inputs = np.concatenate((r[:-1, :], vol_diff[:-1, :]), axis=1)
    targets = r[1:, :]

    train_in, test_in = split_data(inputs, test_split)
    train_in, val_in = split_data(train_in, val_split)

    train_targ, test_targ = split_data(targets, test_split)
    train_targ, val_targ = split_data(train_targ, val_split)

    train_data = {'inputs': train_in, 'targets': train_targ}
    val_data = {'inputs': val_in, 'targets': val_targ}
    test_data = {'inputs': test_in, 'targets': test_targ}

    return train_data, val_data, test_data


def split_data(x, ratio):
    n = int(len(x) * (1 - ratio))
    return x[:n], x[n:]


if __name__ == '__main__':
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2017, 3, 1)
    #save_data(start, end)
    load_data_from_yahoo(start_date, end_date, test_split=0.3, val_split=0.2)
