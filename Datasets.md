# Descriptions of Datasets

The datasets used in this project is called ["Bitcoin Historical Data"](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) sourced from Kaggle. Which has a total of 4 million entries spaning from 2012 to 2021 and 7 features that is OHLC (Open, High, Low, Close), volume in BTC, and weighted bitcoin price. 

Another dataset we used is ["Fear and Greed Index"](https://alternative.me/crypto/fear-and-greed-index/) from Alternative that measures sentiment anaylsis for bitcoin. The index tracks how much these individual indicators deviate from past averages and diverges and calculates a score from 0 to 100 with 100 representing maximum greediness and 0 signaling maximum fear. The dataset has 1151 rows and 3 features.

## Within this folder we have 2 datasets.
- bitstamp_2018-02-01_to_2021_03_31.csv
    - The market data converted to daily update instead of minutely (1151, 7)
- merge_bitcoin_n_fear.csv
    - The daily market data, and fear and greed index combined. The fear and greed index is added as additional features to the market data. Which makes the data shape (1151,10)