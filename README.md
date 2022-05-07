# BitcoinPricePredictor

## Group members 
1. Cynthia Cheng 
1. Jamie Mei 

## Project Idea

We want to **predict the price of bitcoin** using historical bitcoin data as well as its sentiment analysis over time. We are planning on experimenting with **time series models such as LSTM** to predict token price. To measure sentiment analysis, we will be using the fear and greed API which measures engagement with Bitcoin. 

## Preprocessing 
The original bitcoin historical data used in the project contains 4 millions entries, the code from **bitcoinDataCleaning.ipynb** will create 2 additional dataset files which can be found in the [data](data) folder. We take the final update per day of the bitcoin historical data and then **match the timeline** of the data to the sentiment analysis that contain only 1151 rows. 
- Daily market data
- Daily market data + sentiment analysis 

## LSTM code
The price predictor can be found in **BitcoinPricePredictotLSTM.py** and **bitcoinPricePredictor.ipynb**. They are the same. 

## [Datasets](data/README-Datasets.md)

- [bitcoint historical data (2012 - 2021)]( https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

- [fear and greed Index]( https://alternative.me/crypto/fear-and-greed-index/)

