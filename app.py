import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas_datareader as data
import yfinance as yf
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
import cufflinks as cf
import datetime
from keras.models import load_model
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


# start='2010-01-01'
# end='2023-04-30'

st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

ticker_list = pd.read_csv('constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #ge

st.title('STOCK TREND PREDICTION')
# string_logo = '<img src=%s>' % tickerData.info['logo_url']
# st.markdown(string_logo, unsafe_allow_html=True)
st.header(tickerSymbol)
# st.write(tickerData.info)
st.write('-----------------------------------------------------------------------------------------------------------------------------------------')
st.warning(tickerData.info['sector'])



string_summary = tickerData.info['longBusinessSummary']
st.success(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)


st.header("Trends in Close value of stock ")
st.line_chart(tickerDf['Close'])


st.header("Close value and Open value of the Stock")
fig,ax=plt.subplots(figsize=(16,9))
plt.title('Open Value and Close value')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.plot(tickerDf['Close'])
plt.plot(tickerDf['Open'])
st.pyplot(fig)
#plots





data2=tickerDf
data2 = data2.reset_index()


model=load_model('model_stock.h5')

data2.dropna(inplace=True)
data2.head()

data2.reset_index(drop=True, inplace=True)
# data2.fillna(data.mean(), inplace=True)
# data2.head()
df2 = data2.drop('Date', axis=1)

print(df2)

X = []
Y = []
window_size=100 # yes
for i in range(1 , len(df2) - window_size -1 , 1):
    first = df2.iloc[i,4]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df2.iloc[i + j, 4] - first) / first)
    # for j in range(week):
    temp2.append((df2.iloc[i + window_size, 4] - first) / first)
    # X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
    # Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
    # print(stock2.iloc[i:i+window_size,4])
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array(temp2).reshape(1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

model.evaluate(test_X, test_Y)

predicted  = model.predict(test_X)
test_label = test_Y.reshape(-1,1)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp = data2.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')

# plt.show()
st.header("Actual and predicted")
fig,ax=plt.subplots(figsize=(16,9))
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.plot(predicted,color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label,color = 'red', label = 'Real Stock Price')
st.pyplot(fig)
#plots








#

#------------------------------------------------------------



#SENTIMENT ANALYSIS

from transformers import  PegasusForConditionalGeneration
# PegasusTokenizer,
# , TFPegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import torch

import re


from transformers import AutoModel, AutoTokenizer 
# Define the model repo
model_name = "human-centered-summarization/financial-summarization-pegasus" 
# Download pytorch model
# model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model1 = PegasusForConditionalGeneration.from_pretrained(model_name)




exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

monitored_tickers=['GME','TSLA','BTC']

def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs 




raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}






# def search_for_stock_news_urls(ticker):
#     search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
   
#     # search_url="https://economictimes.indiatimes.com/industry/transportation/airlines-/-aviation/indian-oil-to-recover-most-jet-fuel-dues-from-go-first-through-bank-guarantees/articleshow/99965674.cms"
#     r = requests.get(search_url)
#     soup = BeautifulSoup(r.text, 'html.parser')
#     atags = soup.find_all('a')
#     hrefs = [link['href'] for link in atags]
#     return hrefs 


# raw_urls=search_for_stock_news_urls(tickerSymbol)

# # st.write(raw_urls)


def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

# def strip_unwanted_urls(urls, exclude_list):
#     val = []
#     for url in urls: 
#         if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
#             res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
#             val.append(res)
#     return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}


def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
st.write(articles)
# import transformers
# from transformers import pipeline
# sentiment = pipeline('sentiment-analysis')

# def summarize(articles):
#     summaries = []
#     for article in articles:
#         input_ids = tokenizer.encode(article, return_tensors='pt')
#         output = model1.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
#         summary = tokenizer.decode(output[0], skip_special_tokens=True)
#         summaries.append(summary)
#     return summaries


# summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
# scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}




# def create_output_array(summaries, scores, urls):
#     output = []
#     for ticker in monitored_tickers:
#         for counter in range(len(summaries[ticker])):
#             output_this = [
#                 ticker,
#                 summaries[ticker][counter],
#                 scores[ticker][counter]['label'],
#                 scores[ticker][counter]['score'],
#                 urls[ticker][counter]
#             ]
#             output.append(output_this)
#     return output

# final_output = create_output_array(summaries, scores, cleaned_urls)
# st.write(final_output)




# cleaned_urls=strip_unwanted_urls(raw_urls,exclude_list)
# st.write(cleaned_urls)

# # print(cleaned_urls)


# def scrape_and_process(URLs):
#     ARTICLES = []
#     for url in URLs: 
#         r = requests.get(url)
#         soup = BeautifulSoup(r.text, 'html.parser')
#         paragraphs = soup.find_all('p')
#         text = [paragraph.text for paragraph in paragraphs]
#         words = ' '.join(text).split(' ')[:350]
#         ARTICLE = ' '.join(words)
#         ARTICLES.append(ARTICLE)
#     return ARTICLES


# articles=scrape_and_process(cleaned_urls)
# print(articles)
# st.write(articles)


#     # for article in articles:
#     #     input_ids = tokenizer.encode(article, return_tensors='pt')
#     #     output = model1.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
#     #     summary = tokenizer.decode(output[0], skip_special_tokens=True)
#     # for article in articles:
# # input_ids = tokenizer.encode(articles, return_tensors='pt')
# # print(input_ids)
# # output = model1.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
# # summaries = tokenizer.decode(output[0], skip_special_tokens=True)
# # summaries.append(summary)

#     # return summaries






# # import transformers
# # from transformers import pipeline
# # sentiment = pipeline('sentiment-analysis')

# # scores=sentiment(summaries)

# # def create_output_array(summaries, scores, urls):
# #     output = []
# #     for counter in range(len(summaries)):
# #         output_this = [
# #             summaries[counter],
# #                 scores[counter]['label'],
# #                 scores[counter]['score'],
# #                 urls[counter]
# #                 ]
# #         output.append(output_this)
# #     return output


# # final_output = create_output_array(summaries, scores, cleaned_urls)

# # print(final_output)




