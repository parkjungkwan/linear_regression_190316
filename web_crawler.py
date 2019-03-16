from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import csv

driver = webdriver.Chrome('data/chromedriver')
driver.get('https://movie.naver.com/movie/sdb/rank/rmovie.nhn')
soup = BeautifulSoup(driver.page_source, 'html.parser')
all_divs = soup.find_all('div', attrs={'class','tit3'})
products = [div.a.string for div in all_divs]

for product in products:
    print(product)

    with open('data/crawl_result.csv', 'w', encoding='UTF-8', newline='') as f:
        # 자동 줄바꿈 생성 방지
        wr = csv.writer(f, delimiter=',')
        wr.writeRow(product)


driver.close()