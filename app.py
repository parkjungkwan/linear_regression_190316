import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from linear_regression import LinearRegression
from gradient_descent import GradientDescent
from k_mean import Kmean
from naive_bayse import NaiveBayseClassfier
import pandas as pd

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    l = LinearRegression()
    g = GradientDescent()
    k = Kmean()
    nb = NaiveBayseClassfier()
    nb.train('data/review_train.csv')

    print("댓글이 긍정인지 부정인지 파악하라")
    check = nb.classify("너무 좋아요. 내 인생 최고의 명작 영화")
    print(check)
    #name = g.execute()
    return render_template('index.html', name=check)


if __name__ == '__main__':
    app.run()
