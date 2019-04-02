import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
% matplotlib auto



def process():
    rnames = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_table(r'ratings.data',sep='\t',header=None,names=rnames)
    data = ratings.pivot(index='user_id',columns='movie_id',values='rating')
    foo = DataFrame(np.empty((len(data.index),len(data.index)),dtype=int),index=data.index,columns=data.index)
    for i in foo.index:
        for j in foo.columns:
            foo.ix[i,j] = data.ix[i][data.ix[j].notnull()].dropna().count()
    for i in foo.index:
        foo.ix[i,i]=0#the diagonal numbers' value are 0

    ser = Series(np.zeros(len(foo.index)))
    for i in foo.index:
        ser[i]=foo[i].max()#calculate the max value

    max_value=ser.idxmax()
    max_line_1=ser[max_value]
    x=foo[foo==max_line_1][max_value].dropna()#find the other user_id which have the max value
    max_line_2=x.index[0]
    print(max_line_1,max_line_2)
    h=data.ix[max_line_1].corr(data.ix[max_line_2])
    print(h)
    test = data.reindex([max_line_2,max_line_1],columns=data.ix[max_line_1][data.ix[max_line_2].notnull()].dropna().index)
    periods_test = DataFrame(np.zeros((20,6)),columns=[10,20,50,100,200,500])
    for i in periods_test.index:
        for j in periods_test.columns:
            sample = test.reindex(columns=np.random.permutation(test.columns)[:j])
            periods_test.ix[i,j] = sample.iloc[0].corr(sample.iloc[1])


    print(periods_test[:5])
    print(periods_test.describe())

def showImage_max_line_1() :

    test.ix[max_line_1].value_counts(sort=False).plot(kind='bar')


def showImage_max_line_2() :

    test.ix[max_line_2].value_counts(sort=False).plot(kind='bar')

process()


