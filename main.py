import pandas as pd
from pandas import Series,DataFrame
import numpy as np


def recommend():
    rnames = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_table(r'ratings.data',sep='\t',header=None,names=rnames)
    data = ratings.pivot(index='user_id',columns='movie_id',values='rating')


    corr = data.T.corr(min_periods=100)
    corr_clean = corr.dropna(how='all')
    corr_clean = corr_clean.dropna(axis=1,how='all')
    lucky = input("please input a user id (1~943)")
    #lucky = np.random.permutation(corr_clean.index)[0]
    gift = data.ix[lucky]
    gift = gift[gift.isnull()]#now gift is empty
    corr_lucky = corr_clean[lucky].drop(lucky)#the correlation coefficient between lucky and other user
    corr_lucky = corr_lucky[corr_lucky>0.1].dropna()#selected the correlation coefficient >=0.1
    for movie in gift.index:#find the movies which lucky haven seen before
        prediction = []
        for other in corr_lucky.index:#find all other users whose correlation coefficient>=0.1
            if not np.isnan(data.ix[other,movie]):
                prediction.append((data.ix[other,movie],corr_clean[lucky][other]))
        if prediction:
            gift[movie] = sum([value*weight for value,weight in prediction])/sum([pair[1] for pair in prediction])

    print(lucky)
    gift=gift.dropna().sort_values(ascending=False)#sort the gift by descending order

    print(gift)




recommend()