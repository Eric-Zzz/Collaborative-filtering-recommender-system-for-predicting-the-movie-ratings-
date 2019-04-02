import pandas as pd
from pandas import Series, DataFrame
import numpy as np



def check_min_period():
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(r'ratings.data', sep='\t', header=None, names=rnames)
    data = ratings.pivot(index='user_id', columns='movie_id', values='rating')

    check_size = 250
    check = {}
    check_data = data.copy()#copy a  data to check result
    check_data = check_data.ix[check_data.count(axis=1)>100]#delete the users who didn't rating enough movies(<100)
    for user in np.random.permutation(check_data.index):
        movie = np.random.permutation(check_data.ix[user].dropna().index)[0]
        check[(user,movie)] = check_data.ix[user,movie]
        check_data.ix[user,movie] = np.nan
        check_size -= 1
        if not check_size:
            break


    corr = check_data.T.corr(min_periods=100)
    corr_clean = corr.dropna(how='all')
    corr_clean = corr_clean.dropna(axis=1,how='all')#delete the empty data
    check_ser = Series(check)#these are the 1000 ture data which are selected
    print(check_ser[:5])

    result = Series(np.nan,index=check_ser.index)
    for user,movie in result.index:#calculate the weighted average
        prediction = []
        if user in corr_clean.index:
            corr_set = corr_clean[user][corr_clean[user]>0.1].dropna()#users whose correlation coefficient>=0.1
        else:continue
        for other in corr_set.index:
            if  not np.isnan(data.ix[other,movie]) and other != user:# PS:bool(np.nan)==True
                prediction.append((data.ix[other,movie],corr_set[other]))
        if prediction:
            result[(user,movie)] = sum([value*weight for value,weight in prediction])/sum([pair[1] for pair in prediction])


    result.dropna(inplace=True)
    len(result)#the 250 user been selected include the users whose min_periods<100


    print(result[:5])


    print(result.corr(check_ser.reindex(result.index)))


    print((result-check_ser.reindex(result.index)).abs().describe())
    #The absolute value of the difference between the recommended recommendation and the actual evaluation


check_min_period()