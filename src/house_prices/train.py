from house_prices.data_processing import pipeline
from sklearn.linear_model import LinearRegression
import pickle


def build_model(df):
    
    xtrain, xtest, ytrain, ytest = pipeline(df)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    pickle.dump(model, open('../models/model.pickle', mode='wb'))
    score = model.score(xtrain, ytrain)
    return score
