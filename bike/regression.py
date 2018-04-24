import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def separate(df, train_percent=85):
    npa = df.as_matrix()
    train_size = int((len(npa)*train_percent)/100)
    return npa[:train_size], npa[train_size:]

def fit(features, ground_truth):
    clf = linear_model.SGDRegressor()
    clf.fit(features, ground_truth)
    return clf

def predict(features, clf):
    result = clf.predict(features)
    size = result.shape[0]
    return result.reshape((size, 1))

def plot_results(predictions, test_gt):
    error = np.absolute(predictions - test_gt)
    items = np.concatenate((predictions, test_gt, error), axis=1)
    results = pd.DataFrame(items, columns=['predictions', 'ground_truth',
                                           'error'])
    plottable = results.rolling(15).mean()
    plottable.plot()
    plt.show()
    return results

if __name__ == "__main__":
    dataset = pd.read_csv('BikeDataset.csv', index_col='instant')\
                .drop(columns=['date', 'year'])
    features = dataset.drop(columns=['casual', 'registered', 'cnt'])
    ground_truth = dataset.loc[:, ['cnt']]
    train_feat, test_feat = separate(features)
    train_gt, test_gt = separate(ground_truth)
    clf = fit(train_feat, train_gt)
    predictions = predict(test_feat, clf)
    results = plot_results(predictions, test_gt)
    print (results)