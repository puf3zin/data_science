import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import random

def separate(df, train_percent=85):
    npa = df.as_matrix()
    test_size = int((len(npa)*(100-train_percent))/100)
    test = []
    for elem in range(test_size):
        chosen = random.randint(0, npa.shape[0]-1)
        test.append(npa[chosen])
        np.delete(npa, chosen)
    return npa, np.array(test)

def fit(features, ground_truth):
    clf = linear_model.Perceptron()
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