import sys
import pandas as pd
import matplotlib.pyplot as plt

def season_analysis(dataset):
    df = dataset.groupby(['season']).mean()
    df['ratio'] = df.registered / df.casual
    df.plot(title='rentals on seasons', kind='bar',
            y=['cnt', 'registered', 'casual'])
    df.plot(title='seasonal registered/casual ratio',
            y=['ratio'], kind='bar')
    plt.show()
    return df.ratio

def workday_analysis(dataset):
    df = dataset.groupby(['workingday']).mean()
    df['ratio'] = df.registered / df.casual
    df.plot(title='rentals on working days', kind='bar',
            y=['cnt', 'registered', 'casual'])
    df.plot(title='working days registered/casual ratio',
            y=['ratio'], kind='bar')
    return df.ratio
    

def weather_analysis(dataset):
    df = dataset.groupby(['weather']).mean()
    df['ratio'] = df.registered / df.casual
    df.plot(title='mean bike rentals on different weather situations',
            kind='bar', y=['cnt', 'registered', 'casual'])
    df.plot(title='weather registered/casual ratio',
            y=['ratio'], kind='bar')
    return df.ratio

def rentals_analysis(dataset, rolling=None):
    df = dataset.loc[:, ['cnt', 'registered', 'casual']]
    df['ratio'] = df.registered / df.casual
    if rolling != None:
        df = df.rolling(rolling).mean()   
    df.plot(title='rentals over the two years',
            y=['cnt', 'registered', 'casual'])
    df.plot(title='overall registered/casual ratio',
            y=['ratio'])
    return df.ratio

def statistical_analysis(dataset):
    df = pd.DataFrame()
    df['mean'] = dataset.mean()
    df['min'] = dataset.min()
    df['max'] = dataset.max()
    df['median'] = dataset.median()
    df['var'] = dataset.var()
    df['std'] = dataset.std()
    return df
    

if __name__ == "__main__":
    dataset = pd.read_csv('BikeDataset.csv', index_col='instant')
    analysis = sys.argv[1:]
    if 'rentals' in analysis:
        rentals_ratio = rentals_analysis(dataset, rolling=1)
    if 'weather' in analysis:
        weather_ratio = weather_analysis(dataset)
    if 'workday' in analysis:
        workday_ratio = workday_analysis(dataset)
    if 'season' in analysis:
        season_ratio = season_analysis(dataset)
    if 'statistical' in analysis:
        df = statistical_analysis(dataset)
        print (df)
    #print (ratio)
    plt.show()