import pandas as pd
from sklearn.model_selection import train_test_split 

def date_to_int(date, start="26.12.2020"):
    start_dt = pd.to_datetime(start, dayfirst=True)
    date_dt  = pd.to_datetime(date, dayfirst=True) 

    return (date_dt - start_dt).days

def split_data_by_date(df, date):
    split_date = date_to_int(date)

    train =  df[df["T1"] <= split_date] 
    test = df[df["T1"] > split_date]

    conflicting = train["duration"] > split_date

    # copy them to test set  
    test = test.append(train[conflicting])
    
    # consore them in train set
    train.loc[conflicting, "duration"] = split_date 
    train.loc[conflicting, "event"] = 0
    
    return test, train


def sample_fast(data, sample_size=0.05, random_state=42):
    test = data.sample(frac=sample_size, random_state=random_state)
    
    mmmap = ~data.index.isin(test.index)
    return test, data[mmmap]


# TODO: for more columns
# TODO: somehow check if we do not broke something
def split_data_stratified_slow(df, test_size=0.05, column="VaccStatus", random_state=42):
    vacc = df[column]
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=vacc)
    return test, train 

def split_data_stratified(df, test_size=0.05, column="VaccStatus", random_state=42):
    selection = []
    for _, gdf in df.groupby(column):
        selection.append(gdf.sample(frac=test_size, random_state=random_state))
    test = pd.concat(selection).sample(frac=1, random_state=random_state) # shuffle

    mmmap = ~df.index.isin(test.index)
    return test, df[mmmap]


# TODO:
# nejaky advance splitting https://towardsdatascience.com/stratified-splitting-of-grouped-datasets-using-optimization-bdc12fb6e691



if __name__ == "__main__":

    smid = pd.read_csv("smid_process.csv")
    print(smid)

    # import timeit

    # f = lambda: split_data_stratified_slow(smid)
    # g = lambda: sample_fast(smid)
    # h = lambda: split_data_stratified(smid)
    
    # print("stratified_slow", timeit.timeit(f, number=10))
    # print("gabi fast (not stratified)", timeit.timeit(g, number=10))
    # print("stratified attempt", timeit.timeit(h, number=10))

    # stratified_slow 575.7302003540099
    # gabi fast (not stratified) 47.51555997901596
    # stratified attempt 105.87422459409572

    
    train, test = split_data_stratified(smid)
    print(" *** train *** ")
    print(train)
    print(" *** test *** ")
    print(test)

    assert len(smid) == len(train) + len(test)

    assert (smid.index.isin(train.index) | smid.index.isin(test.index)).all()
    
    exit()    
    
    smid = pd.read_csv("smid_process.csv")
    print(smid)

    date_to_split = "1.9.2021" 
    train, test = split_data_by_date(smid, date_to_split)
    print(" *** train *** ")
    print(train)
    print(" *** test *** ")
    print(test)

    split_date = date_to_int(date_to_split)
    
    twice = len(smid[(smid["T1"] <= split_date) & (smid["T2"] > split_date)])
    assert len(smid) + twice == len(train) + len(test)
    
    assert (train["T1"] <= split_date).all()

    cfl = test[test["T1"] <= split_date].index

    assert (train.loc[cfl]["event"] == 0).all()
    assert (train.loc[cfl]["T2"] == split_date).all()
    
