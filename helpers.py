import os

V2 = False if os.getenv("V2") is None else True

if not V2: 
    class Vacc():
        NONE = '_novacc'
        
        P1 = 'Ppart1'
        P2 = 'Pfull1'
        P3 = 'Pboost1'
        
        M1 =  'Mpart1'
        M2 =  'Mfull1'
        M3 =  'Mboost1'
        
        A1 = 'Apart1'
        A2 = 'Afull1'
    
        J = 'Jfull1'

    class Inf():
        YES = 'inf1+'
        NO  = '_noinf'

    class Severity():
        NONE = '_no_or_unknown_severity'
        SYMPTOMS = 'symptoms'
        HOSPITALIZED = 'hospitalized'
else:
    class Vacc():
        NONE = 'unvacc'
        
        P1 = 'P_first1'
        P2 = 'P1'
        P3 = 'Pboost1'
        
        M1 =  'M_first1'
        M2 =  'M1'
        M3 =  'Mboost1'
        
        A1 = 'A_first1'
        A2 = 'A1'
    
        J = 'J1'

    class Inf():
        YES = 'rest'
        NO  = '_none'

    

def odsmiduj(df):
    df.columns = [x.strip() for x in df.columns]
    return df

def kill_ghosts(df, age_limit=110, inplace=True):
    old = df["Age"] > age_limit
    if inplace:
        df.drop(df[old].index, inplace=True)
        return df
    else:
        return df.drop(df[old].index)

def filter_inf_after_vacc(df):
    return df[(df["InfPrior"] == Inf.YES) & (df["InfPriorTime"] > df["LastVaccTime"])] 

def filter_inf_before_vacc(df):
    return df[(df["InfPrior"] == Inf.YES) & (df["InfPriorTime"] <= df["LastVaccTime"])] 

def filter_inf_after_inf(df):
    """ those who were infected second (or more) time """
    return df[(df["InfPrior"] == Inf.YES) & (df["InfPriorTime"] > df["InfFirstTime"])]

def filter_first_inf(df):
    """ those who were infected once """ 
    return df[(df["InfPrior"] == Inf.YES) & (df["InfPriorTime"] ==  df["InfFirstTime"])]

def filter_age_group(df, lower, upper):
    """ returns rows where lower <= age <= upper """
    return df[(df["Age"] >= lower) & (df["Age"] <= upper)]


def filter_men(df):
    """ returns only men """
    return df[df[f"Sex_M"] == 1]


def filter_women(df):
    """ returns only women """
    return df[df[f"Sex_Z"] == 1]


def filter_vaxx(df):
    return df[df[f"VaccStatus_{Vacc.NONE}"] == 0]


def filter_nonvaxx(df):
    return df[df[f"VaccStatus_{Vacc.NONE}"] == 1]


def filter_unexperienced(df):
    """ unexperienced - not infected yet """
    return df[df[f"InfPrior_{Inf.NO}"] == 1]

def filter_experienced(df):
    """ had experience with infection """
    return df[df[f"InfPrior_{Inf.YES}"] == 1]


def filter_vacc_by_dose(df, dose_number, vacc_type):
    """ dose number: 1, 2, 3 
        vacc_type: 'P', 'M', 'A', 'J' 
    """
    smid_names = {
        (1, 'P') : Vacc.P1,
        (2, 'P') : Vacc.P2,
        (3, 'P') : Vacc.P3,
        (1, 'M') : Vacc.M1,
        (2, 'M') : Vacc.M2,
        (3, 'M') : Vacc.M3,
        (1, 'A') : Vacc.A1,
        (2, 'A') : Vacc.A2,        
        (1, 'J') : Vacc.J,
    }

    vacc_name = smid_names[(dose_number, vacc_type)]
    return df[df[f"VaccStatus_{vacc_name}"] == 1]


