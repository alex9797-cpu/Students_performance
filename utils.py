## Contains used fucntion
import pandas as pd


# Function for creating dummy variable and adding it to the dataframe
def concat_dummy(df:pd.DataFrame,colname:str):
    dummy = pd.get_dummies(df[colname])
    if 'no' in dummy.columns:
        dummy=dummy.rename(columns={'no': 'no_'+colname, 'yes': 'yes_'+colname})
    df=pd.concat([df,dummy],axis=1)
    df=df.drop(columns=colname)

    return(df)








