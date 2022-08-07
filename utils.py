## Contains used fucntion
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error, max_error


# Function for creating dummy variable and adding it to the dataframe
def concat_dummy(df:pd.DataFrame,colname:str):
    dummy = pd.get_dummies(df[colname])
    if 'no' in dummy.columns:
        dummy=dummy.rename(columns={'no': 'no_'+colname, 'yes': 'yes_'+colname})
    df=pd.concat([df,dummy],axis=1)
    df=df.drop(columns=colname)

    return(df)


def metrics_table(y_true,y_pred):
    MSE=mean_squared_error(y_true,y_pred)
    MAE=mean_absolute_error(y_true,y_pred)
    Max_error=max_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred)

    d={'Metric': ['MSE','MAE','Max_error','R2'],'Value': [MSE,MAE,Max_error,r2]}

    return pd.DataFrame(data=d)













