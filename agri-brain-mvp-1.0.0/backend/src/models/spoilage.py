import numpy as np, pandas as pd

def compute_spoilage(df, base_temp=4.0, k0=0.0008):
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df=df.copy(); df['timestamp']=pd.to_datetime(df['timestamp'])
    df['dt_h']=(df['timestamp']-df['timestamp'].iloc[0]).dt.total_seconds()/3600.0
    dT=(df['tempC']-base_temp).clip(lower=0)
    rate=k0*(1+0.25*dT+0.04*(dT**2))
    df['shelf_left']=(1-np.cumsum(rate*(df['dt_h'].diff().fillna(0)))).clip(lower=0)
    return df

def volatility_flags(df,window=8,thresh=1.5):
    v=df['tempC'].rolling(window, min_periods=3).std().fillna(0)
    return np.where(v>thresh,'anomaly','normal')
