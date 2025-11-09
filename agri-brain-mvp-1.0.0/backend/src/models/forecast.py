import numpy as np

def yield_demand_forecast(df, horizon=24):
    d=df['demand_units'].astype(float).to_numpy()
    if len(d)==0: return [0.0]*horizon
    ema=0.6; s=d[-1]
    for x in d[-min(48,len(d)):]: s=ema*x+(1-ema)*s
    trend=(np.gradient(d[-min(48,len(d)):]).mean() if len(d)>1 else 0.0)
    return [max(0.0, s+i*0.2*trend) for i in range(1,horizon+1)]
