import numpy as np
import pandas as pd

def climatology(ts, period=365):
    ts = np.array(ts)
    return pd.Series(ts).groupby(np.arange(len(ts)) % period).mean()
