import numpy as np
import polars as pl

FIELDS=[6,7,8,9,10,11,12,13,14,16,17,19,20,21,22,25]
def learn(s):
    tab=s.drop_nulls().value_counts(name='cnt').sort(['cnt',s.name],descending=[True,False]);return tab.filter(pl.col('cnt')>=20).head(24).with_row_index('code',offset=1)
def encode(s,tab):
    num=s.to_frame().join(tab,on=s.name,how='left',maintain_order='left',validate='m:1')['code'].fill_null(0).to_numpy();return (num[:,None]==np.arange(len(tab)+1)).astype('f4')
