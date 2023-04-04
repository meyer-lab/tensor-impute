import xarray as xa
import numpy as np
import tensorly as tl
from timpute.decomposition import Decomposition
from timpute.tensorly_als import perform_ALS
from timpute.common import *
from timpute.plot import q2xentry,q2xchord
import time
from timpute.figures import compare_imputation

np.random.seed(123)

ax, f = getSetup((8,8), (2,2))
rec_tensor = xa.open_dataarray("CoHcomponents/CoH_Rec.nc").to_numpy() # 36 x 23 x 12
resp_tensor = xa.open_dataarray("CoHcomponents/CoH_Tensor_DataSet.nc").to_numpy() # 38 x 6 x 23 x 6

rec = Decomposition(rec_tensor, 24, perform_ALS)
resp = Decomposition(resp_tensor, 24, perform_ALS)


start = time.time()
rec.Q2X_entry(int(np.sum(np.isfinite(rec.data))*0.1),25, dropany=True)
rec.Q2X_chord(int(rec.data.size/rec.data.shape[0]*0.1),25)

print("Receptor Dataset runtime:" + str(time.time()-start)) # expect ~1k seconds
q2xentry(ax[0], rec, detailed=True)
q2xchord(ax[1], rec, detailed=True)

start = time.time()
resp.Q2X_entry(int(np.sum(np.isfinite(resp.data))*0.1),25, dropany=True)
resp.Q2X_chord(int(resp.data.size/resp.data.shape[0]*0.1),25)

print("Response Dataset runtime:" + str(time.time()-start)) # expect 8k seconds
q2xentry(ax[2], resp, detailed=True)
q2xchord(ax[3], resp, detailed=True)

rec.save('./methodruns/CoH_data/receptor_imputations')
resp.save('./methodruns/CoH_data/response_imputations')

f.savefig('./methodruns/CoH_data/CoH_imputation.svg', bbox_inches="tight", format='svg')

compare_imputation(rec_tensor,impute_r=24, impute_reps=20, impute_type='entry', impute_perc=0.1, save="CoH/receptor_imputation")
compare_imputation(resp_tensor,impute_r=24, impute_reps=20, impute_type='entry', impute_perc=0.1, save="CoH/response_imputation")