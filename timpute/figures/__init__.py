from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO

METHODS = (perform_DO, perform_ALS, perform_CLS)
METHODNAMES = ["DO","ALS","CLS"]
SAVENAMES = ["zohar", "alter", "hms", "coh_response"]
DATANAMES = ['Covid serology', 'HIV serology', 'DyeDrop profiling', 'BC cytokine']
LINESTYLES = ('dashdot', (0,(1,1)), 'solid', (3,(3,1,1,1,1,1)), 'dotted', (0,(5,1)))
DROPS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)