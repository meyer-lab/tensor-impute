from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
from ..method_PM import perform_PM

METHODS = (perform_DO, perform_ALS, perform_PM, perform_CLS)
METHODNAMES = ["DO","ALS-SI","PM","C-ALS"]
SAVENAMES = ["zohar", "alter", "hms", "coh_response"]
DATANAMES = ['Covid serology', 'HIV serology', 'DyeDrop profiling', 'BC cytokine']
DROPS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

SUBTITLE_FONTSIZE = 15
TEXT_FONTSIZE = 13