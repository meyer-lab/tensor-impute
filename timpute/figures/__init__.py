from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO

METHODS = (perform_DO, perform_ALS, perform_CLS)
METHODNAMES = ["DO", "ALS-SI", "C-ALS"]
SAVENAMES = ["zohar", "alter", "hms", "coh_response"]
DATANAMES = ["SARS-COV-2 serology", "HIV serology", "DyeDrop profiling", "BC cytokine"]
DROPS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

SUBTITLE_FONTSIZE = 15
TEXT_FONTSIZE = 13
LINE_WIDTH = 2
