from .FLCG import FacilityLocationConditionalGain
from .GCCG import GraphCutConditionalGain
from .LogDetCG import LogDetConditionalGain

def get_selection_function(function_name):
    if "FLCG" in function_name:
        return FacilityLocationConditionalGain
    elif "GCCG" in function_name:
        return GraphCutConditionalGain
    elif "LDCG" in function_name:
        return LogDetConditionalGain
    else:
        raise ValueError("Such a selection function does not exist.")
