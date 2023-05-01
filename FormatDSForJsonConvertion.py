"""
Developed by Sumedh Nagrale
"""
import numpy as np
# Convert python array to list so that it can be converted to json formatted correctly
def FormatDSForJsonConvertion(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: FormatDSForJsonConvertion(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [FormatDSForJsonConvertion(item) for item in obj]
    else:
        return obj