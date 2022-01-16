import numpy as np

def check_float(n):
    try:
        float(n)
        return True
    except ValueError:
        return False