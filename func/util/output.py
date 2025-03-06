#### ignore warnings
def no_warnings():
    import warnings
    warnings.filterwarnings("ignore")
    
#### ingore all outputs
import os
import sys

def ignore_all_output():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# Call this function to ignore all outputs
#ignore_all_output()

#### ignore output of a code block
# %%capture


