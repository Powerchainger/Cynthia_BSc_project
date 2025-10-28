import pandas as pd 

# select households that possess a hot water system
# this is defined as HAS_GAS_HOT_WATER in SGSC
def kong_et_al_data() :
    # Kong et al uses the SGSC dataset, only the households that contain a hot water system
    # use an interator because this csv is extremely huge
    iter_csv = pd.read_csv("../dataset/sgsc-cthanplug-readings.csv", iterator=True, chunksize=1000)
    households_used = pd.concat([chunk[(chunk[' PLUG_NAME'] == 'Hot Water System')] for chunk in iter_csv])
    print(households_used['CUSTOMER_ID'].nunique())

kong_et_al_data()    