import pandas as pd 

# For the dataset used by the paper by Kong et al:
#   The households are from the dataset SGSC, the subset used is the households 
#   that possess a hot water system. The households can be identified inside 
#   the file 'sgsc-cthanplug-readings.csv', they are the customers with
#   'Hot Water System' under ' PLUG_NAME' (yes the space is relevant) 
# 
#   The actual load of the households are in the file:
#   'CD_INTERVAL_READING_ALL_NO_QUOTES'
#   relevant fields are 'CUSTOMER_ID', 'READING_DATETIME', ' GENERAL_SUPPLY_KWH', and ' CONTROLLED_LOAD_KWH'
#   The file: CUSTOMER_IDs_Kong_et_al.csv in ../dataset/ contains the list of 
#   the CUSTOMER_ID fields that belong to the 69 households used in the paper. 

customer_ids_path = '../dataset/CUSTOMER_IDs_Kong_et_al.csv'
dataset_path = '../dataset/CD_INTERVAL_READING_ALL_NO_QUOTES.csv'
#dataset_path = '../dataset/test.csv'

training_path = '../dataset/training_data.csv'
validation_path = '../dataset/validation_data.csv'
testing_path = '../dataset/testing_data.csv'

# loads the customer ids and returns it as a list 
def load_customer_ids() :
    customer_ids_csv = pd.read_csv(customer_ids_path)
    customer_ids = [ id[1] for id in customer_ids_csv.values ]
    return customer_ids

def load_dataset_iterator() :
    # needed from CD_INTERVAL_READING_ALL_NO_QUOTES:
    # CUSTOMER_ID
    # READING_DATETIME
    # GENERAL_SUPPLY_KWH
    # CONTROLLED_LOAD_KWH
    
    # first define the columns
    columns = ['CUSTOMER_ID', 'READING_DATETIME', ' GENERAL_SUPPLY_KWH', ' CONTROLLED_LOAD_KWH']
    
    #chunksize is arbitrary, 1000 is good for now
    dataset_iter = pd.read_csv(dataset_path, iterator=True, chunksize=1000, usecols=columns)

    return dataset_iter

def kong_et_al_data() :
    
    dataset_iterator = load_dataset_iterator()

    customer_ids = load_customer_ids()
    #customer_ids.append(10006414) # test value so I actually see some data
 
    # Training data: 01/06/2013 to 05/08/2013
    training_start = '2013-06-01 00:00:00'
    training_end = '2013-08-05 23:59:59'
    training_data = [] 

    # Validation data: 06/08/2013 to 22/08/2013
    validation_start = '2013-06-01 00:00:00'
    validation_end = '2013-08-05 23:59:59'
    validation_data = []

    # Testing data: 23/08/2013 to 31/08/2013
    testing_start = '2013-08-23 00:00:00'
    testing_end = '2013-08-31 23:59:59'
    testing_data = [] 
    
    cnt = 0
    for chunk in dataset_iterator:
        chunk['READING_DATETIME'] = pd.to_datetime(chunk['READING_DATETIME'], format="%Y-%m-%d %H:%M:%S")

        training_chunk = chunk.loc[(chunk['CUSTOMER_ID'].isin(customer_ids))
                 & (chunk['READING_DATETIME'] >= training_start)
                 & (chunk['READING_DATETIME'] <= training_end)]
        training_data.append(training_chunk) 

        validation_chunk =  chunk.loc[(chunk['CUSTOMER_ID'].isin(customer_ids))
                 & (chunk['READING_DATETIME'] >= validation_start)
                 & (chunk['READING_DATETIME'] <= validation_end)]
        validation_data.append(validation_chunk)

        testing_chunk = chunk.loc[(chunk['CUSTOMER_ID'].isin(customer_ids))
                 & (chunk['READING_DATETIME'] >= testing_start)
                 & (chunk['READING_DATETIME'] <= testing_end)]
        testing_data.append(testing_chunk)
        
        cnt = cnt + 1
        if (cnt % 100 == 0) :
            print("processed: ", cnt*100, " lines")


    training_data = pd.concat(training_data)
    validation_data = pd.concat(validation_data)
    testing_data = pd.concat(testing_data)
 
    training_data.to_csv(training_path)
    validation_data.to_csv(validation_path)
    testing_data.to_csv(testing_path)

kong_et_al_data()
