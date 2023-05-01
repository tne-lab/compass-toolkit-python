"""
Developed by Sumedh Nagrale
"""
import json
import wrapp as example_json
import FormatDSForJsonConvertion as FC
import algorithms_details as algo_det

# Example data
[data, algodet] = example_json.wrapp()
'''Example json conversion for communication with applications 
# Json format conversions
data = FC.FormatDSForJsonConvertion(data)
data = json.dumps(data)
data = json.loads(data)

# Json format conversions
algodet = FC.FormatDSForJsonConvertion(algodet)
algodet = json.dumps(algodet)
algodet = json.loads(algodet)
'''

def receiverWrapper(data, algodet):
    try:
        # call the process_dict function with the JSON data
        [CurrentData, algodetails] = algo_det.algorithms_details(data, algodet[0])
        '''
        Need to finalize upon the data that needs to accessed and tranfered to the application
        '''
        # convert back the data and send it to the application
        datart = FC.FormatDSForJsonConvertion(CurrentData)
        currentdet = json.dumps(datart)
        # convert back the data and send it to the application
        algodetails = FC.FormatDSForJsonConvertion(algodetails)
        algodet = json.dumps(algodetails)

        return currentdet, algodet
    except Exception as e:
        print("Error:", e)


# example run
for i in range(0,len(data)):
    print(receiverWrapper(data[i], algodet))
