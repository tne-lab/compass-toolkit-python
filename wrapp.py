"""
Developed by Sumedh Nagrale
"""
def wrapp():
    json_file = [
        {
            "rt": 592,
            "stimulus": "332",
            "response": "2",
            "task": "response",
            "stimulus_id": "1",
            "correct_response": "2",
            "interference_level": "CONFLICT",
            "trial_type": "html-keyboard-response",
            "trial_index": 16,
            "time_elapsed": 18090,
            "internal_node_id": "0.0-15.0-1.0",
            "correct": True
        },
        {
            "rt": 461,
            "stimulus": "100",
            "response": "1",
            "task": "response",
            "stimulus_id": "2",
            "correct_response": "1",
            "interference_level": "NON-CONFLICT",
            "trial_type": "html-keyboard-response",
            "trial_index": 18,
            "time_elapsed": 19805,
            "internal_node_id": "0.0-15.0-1.1",
            "correct": True
        },
        {
            "rt": 635,
            "stimulus": "211",
            "response": "2",
            "task": "response",
            "stimulus_id": "3",
            "correct_response": "2",
            "interference_level": "CONFLICT",
            "trial_type": "html-keyboard-response",
            "trial_index": 20,
            "time_elapsed": 21696,
            "internal_node_id": "0.0-15.0-1.2",
            "correct": True,
        },
        {
            "rt": 648,
            "stimulus": "221",
            "response": "1",
            "task": "response",
            "stimulus_id": "4",
            "correct_response": "1",
            "interference_level": "CONFLICT",
            "trial_type": "html-keyboard-response",
            "trial_index": 22,
            "time_elapsed": 23598,
            "internal_node_id": "0.0-15.0-1.3",
            "correct": True
        },
        {
            "rt": 656,
            "stimulus": "232",
            "response": "3",
            "task": "response",
            "stimulus_id": "5",
            "correct_response": "3",
            "interference_level": "CONFLICT",
            "trial_type": "html-keyboard-response",
            "trial_index": 24,
            "time_elapsed": 25508,
            "internal_node_id": "0.0-15.0-1.4",
            "correct": True
        }
    ]

    '''Required data'''
    algodetails = [{
        "exp_distribution_type": 'greedy',
        "algo_details": {"epsilon": 0.9},
        "patient_details": {"numberArms": 10},
        "recommendation": -1,
        "current_details": {'armselectedCount': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            'meanRTvalue': [0.3, 0.5, 0.6, 0.75, 0.81, 0.68, 0.05, 0.35, 0.56, 0.3], 'trialNumber': 1}
    }]
    return json_file,algodetails
