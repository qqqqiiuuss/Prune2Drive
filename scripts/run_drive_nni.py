search_space = {
    'view_1': {'_type': 'uniform', '_value': [0.1, 1.0]},  
    'view_2': {'_type': 'uniform', '_value': [0.1, 1.0]},  
    'view_3': {'_type': 'uniform', '_value': [0.1, 1.0]},  
    'view_4': {'_type': 'uniform', '_value': [0.1, 1.0]},  
    'view_5': {'_type': 'uniform', '_value': [0.1, 1.0]},  
    'view_6': {'_type': 'uniform', '_value': [0.1, 1.0]},  
}

from nni.experiment import Experiment
experiment = Experiment('local')

experiment.config.trial_command = 'python drive_nni.py --data ../test_llama.json'
experiment.config.trial_code_directory = '.'

experiment.config.experiment_working_directory = './nni_logs'

experiment.config.tuner.name = "TPE"

experiment.config.search_space = search_space
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1

experiment.run(18091)
