class Args:
    def __init__(self):
        self.epochs = 20
        self.lr = 0.01
        self.random_state = 42
        self.test_set_size = 0.05
        self.batch_size = 32
        self.per_exponent = 2
        self.output_folder = 'script_outputs/'
        self.output_file_suffix = '-output.txt'
        self.metrics_file_suffix = '-metrics.npy'
        
        self.nsl_columns = 33
        self.isot_columns = 85
        self.mqtt_columns = 46
        
        self.agent_data_splits = 10
        self.num_clients = 10
        self.fparam_k = 30
        self.fparam_a = 50
        self.param_gamma = 0.6
        self.param_tau = 0.8
        
# =============================================================================
#         Training Specifications
# =============================================================================
#         self.dataset = 'isot'
        self.dataset = 'nsl'
        self.data_split_type = 'random'
#         self.data_split_type = 'customized'
# =============================================================================
        
        if(self.dataset == 'nsl'):
            self.n_columns = self.nsl_columns            
        elif self.dataset=='isot':
            self.n_columns = self.isot_columns
        elif self.dataset=='mqtt':
            self.n_columns = self.mqtt_columns
        else: self.n_columns = 500
            
        
        if(self.data_split_type == 'customized'):
            if(self.dataset == 'nsl'):
                self.num_clients = 2
                self.fparam_k = 50000
                self.fparam_a = 200
            else:
                self.num_clients = 5
        
        
args = Args()

# Exports: args