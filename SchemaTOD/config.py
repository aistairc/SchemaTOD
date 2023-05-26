import sys
import logging
import os
import time


class _Config:
    def __init__(self):
        self.root_dir = '../'
        sys.path.append(self.root_dir)

        self._init()

    def _init(self):
        self.model_path = 't5-small'
        self.tok_path = 't5-small'
        self.opt_path = ''

        self.data_path = ''
        self.experiment_path = ''

        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # general settings
        self.mode = 'unknown'
        self.cuda = True
        self.exp_no = ''
        self.seed = 11
        self.save_log = True  # tensorboard
        self.validate_during_training = True
        self.report_interval = 4
        self.save_model_interval = 1
        self.truncated = False
        self.max_context_length = 768

        # training settings
        self.train_dspn = True
        self.train_bspn = True
        self.train_resp = True

        self.use_context = True
        self.include_context_desc = True
        self.include_context_val_cons = True

        self.dspn_coeff = 1.0
        self.bspn_coeff = 1.0
        self.resp_coeff = 1.0

        self.lr = 3e-4
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 4
        self.batch_size = 4

        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.epoch_num = 50
        self.early_stop_count = 5
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.valid_loss = 'score'

        # evaluation settings
        self.inference_path = ''
        self.inference_per_domain = False

        self.result_path = ''
        self.eval_domains = ['all']
        self.eval_dials = ['all']

        self.use_true_prev_dspn = False
        self.use_true_prev_bspn = False
        self.use_true_db_pointer = False
        self.use_true_prev_aspn = False
        self.use_true_prev_resp = False

        self.use_true_curr_dspn = False
        self.use_true_curr_bspn = False
        self.use_all_previous_context = True

        self.exp_domains = ['all']  # domains to be excluded
        self.ft_domains = ['all']  # domains to be fine-tuned
        # self.exp_domains = ['flights']

        self.v_schema = ''
        self.log_path = 'inference-log/'
        self.low_resource = False

        # model settings
        self.bspn_mode = 'bspn'  # 'bspn' or 'bsdx'

        self.same_eval_as_cambridge = True
        self.use_true_dspn_for_ctr_eval = True

        self.do_sample = False
        self.temperature = 0.7
        self.top_k = 0
        self.top_p = 1.0
        self.save_optimizer = False

    def construct_paths(self):
        # data/sgd-processed for original sgd dialogues
        # data/sgd-processed-v{1-5} for extended version of sgd dialogues
        self.processed_data_path = 'data/sgd-processed/'
        self.processed_data_path += '{}/'.format(self.v_schema) if self.v_schema else ''

        # db/ for original sgd dialogues
        # db-v{1-5} for extended version of sgd dialogues
        self.db_path = 'db/'
        self.db_path += '{}/'.format(self.v_schema) if self.v_schema else ''

        self.data_file = self.root_dir + self.processed_data_path + 'data.json'
        self.train_list = self.root_dir + self.processed_data_path + 'train_list.txt'
        self.dev_list = self.root_dir + self.processed_data_path + 'dev_list.txt'
        self.test_list = self.root_dir + self.processed_data_path + 'test_list.txt'

        self.slot_value_set = self.root_dir + self.db_path + 'value_set.json'
        self.dbs = {}
        for f in os.listdir(self.root_dir + self.db_path):
            if not f.endswith('.json'):
                continue
            if 'value_set.json' in f:
                continue
            self.dbs[f.replace('_db.json', '')] = self.root_dir + self.db_path + f

        self.service_set_ids_file = self.root_dir + self.processed_data_path + 'service_sets.json'
        self.service_ids_file = self.root_dir + self.processed_data_path + 'services.json'
        self.service_context_file = self.root_dir + self.processed_data_path + 'service_context.json'
        self.slot_context_file = self.root_dir + self.processed_data_path + 'slot_context.json'
        self.canonical_value_mapping_file = self.root_dir + self.processed_data_path + 'canonical_value_mapping.json'

        self.log_path += '{}/'.format(self.v_schema) if self.v_schema else ''

        if 'all' not in self.exp_domains:
            experiment_folder1 = 'experiments-Xdomain/'
        elif 'all' not in self.ft_domains:
            experiment_folder1 = 'experiments-Fdomain/'
        elif self.v_schema:
            experiment_folder1 = 'experiments/{}/'.format(self.v_schema)
        else:
            experiment_folder1 = 'experiments/'

        if 'all' in self.ft_domains:
            experiment_folder2 = '{}_{}_bs{}_ga{}'.format('-'.join(self.exp_domains),
                                                          self.exp_no, self.batch_size,
                                                          self.gradient_accumulation_steps)
        else:
            experiment_folder2 = '{}_{}_bs{}_ga{}'.format('-'.join(self.ft_domains),
                                                          self.exp_no, self.batch_size,
                                                          self.gradient_accumulation_steps)

        if not self.data_path:
            self.data_path = 'data/'
        if not self.experiment_path:
            self.experiment_path = experiment_folder1 + experiment_folder2

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()

        if not os.path.exists('log'):
            os.mkdir('log')

        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler(
                './log/log_{}_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode,
                                                           '' if 'all' in self.exp_domains else 'x',
                                                           '-'.join(self.exp_domains),
                                                           self.exp_no,
                                                           self.seed))
        elif self.save_log and self.mode == 'finetune':
            file_handler = logging.FileHandler(
                './log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode,
                                                        '-'.join(self.ft_domains),
                                                        self.exp_no,
                                                        self.seed))
        else:
            inference_path = os.path.join(self.inference_path, 'inference_log.json')
            file_handler = logging.FileHandler(inference_path)

        formatter = logging.Formatter('%(asctime)s - %(name)s:%(levelname)s:%(message)s')
        stderr_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logging.basicConfig(handlers=[stderr_handler, file_handler])

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()
