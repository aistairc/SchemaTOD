import copy
import os
import csv
import random
import logging
import json
import numpy as np

from transformers import T5Tokenizer
from collections import OrderedDict

import utils
from ontology import SGD_Ontology
from db_ops import SGD_DB
from config import global_config as cfg


class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        # group dialogues with the same turn length
        # turn length: [dial, dial]
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug('bucket {} instance {}'.format(k, len(turn_bucket[k])))

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch,
        # otherwise form a new batch
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)

        return all_batches

    def transpose_batch(self, dial_batch):  # batch by turn

        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def inverse_transpose_turn(self, turn_list):
        # eval, one dialog at a time
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key == 'dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_serv = turn['turn_serv'][-1]
                    value = self.db.pointerBack(value, turn_serv)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        # :param turn_batch_list: list of transpose dial batch
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_serv = turn_batch['turn_serv'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_serv)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if cfg.low_resource and set_name in ['test', 'dev']:
            dial = random.sample(dial, 1)
            logging.info('low resource setting, testing size: {}'.format(len(dial)))

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}

        num_turns, num_dials = 0, len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial

    def get_batches(self, set_name):
        log_str = '\n'
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        if cfg.low_resource and set_name == 'train':
            dial = random.sample(dial, int(len(dial) * 0.1))
            logging.info('low resource setting, size: {}'.format(len(dial)))
        elif cfg.low_resource and set_name != 'train':
            dial = random.sample(dial, 1)
            logging.info('low resource setting, testing size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}

        all_batches = []
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name == 'train' and (k == 1 or k >= 17):  # filter dialogues
                continue

            batches = self._construct_mini_batch(turn_bucket[k])
            log_str += '#turn: {} -> '.format(k) + \
                       '#dial: {}, '.format(len(turn_bucket[k])) + \
                       '#batch: {}, '.format(len(batches)) + \
                       'last batch len: {}\n'.format(len(batches[-1]))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        log_str += 'total #batch: {}\n'.format(len(all_batches))
        logging.info(log_str)

        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        if set_name == 'train':
            random.shuffle(all_batches)

        return all_batches

    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def save_result(self, write_mode, results, field, write_title=''):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title + '\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None


class SGD_Reader(_ReaderBase):
    def __init__(self):
        super().__init__()

        self.ontology = SGD_Ontology()
        self.db = SGD_DB(cfg.dbs, self.ontology.db_servs, extractive_slots=self.ontology.extractive_slots)

        self.tokenizer = T5Tokenizer.from_pretrained(cfg.tok_path)

        if cfg.mode == 'train':
            self.add_special_tokens()

        cfg.pad_token_id = self.tokenizer.pad_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id

        cfg.vocab_size = len(self.tokenizer)
        cfg.domain_size = len(self.ontology.all_domains)
        cfg.slot_size = len(self.ontology.all_slots)

        self.service_set_ids = json.loads(open(cfg.service_set_ids_file, 'r').read())

        self.dev_files = {l.strip(): 1 for l in open(cfg.dev_list, 'r').readlines()}
        self.test_files = {l.strip(): 1 for l in open(cfg.test_list, 'r').readlines()}

        self.base_keys = ['dial_contexts', 'slot_contexts',
                          'dspn_lbls', 'bspn_lbls', 'resp_lbls', 'domain_lbls', 'slot_lbls',
                          'slot_turn_servs_flag', 'slot_dial_servs_flag']
        self.base_eval_keys = ['dial_contexts', 'hist', 'domain_lbls']

        self.slots = self.ontology.slots
        self.extractive_slots = self.ontology.extractive_slots

        # exclude domains
        self.except_files = {}
        all_serv_sets_list = list(self.service_set_ids.keys())
        if 'all' not in cfg.exp_domains:
            exp_serv_sets = self.get_serv_sets(cfg.exp_domains, all_serv_sets_list)
            for serv_set in exp_serv_sets:
                dial_id_list = self.service_set_ids[serv_set]
                for dial_id in dial_id_list:
                    self.except_files[dial_id] = 1

        # fine-tune domains
        self.only_files = {}
        if 'all' not in cfg.ft_domains:
            only_serv_sets = self.get_serv_sets(cfg.ft_domains, all_serv_sets_list, single_serv=True)
            for serv_set in only_serv_sets:
                dial_id_list = self.service_set_ids[serv_set]
                for dial_id in dial_id_list:
                    self.only_files[dial_id] = 1

        self._load_data()
        # self._print_data_sample()

    def get_serv_sets(self, domains, all_serv_sets_list, single_serv=False):
        serv_sets = []
        for serv_set in all_serv_sets_list:
            servs = [s for s in serv_set.split('-')]
            doms = [s.split('_')[0] for s in servs]
            if single_serv and len(servs) == 1:
                if doms == domains:
                    serv_sets.append(serv_set)
            else:
                if set(doms).intersection(set(domains)):
                    serv_sets.append(serv_set)
        return serv_sets

    def add_special_tokens(self):
        # add special tokens to encoder and decoder tokenizers
        # serves a similar role of Vocab.construct()
        # make a dict of special tokens

        special_tokens = []

        special_tokens.extend(self.ontology.special_tokens)

        for word in self.ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)

        for word in self.ontology.shared_slots:
            word = '[' + word + ']'
            special_tokens.append(word)

        for word in self.ontology.all_domains:
            word = '[' + word + ']'
            special_tokens.append(word)

        for word in self.ontology.all_slots:
            word = '[' + word + ']'
            special_tokens.append(word)

        # for word in self.ontology.all_slots:
        #     word = '[value_' + word + ']'
        #     special_tokens.append(word)

        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        logging.info('added {} special tokens to tokenizer.'.format(len(special_tokens)))

    def _load_data(self, save_temp=True):
        # load processed data and encode, or load already encoded data
        dial_enc_fn = 'dial_encoded'
        slot_enc_fn = 'slot_encoded'
        slot_enc_fn += '_only_cons' if not cfg.include_context_desc else ''
        slot_enc_fn += '_only_desc' if not cfg.include_context_val_cons else ''

        dial_enc_file = os.path.join(cfg.data_path, dial_enc_fn + '.data.json')
        slot_enc_file = os.path.join(cfg.data_path, slot_enc_fn + '.data.json')

        self.dial_enc_data, self.slot_enc_data = {}, {}
        if save_temp:  # save encoded data
            if os.path.exists(dial_enc_file) and os.path.exists(slot_enc_file):
                logging.info('reading encoded dialogue data from {}'.format(dial_enc_file))
                self.data = json.loads(open(cfg.data_file, 'r', encoding='utf-8').read().lower())
                self.dial_enc_data = json.loads(open(dial_enc_file, 'r', encoding='utf-8').read())
                self.train = self.dial_enc_data['train']
                self.dev = self.dial_enc_data['dev']
                self.test = self.dial_enc_data['test']

                logging.info('reading encoded slot data from {}'.format(slot_enc_file))
                self.slot_enc_data = json.loads(open(slot_enc_file, 'r', encoding='utf-8').read())

        # if not exists, directly read processed data, encode and save
        if not self.dial_enc_data and not self.slot_enc_data:
            self.data = json.loads(open(cfg.data_file, 'r', encoding='utf-8').read().lower())
            self.train, self.dev, self.test = [], [], []
            self.slot_enc_data = {}
            for idx, (dial_id, dial) in enumerate(self.data.items()):
                if self.dev_files.get(dial_id):
                    self.dev.append(self._get_dial_encoded_data(dial_id, dial))
                elif self.test_files.get(dial_id):
                    self.test.append(self._get_dial_encoded_data(dial_id, dial))
                else:
                    self.train.append(self._get_dial_encoded_data(dial_id, dial))

                self.slot_enc_data = self._get_slot_encoded_data(self.slot_enc_data, dial)

            if save_temp:  # save encoded data
                dial_enc_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(dial_enc_data, open(dial_enc_file, 'w'), indent=2)
                json.dump(self.slot_enc_data, open(slot_enc_file, 'w'), indent=2)

        if self.except_files:
            del_indices = []  # delete from train collections
            for idx, sample in enumerate(self.train):
                if self.except_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.train[i]

            del_indices = []  # delete from dev collections
            for idx, sample in enumerate(self.dev):
                if self.except_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.dev[i]

            del_indices = []  # delete from test collections
            for idx, sample in enumerate(self.test):
                if self.except_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.test[i]

            logging.info('{} on all domains except {}'.format(cfg.mode, ','.join(cfg.exp_domains)))

        if self.only_files:
            del_indices = []  # delete from train collection
            for idx, sample in enumerate(self.train):
                if not self.only_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.train[i]
            self.train = self.sample_dial(self.train, 32)
            logging.info('training samples: {}'.format([d[0]['dial_id'] for d in self.train]))

            del_indices = []  # delete from dev collection
            for idx, sample in enumerate(self.dev):
                if not self.only_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.dev[i]
            self.dev = self.sample_dial(self.dev, 16)
            logging.info('dev samples: {}'.format([d[0]['dial_id'] for d in self.dev]))

            del_indices = []  # delete from test collection
            for idx, sample in enumerate(self.test):
                if not self.only_files.get(sample[0]['dial_id']):
                    del_indices.append(idx)
            for i in sorted(del_indices, reverse=True):
                del self.test[i]
            self.test = self.sample_dial(self.test, 16)
            logging.info('test samples: {}'.format([d[0]['dial_id'] for d in self.test]))

            logging.info('{} on only {}'.format(cfg.mode, cfg.ft_domains))
        else:
            logging.info('{} on all domains'.format(cfg.mode))

        self.domain_enc_data = self._get_domain_encoded_data(self.ontology.all_domains)

        random.shuffle(self.train)
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def sample_dial(self, dial_set, num_sample):
        sample_set = []
        for d in dial_set:
            if len(d) == 10:
                sample_set.append(d)
            if len(sample_set) == num_sample:
                break
        return sample_set

    def _print_data_sample(self, set_name='train'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        print('sample of dialogue convo:')
        for dial in random.sample(name_to_set[set_name], 1):
            for idx, turn in enumerate(dial[:2]):
                print('  turn #{}'.format(idx))
                print('    user', self.decode(turn['user']))
                print('    usdx', self.decode(turn['usdx']))
                print('    dspn', self.decode(turn['dspn']))
                print('    bspn', ''.join([self.decode(bspn) for bspn in turn['bspn']]))
                print('    bsdx', ''.join([self.decode(bsdx) for bsdx in turn['bsdx']]))
                print('    db', self.decode(turn['db']))
                print('    aspn', self.decode(turn['aspn']))
                print('    resp', self.decode(turn['resp']))
                print()

        for serv in random.sample(list(self.slot_enc_data), 1):
            slot_dat = self.slot_enc_data[serv]
            print('sample of slot context of "{}":'.format(serv))
            print(' ', ', '.join([self.decode(ct) for ct in slot_dat['slot_ct']]))

    def encode(self, txt):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(txt))

    def decode(self, txt):
        return self.tokenizer.decode(txt)

    def batch_decode(self, txt):
        return self.tokenizer.batch_decode(txt)

    def _get_dial_encoded_data(self, fn, dial):
        encoded = []
        d_servs = []  # all services of the dialogue
        offered = {}  # accumulated offered slots by turn service
        for idx, t in enumerate(dial['log']):
            t_servs = t['turn_serv'].split()  # turn services
            t_serv = t_servs[-1]
            t_doms = [serv.split('_')[0] + ']' for serv in t['turn_serv'].split()]  # turn domains
            t_dom = t_doms[-1]
            d_servs = d_servs + [serv for serv in t_servs if serv not in d_servs]  # preserve the order

            user = '<sos_u> ' + t['user'] + ' <eos_u>'
            usdx = '<sos_u> ' + t['user_delex'] + ' <eos_u>'
            resp = '<sos_r> ' + t['resp'] + ' <eos_r>'

            enc_dat = {'dial_id': fn,
                       'user': self.encode(user),
                       'usdx': self.encode(usdx),
                       'resp': self.encode(resp),
                       'bspn': [],
                       'bsdx': [],
                       'aspn': self.encode('<sos_a> ' + t['sys_act'] + ' <eos_a>'),
                       'dspn': self.encode('<sos_d> ' + ' '.join(t_doms) + ' <eos_d>'),
                       'pointer': [int(i) for i in t['pointer'].split(',')],
                       'turn_serv': t_servs,
                       'turn_num': t['turn_num']}

            constraint, cons_delex = t['constraint'].split('||'), t['cons_delex'].split('||')

            # track the offered slots and slot values by the system
            if not offered.get(t_serv):
                offered[t_serv] = {}

            cons = self.bspn_to_constraint_dict(' '.join(constraint))
            offered[t_serv] = self.update_system_offer(cons[t_dom[1:-1]] if cons.get(t_dom[1:-1]) else {},
                                                       offered[t_serv],
                                                       t['sys_act'])
            if cons.get(t_dom[1:-1]):
                for slot in offered[t_serv]:
                    if cons[t_dom[1:-1]].get(slot[1:-1]) and cons[t_dom[1:-1]].get(slot[1:-1]) != 'dontcare':
                        offered[t_serv][slot] = cons[t_dom[1:-1]].get(slot[1:-1])

            enc_dat['offer'] = copy.deepcopy(offered)

            constraint_list = []
            for cons, cons_d in zip(constraint, cons_delex):
                if cons and cons_d:
                    constraint_list.append((cons.split()[0], cons, cons_d))

            for serv in d_servs:
                dom = serv.split('_')[0] + ']'  # [hotels_3] -> [hotels]
                if constraint_list:
                    if constraint_list[0][0] == dom:
                        enc_dat['bspn'].append(self.encode('<sos_b> ' + constraint_list[0][1].strip() + ' <eos_b>'))
                        enc_dat['bsdx'].append(self.encode('<sos_b> ' + constraint_list[0][2].strip() + ' <eos_b>'))
                        del constraint_list[0]
                    else:
                        enc_dat['bspn'].append(self.encode('<sos_b> <eos_b>'))
                        enc_dat['bsdx'].append(self.encode('<sos_b> <eos_b>'))
                else:
                    enc_dat['bspn'].append(self.encode('<sos_b> <eos_b>'))
                    enc_dat['bsdx'].append(self.encode('<sos_b> <eos_b>'))

            # add db results to enc, at every turn
            db_pointer = self.bspn_to_DBpointer(' '.join(constraint), t['turn_serv'].split())
            enc_dat['db'] = self.encode('<sos_db> ' + db_pointer + ' <eos_db>')

            encoded.append(enc_dat)

        return encoded

    def _get_domain_encoded_data(self, domains):
        encoded = {'lbls': []}
        for domain in domains:
            encoded['lbls'].extend(self.encode('[{}]'.format(domain)))

        return encoded

    def _get_slot_encoded_data(self, encoded, dial):
        for idx, t in enumerate(dial['log']):
            for serv in t['turn_serv'].split():

                if serv in encoded:
                    continue

                lbls, cts = [], []
                for slot, context in self.ontology.slots[serv[1:-1]].items():
                    lbls.extend(self.encode('[{}]'.format(slot)))
                    ct_txt = '[' + slot + ']'
                    ct_txt += ' [desc] ' + context['description'] if cfg.include_context_desc else ''
                    if cfg.include_context_val_cons:
                        ct_txt += ' [cons] '
                        ct_txt += ', '.join(context['possible_values']) if context['possible_values'] else '-'
                    cts.append(self.encode('<sos_c> ' + ct_txt + ' <eos_c>'))

                sorted_idx = np.argsort(lbls)  # get sorted indexes of labels in ascending order
                encoded[serv] = {'ct': [cts[i] for i in sorted_idx],  # sort context by sorted indexes of labels
                                 'lbls': [lbls[i] for i in sorted_idx]}  # sort labels

        return encoded

    def process_batches(self, all_batches):
        all_batch_inputs = []
        for dial_batch in all_batches:
            if not dial_batch:
                continue

            batch_inputs = {k: [] for k in self.base_keys}

            for dial in dial_batch:
                dial_inputs = self.convert_session(dial)
                for k in dial_inputs:
                    batch_inputs[k].append(dial_inputs[k])

            for k in batch_inputs:  # turn first
                batch_inputs[k] = self.transpose_batch(batch_inputs[k])

            all_batch_inputs.append(batch_inputs)
        return all_batch_inputs

    def get_data_iterator(self, all_batches):
        for dial_batch in all_batches:
            n_turn = len(dial_batch['dial_contexts'])
            for t_i in range(n_turn):
                inputs = {'dspn_len': [len(i) for i in dial_batch['slot_lbls'][t_i]]}
                inputs = utils.to_tensor(inputs)
                for k in dial_batch:
                    batch = utils.flatten_list(dial_batch[k][t_i]) if 'bspn' in k else dial_batch[k][t_i]
                    inputs[k] = utils.pad(batch, cfg.pad_token_id, trunc_len=cfg.max_context_length)
                    inputs[k] = inputs[k].long()
                yield inputs

    def convert_session(self, dial):
        # convert the whole dialogue session for training
        # concat [U_0, D_0, B_0, DB_0, A_0, R_0, ... , U_t, D_t, B_t, DB_t, A_t, R_t]
        # [user, dspn, bspn, db, aspn, resp]

        max_cxt_len = cfg.max_context_length

        inputs = {k: [] for k in self.base_keys}
        hist_context_list = ['user', 'dspn', 'bspn', 'db', 'aspn', 'resp']
        hist, dial_servs = [], []

        for turn in dial:
            inputs['dial_contexts'].append((hist + turn['user'])[-cfg.max_context_length:] + [cfg.eos_token_id])
            inputs['dspn_lbls'].append(turn['dspn'] + [cfg.eos_token_id])
            inputs['bspn_lbls'].append([lb + [cfg.eos_token_id] for lb in turn['bspn']])
            inputs['resp_lbls'].append(turn['db'] + turn['aspn'] + turn['resp'] + [cfg.eos_token_id])

            turn_serv = turn['turn_serv'][-1]  # IMPORTANT: only use the last service of the turn

            if turn_serv not in dial_servs:
                dial_servs.append(turn_serv)

            inputs['domain_lbls'].append(self.domain_enc_data['lbls'])

            # slot context and labels for a dialogue (multi-domain, for belief state)
            slot_contexts, slot_lbls, slot_turn_serv, slot_dial_serv = [], [], [], []
            for serv in dial_servs:
                slot_contexts.append([ct + [cfg.eos_token_id] for ct in self.slot_enc_data[serv]['ct']])
                slot_lbls.append(self.slot_enc_data[serv]['lbls'])
                slot_turn_serv += [1] if serv == turn_serv else [0]
                slot_dial_serv += [1]
            inputs['slot_contexts'].append(slot_contexts)
            inputs['slot_lbls'].append(slot_lbls)
            inputs['slot_turn_servs_flag'].append(slot_turn_serv)
            inputs['slot_dial_servs_flag'].append(slot_dial_serv)

            # update history
            for c in hist_context_list:
                if c == 'bspn':
                    hist += self.flatten_bspn(turn[c])
                else:
                    hist += turn[c]

        return inputs

    def flatten_bspn(self, bspns):
        flat_bspn = []
        sos_b_id = self.encode('<sos_b>')[0]
        eos_b_id = self.encode('<eos_b>')[0]
        for bspn in bspns:
            flat_bspn.extend(bspn[1:-1])
        return [sos_b_id] + flat_bspn + [eos_b_id]

    def process_batch_turn_eval(self, turn_idx, dial_batch, batch_pv_turns):
        batch_turns = []
        batch_inputs = {k: [] for k in self.base_eval_keys}
        first_turn = (turn_idx == 0)
        for dial, pv_turns in zip(dial_batch, batch_pv_turns):
            turn = dial[turn_idx]
            inputs = self.convert_turn_eval(turn, pv_turns, first_turn)
            for k in inputs:
                batch_inputs[k].append(inputs[k])
            batch_turns.append(turn)
        return batch_turns, batch_inputs

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        # use true dspn -> input: [H_t (all prev), U_t] predict D_t, B_t, DB_t, A_t
        # first turn: [U_t, D_t, B_t, DB_t, A_t] predict R_t

        max_cxt_len = cfg.max_context_length

        context = turn['user']

        inputs = {}
        if first_turn:
            inputs['dial_contexts'] = context
            inputs['hist'] = context
        else:
            pv_context = pv_turn['hist'] + \
                         pv_turn['dspn'] + \
                         self.flatten_bspn(pv_turn['bspn']) + \
                         pv_turn['db'] + \
                         pv_turn['aspn'] + \
                         pv_turn['resp']
            inputs['dial_contexts'] = pv_context + context
            inputs['hist'] = pv_context + context if cfg.use_all_previous_context else context

        inputs['dial_contexts'] = inputs['dial_contexts'][-max_cxt_len:] + [cfg.eos_token_id]

        inputs['domain_lbls'] = self.domain_enc_data['lbls']

        return inputs

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []

        eos_syntax = self.ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = self.ontology.sos_tokens

        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user',
                 'bspn_gen', 'bspn', 'resp_gen', 'resp',
                 'aspn_gen', 'aspn', 'dspn_gen', 'dspn',
                 'db_gen', 'db', 'pointer']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for f in field[2:]:
                entry[f] = ''  # ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue

                    v = turn.get(key, '')

                    if key == 'turn_serv':
                        v = ' '.join(v)
                    elif key in ['bspn', 'bsdx', 'bspn_gen']:
                        v = self.flatten_bspn(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        for eos in eos_syntax[key]:
                            while True:
                                if eos in v:
                                    v.remove(eos)
                                else:
                                    break
                        for sos in sos_syntax[key]:
                            while True:
                                if sos in v:
                                    v.remove(sos)
                                else:
                                    break
                        v = ' '.join(v)
                    else:
                        pass  # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

    def dspn_to_slots(self, serv):
        if serv[1:-1] not in self.ontology.all_servs:
            return {'slot_doms': [cfg.pad_token_id],
                    'slot_contexts': [[cfg.pad_token_id]],
                    'slot_lbls': [cfg.pad_token_id],
                    'lm_mask_lbls': [cfg.pad_token_id]}

        dom = serv.split('_')[0] + ']'

        slots = {'slot_doms': self.encode(dom),
                 'slot_contexts': self.slot_enc_data[serv]['ct'],
                 'slot_lbls': self.slot_enc_data[serv]['lbls']}
        slots['lm_mask_lbls'] = slots['slot_doms'] + slots['slot_lbls']

        return slots

    def update_system_offer(self, dom_con, serv_offered, sys_act):
        is_offer = False
        for tkn in sys_act.split():
            if tkn[1:-1] in self.ontology.all_acts:
                is_offer = True if tkn == '[offer]' else False
                continue
            if is_offer and tkn[1:-1] in self.ontology.all_slots:
                if not dom_con:
                    serv_offered[tkn] = ''
                elif not dom_con.get(tkn[1:-1]) or dom_con.get(tkn[1:-1]) == 'dontcare':
                    serv_offered[tkn] = ''

        return serv_offered

    def constraint_dict_to_bspn(self, constraint_dict):
        bspn = []
        for dom in constraint_dict:
            bspn += ['[' + dom + ']']
            for slot in constraint_dict[dom]:
                value = constraint_dict[dom][slot]
                # map to canonical value if available
                if self.ontology.canonical_value_mapping.get(value):
                    value = self.ontology.canonical_value_mapping[value]
                bspn += ['[' + slot + ']', value]
        return ' '.join(bspn)

    def bspn_to_constraint_dict(self, bspn, bspn_mode='bspn'):
        constraint_dict = {}
        if not bspn:
            return constraint_dict

        if isinstance(bspn[0], list):
            bspn = ' '.join(self.batch_decode(bspn))
        elif isinstance(bspn[0], int):
            bspn = self.decode(bspn)

        bspn = bspn.split() if not isinstance(bspn, list) else bspn

        sos_b_token = '<sos_b>'
        if sos_b_token in bspn:
            sos_b_token_idx = bspn.index(sos_b_token)
            bspn = bspn[sos_b_token_idx + 1:]

        eos_b_token = '<eos_b>'
        if eos_b_token in bspn:
            eos_b_token_idx = bspn.index(eos_b_token)
            bspn = bspn[:eos_b_token_idx]

        dom = None
        conslen = len(bspn)
        for idx, cons in enumerate(bspn):

            if cons[1:-1] in self.ontology.all_domains:
                dom = cons[1:-1]
                continue

            if cons[1:-1] in self.ontology.get_slot:
                if dom is None:
                    continue
                if not constraint_dict.get(dom):
                    constraint_dict[dom] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[dom][cons[1:-1]] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspn[vidx]
                while vidx < conslen and '[' not in vt and vt[1:-1] not in self.ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspn[vidx]

                if vt_collect:
                    cons_val = ' '.join(vt_collect)

                    # map to canonical value if available
                    if self.ontology.canonical_value_mapping.get(cons_val):
                        cons_val = self.ontology.canonical_value_mapping[cons_val]

                    if not constraint_dict[dom].get(cons[1:-1]):
                        constraint_dict[dom][cons[1:-1]] = cons_val

        return constraint_dict

    def bspn_to_DBpointer(self, bspn, turn_serv):
        constraint_dict = self.bspn_to_constraint_dict(bspn)
        match_serv = turn_serv[-1] if isinstance(turn_serv, list) else turn_serv
        matnum = self.db.get_match_num(constraint_dict, match_serv)
        vector = self.db.addDBIndicator(match_serv, matnum)
        return vector
