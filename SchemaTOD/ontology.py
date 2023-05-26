import copy
import json

from config import global_config as cfg


class SGD_Ontology:
    def __init__(self):

        self.all_servs = sorted(list(json.loads(open(cfg.service_ids_file, 'r').read().lower()).keys()))
        self.all_domains = sorted(list(set([serv.split('_')[0] for serv in self.all_servs])))

        self.domain_descs = {}  # domain: desc
        for serv, desc in json.loads(open(cfg.service_context_file, 'r').read().lower()).items():
            domain = serv.split('_')[0]
            if domain not in self.domain_descs:
                self.domain_descs[domain] = desc
            else:
                self.domain_descs[domain] += ', ' + desc

        # a list of domains with no primary key for querying (not require db)
        self.no_db_domains = ['alarm', 'calendar', 'banks', 'messaging', 'payment', 'ridesharing']

        # a list of domains/services that requires db
        self.db_domains = sorted(list(set(self.all_domains) - set(self.no_db_domains)))
        self.db_servs = sorted([serv for serv in self.all_servs if serv.split('_')[0] in self.db_domains])

        self.slots = {}  # service: slot: desc
        self.extractive_slots = {}  # service: [extractive slots]
        for serv, slot_contexts in json.loads(open(cfg.slot_context_file, 'r').read().lower()).items():
            self.slots[serv], self.extractive_slots[serv] = {}, []
            for slot, context in slot_contexts.items():
                self.slots[serv][slot] = {'description': context['description'],
                                          'is_categorical': True if context['is_categorical'] else False,
                                          'possible_values': context['possible_values']}
                if not context['is_categorical']:
                    self.extractive_slots[serv].append(slot)

        self.all_slots = []
        for serv, slot_desc in self.slots.items():
            for slot in slot_desc:
                self.all_slots.append(slot)
        self.all_slots = sorted(list(set(self.all_slots)))

        self.get_slot = {}
        for s in self.all_slots:
            self.get_slot[s] = 1

        self.all_reqslot = copy.deepcopy(self.all_slots)

        self.all_acts = ['inform', 'request', 'confirm', 'offer', 'offer_intent',
                         'notify_success', 'notify_failure', 'req_more', 'goodbye']
        self.shared_slots = ['count', 'serv_intent']
        self.dialog_acts = {serv: self.all_acts for serv in self.all_servs}
        self.dialog_act_all_slots = self.shared_slots + copy.deepcopy(self.all_slots)

        # sos: start of sentence, eos: end of sentence
        # db_nores: no response, db_#: based on number of responses db
        self.db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

        # sos: start of sentence, eos: end of sentence
        self.dialog_tokens = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>',
                              '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<eos_d>']

        self.context_tokens = ['<sos_c>', '[name]', '[desc]', '[cons]', '<eos_c>']

        self.special_tokens = self.dialog_tokens + self.db_tokens + self.context_tokens

        self.eos_tokens = {
            'user': ['<eos_u>'], 'user_delex': ['<eos_u>'],
            'resp': ['<eos_r>'], 'resp_gen': ['<eos_r>'], 'pv_resp': ['<eos_r>'],
            'Bspn': ['<eos_b>'],
            'bspn': ['<eos_b>'], 'bspn_gen': ['<eos_b>'], 'pv_bspn': ['<eos_b>'],
            'Bsdx': ['<eos_b>'],
            'bsdx': ['<eos_b>'], 'bsdx_gen': ['<eos_b>'], 'pv_bsdx': ['<eos_b>'],
            'db': ['<eos_db>'], 'db_gen': ['<eos_db>'],
            'aspn': ['<eos_a>'], 'aspn_gen': ['<eos_a>'], 'pv_aspn': ['<eos_a>'],
            'dspn': ['<eos_d>'], 'dspn_gen': ['<eos_d>'], 'pv_dspn': ['<eos_d>'],
            'exct': ['<eos_c>']  # external context
        }

        self.sos_tokens = {
            'user': ['<sos_u>'], 'user_delex': ['<sos_u>'],
            'resp': ['<sos_r>'], 'resp_gen': ['<sos_r>'], 'pv_resp': ['<sos_r>'],
            'Bspn': ['<sos_b>'],
            'bspn': ['<sos_b>'], 'bspn_gen': ['<sos_b>'], 'pv_bspn': ['<sos_b>'],
            'Bsdx': ['<sos_b>'],
            'bsdx': ['<sos_b>'], 'bsdx_gen': ['<sos_b>'], 'pv_bsdx': ['<sos_b>'],
            'db': ['<sos_db>'], 'db_gen': ['<sos_db>'],
            'aspn': ['<sos_a>'], 'aspn_gen': ['<sos_a>'], 'pv_aspn': ['<sos_a>'],
            'dspn': ['<sos_d>'], 'dspn_gen': ['<sos_d>'], 'pv_dspn': ['<sos_d>'],
            'exct': ['<sos_c>']  # external context
        }

        self.canonical_value_mapping = json.loads(open(cfg.canonical_value_mapping_file, 'r').read().lower())
