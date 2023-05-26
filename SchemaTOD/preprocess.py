import argparse
import os
import json
import itertools
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm

SETS = ['train', 'dev', 'test']
MAPPING = {'train': '1', 'dev': '2', 'test': '3'}
SCHEMA_FN = 'schema.json'

NO_DB_DOMAINS = ['alarm', 'banks', 'calendar', 'messaging', 'payment', 'ridesharing']


def get_unique_dial_id(set_name, dial_id):
    dial_set_num, dial_id_num = dial_id.split('_')
    dial_set_num = '0' * (3 - len(dial_set_num)) + dial_set_num
    return '{}{}{}'.format(MAPPING[set_name], dial_set_num, dial_id_num)


def read_json(fn):
    dat = json.loads(open(fn, 'r').read().lower())
    df = pd.DataFrame(dat)
    set_name, json_fn = fn.split('/')[-2:]
    df['set'] = set_name
    df['from_origin'] = '{}/{}'.format(set_name, json_fn)  # e.g. train/dialogue_001.json
    if 'dialogue_id' in df.columns:
        df['unique_dialogue_id'] = df['dialogue_id'].apply(lambda x: get_unique_dial_id(set_name, x))
    # print('{}/{}'.format(set_name, json_fn), ', '.join(df['services'].tolist()[0]))
    return df


def read_dialogue(root):
    # read and store all dialogues into dataframe
    files = []
    for set_name in SETS:
        set_files = os.listdir(os.path.join(root, set_name))
        set_files.remove(SCHEMA_FN)
        files.extend([os.path.join(root, set_name, f) for f in sorted(set_files)])

    with mp.Pool(processes=mp.cpu_count()) as pool:
        df_list = tqdm(pool.imap(read_json, files), total=len(files))
        combined_df = pd.concat(df_list, ignore_index=True)
    dial_df = combined_df.reset_index(drop=True)
    print('  number of dialogues: {}'.format(len(dial_df)))

    return dial_df


def read_schema(root):
    set_schema = []
    for set_name in SETS:
        df = read_json(os.path.join(root, set_name, SCHEMA_FN))
        set_schema.append(df)
    schema_df = pd.concat(set_schema)
    schema_df = schema_df.drop_duplicates(subset=['service_name'])
    schema_df = schema_df.reset_index(drop=True)
    return schema_df


def clean_result(res):
    clean_res = {}
    for k, v in res.items():
        v = str(v).replace('"', '')
        clean_res[k] = v
    return clean_res


def get_dbs(db_path, dial_df):
    # aggregate service results of the search intents from dialogues of each service
    # aggregate value sets of each service
    db_records = {}
    for idx, row in tqdm(dial_df.iterrows(), total=len(dial_df)):
        for turn in row['turns']:
            if turn['speaker'] == 'user':
                continue

            for frame in turn['frames']:
                if 'service_results' not in frame.keys():
                    continue

                serv_res = frame['service_results']
                if not serv_res:
                    continue

                # check if it is a failure turn
                # if yes, then skip the service results
                is_failure = False
                serv_act = frame['actions']
                if serv_act:
                    for act in serv_act:
                        if act['act'] == 'notify_failure':
                            is_failure = True
                if is_failure:
                    continue

                # check if it is service result for a domain that doesn't need a db
                serv = frame['service']
                dom = serv.split('_')[0]
                if dom in NO_DB_DOMAINS:
                    continue

                if serv not in db_records.keys():
                    db_records[serv] = []

                serv_res = [clean_result(res) for res in serv_res]
                db_records[serv].extend(serv_res)

    value_sets = {}
    for serv, records in db_records.items():
        serv_df = pd.DataFrame.from_records(records)

        # remove duplicate record
        serv_df = serv_df.drop_duplicates()

        # remove duplicate subset of record
        remove_indexes = []
        for idx, row in serv_df.iterrows():
            nan_bools = row.isnull()

            if not any(nan_bools):
                continue

            q = ['{}=="{}"'.format(k[0], k[1]) for nan_b, k in zip(nan_bools, row.items()) if not nan_b]
            found_idx = serv_df.query('&'.join(q)).index.tolist()
            if len(found_idx) > 1:
                remove_indexes.append(idx)
        serv_df = serv_df.drop(remove_indexes)

        # replace nan with value from other record by "name" slot
        # no replacement for "date", "time", "number", "size" slots
        # no replacement if more than one "name" slot found in the record
        new_records = []
        for idx, row in serv_df.iterrows():
            nan_bools = row.isnull()
            if not any(nan_bools):
                new_records.append(row.to_dict())
                continue

            nan_slots = []
            name_slot, name_slot_val = '', ''
            for nan_b, k in zip(nan_bools, row.items()):
                if nan_b and not k[0].endswith('date') and \
                        not k[0].endswith('time') and \
                        not k[0].startswith('number') and \
                        not k[0].endswith('size'):
                    nan_slots.append(k[0])

                if k[0].endswith("name") and not name_slot:
                    name_slot, name_slot_val = k[0], k[1]
                elif k[0].endswith("name") and name_slot:
                    name_slot, name_slot_val = '', ''

            if not nan_slots or not name_slot:
                new_records.append(row.to_dict())
                continue
            else:
                replaced = False
                q = '{}=="{}"'.format(name_slot, name_slot_val)
                for idy, match_row in serv_df.query(q).iterrows():
                    values = {k: v for k, v in match_row.items() if k in nan_slots and not pd.isna(v)}
                    if len(values) == len(nan_slots):
                        new_row = {k: v if k not in nan_slots else values[k] for k, v in row.items()}
                        new_records.append(new_row)
                        replaced = True
                        break
                if not replaced:
                    new_records.append(row.to_dict())

        serv_df = pd.DataFrame.from_records(new_records)

        # val set
        value_sets[serv] = {}
        for col in serv_df.columns:
            vals = serv_df[col].fillna('').tolist()
            value_sets[serv][col] = [v for v in list(set(vals)) if v]

        # db
        serv_df = serv_df.fillna('?')
        serv_df['id'] = [i for i in range(len(serv_df))]
        serv_records = serv_df.to_json(orient='records')
        parsed = json.loads(serv_records)
        with open(db_path + '{}_db.json'.format(serv), 'w') as outf:
            json.dump(parsed, outf, indent=2)
        print('  number of records in {} db: {}'.format(serv, len(serv_df)))

    with open(db_path + 'value_set.json', 'w') as outf:
        json.dump(value_sets, outf, indent=2)


def process_dialogue_record(dat):
    record, DB = dat

    dial_servs = record['services']
    dial_serv_dom = {serv: serv.split('_')[0] for serv in dial_servs}
    format_rec = {'goal': {}, 'log': []}

    turns = record['turns']
    turn_pairs = [(turns[i], turns[i + 1]) for i in range(len(turns)) if i % 2 == 0]

    dial_canonical_vals = {}
    for i, (user_t, system_t) in enumerate(turn_pairs):
        actions = [(frame['service'], frame['actions']) for frame in user_t['frames']] + \
                  [(frame['service'], frame['actions']) for frame in system_t['frames']]
        actions = [(serv, a) for serv, acts in actions for a in acts]
        for serv, act in actions:
            surface_form_vals = act['values']
            canonical_vals = act['canonical_values']
            if canonical_vals:
                for s_v, c_v in zip(surface_form_vals, canonical_vals):
                    if s_v != c_v:
                        dial_canonical_vals[s_v] = c_v

    dial_constraint = {}
    dial_goals = {}
    char_comb = [''.join(comb) for comb in itertools.product('xyz', repeat=5)]
    for i, (user_t, system_t) in enumerate(turn_pairs):
        # user: user's utterance
        user_utt = user_t['utterance']

        # user_delex: user utterance with slot value replacing by [*slot name*]
        user_delex = user_utt
        for frame in user_t['frames']:
            serv = frame['service']
            for slot in frame['slots']:
                slot_name = slot['slot']
                start = slot['start']
                end = slot['exclusive_end']
                slot_val = user_utt[start:end]
                delex = '[{}]'.format(slot_name)
                user_delex = user_delex.replace(slot_val, delex)

        # resp: system response with slot value replacing by [*slot name*]
        system_resp = system_t['utterance']
        delex_slots = {}
        for frame in system_t['frames']:
            serv = frame['service']
            for slot in frame['slots']:
                slot_name = slot['slot']
                slot_val = system_resp[slot['start']:slot['exclusive_end']]
                delex = '[{}]'.format(slot_name)
                delex_slots[slot_val] = (delex, char_comb.pop())
            for act in frame['actions']:
                slot_name = act['slot']
                slot_val = act['values'][0] if act.get('values') else ''
                if not slot_name or not slot_val:
                    continue
                delex = '[{}]'.format(slot_name)
                delex_slots[slot_val] = (delex, char_comb.pop())

        for slot_val, (delex, c_comb) in delex_slots.items():
            if ' ' + slot_val + ' ' in system_resp:
                system_resp = system_resp.replace(' ' + slot_val + ' ', c_comb)
                delex_slots[slot_val] = (' ' + delex + ' ', c_comb)
            elif slot_val + ' ' in system_resp:
                system_resp = system_resp.replace(slot_val + ' ', c_comb)
                delex_slots[slot_val] = (delex + ' ', c_comb)
            elif ' ' + slot_val in system_resp:
                system_resp = system_resp.replace(' ' + slot_val, c_comb)
                delex_slots[slot_val] = (' ' + delex, c_comb)
            else:
                system_resp = system_resp.replace(slot_val, c_comb)
        for slot_val, (delex, c_comb) in delex_slots.items():
            system_resp = system_resp.replace(c_comb, delex)

        # turn_serv: ([service] ..)
        # (1) can be more than one service in user's turn
        # (2) only one service for the system's turn
        turn_serv = [frame['service'] for frame in user_t['frames']]
        for frame in system_t['frames']:
            if frame['service'] in turn_serv:  # keep the system's turn service at the last index
                turn_serv.remove(frame['service'])
            turn_serv.append(frame['service'])

        # goal: a set of informable and requested slots of each service
        # constraint: belief state as ([*service*] (*slot name* *slot value* ..) ..)
        # cons_delex: belief state act as ([*service*] (*slot name* ..) ..)
        # (1) accumulate belief state (constraints of what the user has requested) from the first user's turn
        # (2) use the canonical values of slot values
        constraint = []
        cons_delex = []
        for frame in user_t['frames']:
            serv_t = frame['service']

            if frame['state']['slot_values']:
                if not dial_goals.get(serv_t):
                    dial_goals[serv_t] = {'info': {}}

                if not dial_constraint.get(serv_t):
                    dial_constraint[serv_t] = {}

                for slot_name, vals in frame['state']['slot_values'].items():
                    v = vals[0]
                    if dial_canonical_vals.get(v):
                        v = dial_canonical_vals[v]
                    dial_goals[serv_t]['info'][slot_name] = v
                    dial_constraint[serv_t][slot_name] = v

            if frame['state']['requested_slots']:
                if not dial_goals.get(serv_t):
                    dial_goals[serv_t] = {'info': {}}
                dial_goals[serv_t]['reqt'] = frame['state']['requested_slots']

        for serv in dial_servs:
            if not dial_constraint.get(serv):
                continue

            constraint.append('[{}]'.format(dial_serv_dom[serv]))
            cons_delex.append('[{}]'.format(dial_serv_dom[serv]))

            for slot, val in dial_constraint[serv].items():
                delex = '[{}]'.format(slot)
                constraint.append(delex)
                constraint.append(val)

                cons_delex.append(delex)

            constraint.append(' || ')
            cons_delex.append(' || ')

        # sys_act: system action as ([*service*] ([*action*] *slot name* ..] ..)
        system_act_dict = {}
        for frame in system_t['frames']:
            serv = frame['service']
            system_act_dict[serv] = {}
            for act in frame['actions']:
                if not system_act_dict[serv].get(act['act']):
                    system_act_dict[serv][act['act']] = []
                if act['slot']:
                    system_act_dict[serv][act['act']].append(act['slot'])
        system_act = []
        for serv, acts in system_act_dict.items():
            system_act.append('[{}]'.format(dial_serv_dom[serv]))
            if 'inform_count' in acts:
                del acts['inform_count']
                if 'offer' in acts:
                    acts['offer'].insert(0, 'count')
                else:
                    acts['offer'] = ['count']
            for act, slots in acts.items():
                system_act.append('[{}]'.format(act))
                if act == 'offer_intent':  # change 'intent' to 'serv_intent'
                    system_act.extend(['[serv_{}]'.format(slot) for slot in slots])
                else:
                    system_act.extend(['[{}]'.format(slot) for slot in slots])

        # match: number of matches (records) within db by the current constraint
        match = ""

        # pointer: pointer to db
        #   based on the number of returned records,
        #   0th: 0 response; 1st: 1 response, 2nd: less than or equal to 3 responses, 3rd: more than 3 responses
        dbpointer = [0, 0, 0, 0]
        notify_pointer = [0, 0]

        for frame in system_t['frames']:
            # if not frame.get('service_results'):
            #     continue

            serv = frame['service']
            dom = serv.split('_')[0]
            if dom in NO_DB_DOMAINS:
                continue
            # get number of db results using the current constraint
            match = DB.get_match_num(dial_constraint, frame['service'])
            dbpointer = DB.addDBPointer(frame['service'], match)

            if system_act_dict.get(serv):
                notify_pointer = DB.addNotifyPointer(system_act_dict[serv])

        pointer = dbpointer + notify_pointer

        # turn_num: number of turn in a dialogue
        turn_num = i

        log_t = {'user': user_utt,
                 'user_delex': user_delex,
                 'resp': system_resp,
                 'pointer': ','.join([str(p) for p in pointer]),
                 'match': str(match),
                 'constraint': ' '.join(constraint[:-1]),
                 'cons_delex': ' '.join(cons_delex[:-1]),
                 'sys_act': ' '.join(system_act),
                 'turn_num': turn_num,
                 'turn_serv': ' '.join(['[{}]'.format(serv) for serv in turn_serv])
                 }
        format_rec['log'].append(log_t)

    format_rec['goal'] = dial_goals

    return record['unique_dialogue_id'], format_rec, dial_canonical_vals


def prepare_dialogue(processed_path, dial_df, DB):
    # prepare file with all dialogues, and files with train, dev and test id lists
    for set_name in SETS:
        dial_ids = dial_df.loc[dial_df['set'] == set_name]['unique_dialogue_id'].tolist()
        with open(processed_path + '{}_list.txt'.format(set_name), 'w') as outf:
            outf.write('\n'.join(dial_ids))
        print('  number of dialogues in {}: {}'.format(set_name, len(dial_ids)))

    dial_records = dial_df.to_json(orient='records')
    parsed = json.loads(dial_records)
    dat = [(rec, DB) for rec in parsed]
    with mp.Pool(processes=os.cpu_count()) as pool:
        processed_list = list(tqdm(pool.imap(process_dialogue_record, dat), total=len(dat)))

    dial_dict, canonical_val_dict = {}, {}
    for d_id, r, c in processed_list:
        dial_dict[d_id] = r
        canonical_val_dict = {**canonical_val_dict, **c}

    with open(processed_path + 'data.json', 'w') as outf:
        json.dump(dial_dict, outf, indent=2)

    with open(processed_path + 'canonical_value_mapping.json', 'w') as outf:
        json.dump(canonical_val_dict, outf, indent=2)


def get_services(processed_path, dial_df):
    # aggregate dialogue ids of different set of services
    dial_df['services_txt'] = dial_df['services'].apply(lambda x: '-'.join(x))
    serv_set_dict = dial_df.groupby('services_txt')['unique_dialogue_id'].apply(list).to_dict()
    with open(processed_path + 'service_sets.json', 'w') as outf:
        json.dump(serv_set_dict, outf, indent=2)
    print('  number of sets of services: {}'.format(len(serv_set_dict.keys())))

    serv_records = []
    for idx, row in dial_df.iterrows():
        for serv in row['services']:
            serv_records.append({'service': serv, 'dialogue_id': row['unique_dialogue_id']})
    serv_dial_df = pd.DataFrame.from_records(serv_records)
    serv_dial_dict = serv_dial_df.groupby('service')['dialogue_id'].apply(list).to_dict()
    with open(processed_path + 'services.json', 'w') as outf:
        json.dump(serv_dial_dict, outf, indent=2)
    print('  number of services: {}'.format(len(serv_dial_dict.keys())))


def get_slots(processed_path, schema_df):
    slot_dict = {}
    for idx, row in schema_df.iterrows():
        serv = row['service_name']
        slot_dict[serv] = {}
        for slot in row['slots']:
            slot_dict[serv][slot['name']] = {'description': slot['description'],
                                             'is_categorical': slot['is_categorical'],
                                             'possible_values': slot['possible_values']}

    with open(processed_path + 'slot_context.json', 'w') as outf:
        json.dump(slot_dict, outf, indent=2)


def get_serv_descs(schema_df):
    desc_dict = {row['service_name']: row['description'] for idx, row in schema_df.iterrows()}
    with open(processed_path + 'service_context.json', 'w') as outf:
        json.dump(desc_dict, outf, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v')  # extended version of sgd (v1-5)
    args = parser.parse_args()

    db_path = '../db/{}'.format(args.v + '/' if args.v else '')
    processed_path = '../data/sgd-processed/{}'.format(args.v + '/' if args.v else '')
    if args.v:
        root = '../data/dstc8-schema-guided-dialogue/sgd_x/data/' + args.v
    else:
        root = '../data/dstc8-schema-guided-dialogue/'

    if not os.path.exists(db_path):
        os.makedirs(db_path)
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    # read dialogues from train, dev, test sets
    print('reading dialogues from', root)
    dial_df = read_dialogue(root)

    # get sets of services with their associated dialogue ids
    print('aggregating sets of services to', processed_path)
    get_services(processed_path, dial_df)

    # read schema
    print('reading schema from', root)
    schema_df = read_schema(root)

    # aggregate slots
    print('aggregating slots to', processed_path)
    get_slots(processed_path, schema_df)

    # get description of services
    print('getting service descriptions')
    get_serv_descs(schema_df)

    # extract system_results for the search intents as records for db
    print('aggregating db records to', db_path)
    get_dbs(db_path, dial_df)

    # read dbs
    from db_ops import SGD_DB

    db_files = [f for f in os.listdir(db_path) if f.endswith(".json")]
    db_files.remove('value_set.json')

    service_ids_file = processed_path + 'services.json'
    all_servs = sorted(list(json.loads(open(service_ids_file, 'r').read().lower()).keys()))
    all_domains = sorted(list(set([serv.split('_')[0] for serv in all_servs])))

    no_db_domains = ['alarm', 'calendar', 'banks', 'messaging', 'payment', 'ridesharing']
    db_domains = sorted(list(set(all_domains) - set(no_db_domains)))
    db_servs = sorted([serv for serv in all_servs if serv.split('_')[0] in db_domains])

    DB = SGD_DB({f.replace('_db.json', ''): db_path + f for f in db_files}, db_servs)

    # prepare dialogues and train, dev, and test lists of ids
    print('preparing processed dialogue data to', processed_path)
    prepare_dialogue(processed_path, dial_df, DB)
