import json
from collections import defaultdict


class SGD_DB(object):
    def __init__(self, db_paths, db_servs, extractive_slots=None):
        self.dbs = {}
        self.sql_dbs = {}
        self.extractive_ontology = {}

        for serv, serv_db_path in db_paths.items():
            with open(serv_db_path, 'r') as f:
                self.dbs[serv] = json.loads(f.read().lower())

            if extractive_slots is not None:
                self.extractive_ontology[serv] = defaultdict(list)
                for ent in self.dbs[serv]:
                    for slot in extractive_slots[serv]:
                        if slot not in ent or ent[slot] in self.extractive_ontology[serv][slot]:
                            continue
                        self.extractive_ontology[serv][slot].append(ent[slot])

        self.db_servs = db_servs

    def oneHotVector(self, num):
        # return number of available entities for particular domain.
        vector = [0, 0, 0, 0]
        if num == '':
            return vector
        elif num == 0:
            vector = [1, 0, 0, 0]
        elif num == 1:
            vector = [0, 1, 0, 0]
        elif num <= 3:
            vector = [0, 0, 1, 0]
        else:
            vector = [0, 0, 0, 1]
        return vector

    def addDBPointer(self, serv, match_num, return_num=False):
        # create database pointer for all related domains.
        if serv in self.db_servs:
            vector = self.oneHotVector(match_num)
        else:
            vector = [0, 0, 0, 0]
        return vector

    def addNotifyPointer(self, turn_da):
        # add information if the request is successful or fail.
        # e.g. fail to reserve a hotel, successfully play selected song
        vector = [0, 0]
        if 'notify_failure' in turn_da:
            vector = [1, 0]
        elif 'notify_success' in turn_da:
            vector = [0, 1]
        return vector

    def addDBIndicator(self, serv, match_num, return_num=False):
        # create database indicator for all related domains.
        serv = serv[1:-1] if serv.startswith('[') else serv

        if serv in self.db_servs:
            vector = self.oneHotVector(match_num)
        else:
            vector = [0, 0, 0, 0]

        # '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]'
        if vector == [0, 0, 0, 0]:
            indicator = '[db_nores]'  # no response
        else:
            indicator = '[db_%s]' % vector.index(1)
        return indicator

    def get_match_num(self, constraints, serv, return_entry=False):
        # create database pointer for service.
        serv = serv[1:-1] if serv.startswith('[') else serv

        matched_ents = []
        if serv in self.db_servs:
            if constraints.get(serv):  # service
                matched_ents = self.queryJsons(serv, constraints[serv])
            elif constraints.get(serv.split('_')[0]):  # domain
                matched_ents = self.queryJsons(serv, constraints[serv.split('_')[0]])

        if return_entry:
            return matched_ents
        return len(matched_ents)

    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num

        if domain.endswith(']'):
            domain = domain[1:-1]
        nummap = {
            0: '0',
            1: '1',
            2: '2-3',
            3: '>3'
        }

        if vector[:4] == [0, 0, 0, 0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain + ': ' + nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'success'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'failure'

        return report

    def queryJsons(self, serv, constraints, name_match=True, exactly_match=True, return_name=False, return_id=False):
        # returns the list of entities for a given service
        # based on the constraints of the belief state
        # constraints: dict e.g. {'[price_range]': 'cheap', '[area]': 'west'}

        for v in constraints.values():
            if v in ['not mentioned', '']:
                return []

        match_result = []
        skip_cases = {
            "don't care": 1,
            "do n't care": 1,
            "dont care": 1,
            "not mentioned": 1,
            "dontcare": 1,
            "": 1
        }

        for db_ent in self.dbs[serv]:
            match = True
            for s, v in constraints.items():
                s = s.split('/')[-1]  # remove service name
                v = v.lower()

                if s.endswith('name') and not name_match:
                    continue

                if skip_cases.get(v):
                    continue

                if s not in db_ent:
                    match = False
                    break

                v = 'yes' if v == 'free' else v

                db_v = db_ent[s]

                # inbound_arrival_time, outbound_arrival_time,
                # leaving_time, departure_time, inbound_departure_time, outbound_departure_time
                # available_start_time, available_end_time
                # event_time, show_time, pickup_time, wait_time, appointment_time, time
                if s.endswith('time'):

                    try:
                        # raise error if time value is not xx:xx format (constraint val)
                        h, m = v.split(':')
                        v = int(h) * 60 + int(m)
                        # raise error if time value is not xx:xx format (db val)
                        db_h, db_m = db_v.split(':')
                        db_v = int(db_h) * 60 + int(db_m)
                    except ValueError:
                        match = False
                        break

                    # time = int(db_ent[s].split(':')[0])*60 + int(db_ent[s].split(':')[1])
                    if 'arrival' in s and v > db_v:
                        match = False
                        break
                    elif 'leaving' in s and v < db_v:
                        match = False
                        break
                    elif 'departure' in s and v < db_v:
                        match = False
                        break
                    elif 'time' == s:
                        if v - 30 > db_v or v + 30 < db_v:
                            match = False
                            break
                    elif v > db_v:
                        match = False
                        break

                else:
                    if exactly_match and v != db_v:
                        match = False
                        break
                    elif v not in db_v:
                        match = False
                        break

            if match:
                match_result.append(db_ent)

        if return_name and match_result:  # return only fields indicating names
            names = [k for k in match_result[0] if 'name' in k]
            if names:
                match_result = [{n: res[n] for n in names} for res in match_result]

        elif return_id and match_result:  # return id of the matched record
            match_result = [res['id'] for res in match_result]

        return match_result


if __name__ == '__main__':
    db_paths = {
        'movies_1': '../db/movies_1_db.json',
        'services_1': '../db/services_1_db.json',
        'restaurants_1': '../db/restaurants_1_db.json',
        'flights_1': '../db/flights_1_db.json',
        'hotels_1': '../db/hotels_1_db.json',
        'hotels_2': '../db/hotels_2_db.json'
    }

    db_servs = ['movies_1', 'services_1', 'restaurants_1',
                'flights_1', 'hotels_1', 'hotels_2']

    db = SGD_DB(db_paths, db_servs)

    # restaurants_2-location=san francisco;category=japanese;price_range=moderate;number_of_seats=3;restaurant_name=2g japanese brasserie;time=11:00;date=2019-03-01

    while True:
        inp = input('input belief state in format: service-slot1=value1;slot2=value2...\n')

        constraints = {}
        inp_slits = inp.lower().split('-')
        serv, cons = inp_slits[0], '-'.join(inp_slits[1:])
        for sv in cons.split(';'):
            s, v = sv.split('=')
            constraints[s] = v
        print(constraints)

        print('---- JSON results ----')
        res = db.queryJsons(serv, constraints, return_id=True)
        print(res)
        print('count:', len(res))
        print('db indicator:', db.addDBIndicator('['+serv+']', len(res)))
