import math
from collections import Counter
from nltk.util import ngrams


class BLEUScorer(object):
    # BLEU score calculator via GentScorer interface
    # it calculates the BLEU-4 by taking the entire corpus in
    # calculate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        if c > r:
            bp = 1
        elif c == 0.0:
            bp = 0
        else:
            bp = math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


class SGD_Evaluator(object):
    def __init__(self, reader):
        self.reader = reader
        self.ontology = reader.ontology
        self.servs = self.ontology.all_servs
        self.service_set_ids = self.reader.service_set_ids
        self.all_data = self.reader.data
        self.test_data = self.reader.test

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for serv, s_list in self.ontology.slots.items():
            dom = serv.split('_')[0]
            for s in s_list:
                if s in ['serv_intent', 'count']:
                    continue
                info_slot = dom + '-' + s
                if info_slot not in self.all_info_slot:
                    self.all_info_slot.append(info_slot)
        # for serv, s_list in self.ontology.slots.items():
        #     for s in s_list:
        #         self.all_info_slot.append(serv + '-' + s)

        # only evaluate these slots for dialog success
        # self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        self.requestables = self.ontology.all_reqslot

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def validation_metric(self, data, same_eval_as_cambridge, use_true_dspn_for_ctr_eval, use_true_curr_bspn):
        bleu = self.bleu_metric(data)
        eval_stats = self.context_to_response_eval(data, same_eval_as_cambridge=same_eval_as_cambridge,
                                                   use_true_dspn_for_ctr_eval=use_true_dspn_for_ctr_eval,
                                                   use_true_curr_bspn=use_true_curr_bspn)
        eval_stats['bleu'] = bleu
        return eval_stats

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [], []
        for row in data:
            if eval_dial_list and row['dial_id'] not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth)) if gen and truth else 0.0

        return sc

    def context_to_response_eval(self, data, eval_dial_list=None, same_eval_as_cambridge=False,
                                 use_true_dspn_for_ctr_eval=False, use_true_curr_bspn=False):
        dials = self.pack_dial(data)

        counts = {}
        for req in self.requestables:
            counts[req + '_total'] = 0
            counts[req + '_offer'] = 0

        dial_num = 0
        dial_w_goal = 0
        soft_successes, soft_matches = [], []
        hard_successes, hard_matches = [], []

        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue

            dial = dials[dial_id]
            goal = self.all_data[dial_id]['goal'] if self.all_data[dial_id].get('goal') else {}

            reqs = {}
            for serv in goal.keys():
                reqs[serv] = goal[serv]['reqt'] if goal[serv].get('reqt') else []

            if len(goal.keys()) > 0:
                soft_success, hard_success, soft_match, hard_match, stats, counts = self._evaluateGeneratedDialogue(
                    dial, goal, reqs, counts, same_eval_as_cambridge=same_eval_as_cambridge,
                    use_true_dspn_for_ctr_eval=use_true_dspn_for_ctr_eval, use_true_curr_bspn=use_true_curr_bspn)
                dial_w_goal += 1
            else:
                soft_success, hard_success, soft_match, hard_match = 0., 0., 0., 0.
                # stats = {serv: [0, 0, 0] for serv in self.ontology.all_servs}

            soft_successes.append(soft_success)
            hard_successes.append(hard_success)
            soft_matches.append(soft_match)
            hard_matches.append(hard_match)

            dial_num += 1

        eval_stats = {'soft_successes': soft_successes,
                      'soft_succ_rate': sum(soft_successes) / (float(dial_w_goal) + 1e-10) * 100,
                      'hard_successes': hard_successes,
                      'hard_succ_rate': sum(hard_successes) / (float(dial_w_goal) + 1e-10) * 100,
                      'soft_matches': soft_matches,
                      'soft_match_rate': sum(soft_matches) / (float(dial_w_goal) + 1e-10) * 100,
                      'hard_matches': hard_matches,
                      'hard_match_rate': sum(hard_matches) / (float(dial_w_goal) + 1e-10) * 100,
                      'counts': counts,
                      'dial_with_goal': dial_w_goal,
                      'dial_num': dial_num}

        return eval_stats

    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                   soft_acc=False, same_eval_as_cambridge=False,
                                   use_true_dspn_for_ctr_eval=False, use_true_curr_bspn=False):
        # evaluate the dialogue created by the model.
        # first, we load the user goal of the dialogue.
        # then, for each turn generated by the system, we look for key-words.
        # inform measures whether a dialogue system provides correct entities (e.g., the name of restaurant)
        # success measures whether it has answered all the requested information

        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}  # the request slot information that have been provided
        venue_offered = {}  # number of returned results

        for serv in goal.keys():
            venue_offered[serv] = []
            provided_requestables[serv] = []

        for t, turn in enumerate(dialog):
            if t == 0:
                continue

            sent_t = turn['resp_gen']
            for serv in goal.keys():
                dom = serv.split('_')[0]  # e.g. restaurants

                # for computing success
                if same_eval_as_cambridge:
                    if use_true_dspn_for_ctr_eval:
                        dom_pred = [d[1:-1] for d in turn['dspn'].split()]  # e.g. [restaurants] -> restaurants
                    else:
                        dom_pred = [d[1:-1] for d in turn['dspn_gen'].split()]  # e.g. [restaurants] -> restaurants

                    if dom not in dom_pred:  # fail
                        continue

                if serv in self.ontology.db_servs:
                    if not use_true_curr_bspn:
                        bspn = turn['bspn_gen']
                    else:
                        bspn = turn['bspn']

                    constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

                    if constraint_dict.get(dom):
                        cons_by_dom = constraint_dict[dom]
                        venues = self.reader.db.queryJsons(serv, cons_by_dom, return_id=True)
                    else:
                        venues = []

                    venue_offered[serv] = venues

                    # if venues:
                    #     venue_offered[serv] = venues
                    # else:
                    #     flag = False
                    #     for ven in venues:
                    #         if ven not in venue_offered[serv]:
                    #             flag = True
                    #             break
                    #     if flag and venues:  # sometimes there are no results so sample won't work
                    #         venue_offered[serv] = venues

                else:  # services that do not require db
                    venue_offered[serv] = '[value_id]'

                # check whether the requested slots appear in the generated sentence
                for requestable in requestables:
                    # checking slot in resp, ex. [value_rating] or [rating]
                    if '[value_' + requestable + ']' in sent_t or '[' + requestable + ']' in sent_t:
                        provided_requestables[serv].append(requestable)

        # serv match rate is 1 if db results of the service in goal match with ground truth
        # serv success rate is 1 if 1) serv match rate is 1 and
        #                           2) every requestable slot of the service in goal appear in response
        # HARD EVAL: match rate = 1 if sum(serv match rate) / sum(services) == 1, otherwise 0.
        #          : success rate = 1 if sum(serv success rate) / sum(services) == 1, otherwise 0.
        # SOFT EVAL: match rate = sum(serv match rate) / sum(services),
        #          : success rate = sum(serv success rate) / sum(services)
        stats = {serv: [0, 0, 0] for serv in self.ontology.all_servs}
        match = {serv: 0. for serv in goal.keys()}
        success = {serv: 0. for serv in goal.keys()}

        # MATCH (INFORM): compare db result ids by the final predicted constraints and the goal db result ids
        for serv in goal.keys():
            match_stat = 0
            if serv in self.ontology.db_servs:
                goal_serv = goal[serv]['info']
                goal_venues = self.reader.db.queryJsons(serv, goal_serv, return_id=True) if goal_serv else []
                # if len(venue_offered[serv]) > 0 and set(venue_offered[serv]) == set(goal_venues):
                if set(venue_offered[serv]) == set(goal_venues):
                    match[serv] = 1.
                    match_stat = 1
            else:
                if '_id]' in venue_offered[serv]:
                    match[serv] = 1.
                    match_stat = 1

            stats[serv][0] = match_stat
            stats[serv][2] = 1

        # if no goal then automatically set match to one
        soft_match = sum(match.values()) / len(goal.keys())
        hard_match = 1.0 if sum(match.values()) == len(goal.keys()) else 0.0

        for serv in goal.keys():
            for request in real_requestables[serv]:
                counts[request + '_total'] += 1
                if request in provided_requestables[serv]:
                    counts[request + '_offer'] += 1

        # SUCCESS: check if request's slots of every service appears in the system response
        for serv in goal.keys():
            success_stat = 0

            if match[serv] == 1. and len(real_requestables[serv]) == 0:  # no request slots
                success[serv] = 1.
                success_stat = 1
                stats[serv][1] = success_stat
                continue

            request_success = 0
            for request in real_requestables[serv]:
                if request in provided_requestables[serv]:
                    request_success += 1

            if match[serv] == 1. and request_success == len(real_requestables[serv]):
                success[serv] = 1.
                success_stat = 1

            stats[serv][1] = success_stat

        soft_success = sum(success.values()) / len(goal.keys())
        hard_success = 1 if hard_match == 1. and sum(success.values()) >= len(goal.keys()) else 0

        return soft_success, hard_success, soft_match, hard_match, stats, counts

    def _bspn_to_dict(self, bspn):
        constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s, v in cons.items():
                key = domain + '-' + s
                constraint_dict_flat[key] = v

        return constraint_dict_flat

    def value_similar(self, a, b):
        return True if a == b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True

        return False

    def _constraint_compare(self, truth_cons, gen_cons, slot_appear_num=None, slot_correct_num=None):
        tp, fp, fn = 0, 0, 0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            # v_truth = truth_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(
                        slot) else slot_correct_num.get(slot) + 1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(
                    slot) else slot_appear_num.get(slot) + 1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp, fp, fn, acc, list(set(false_slot))

    def dialog_state_tracking_eval(self, dials, eval_dial_list=None):
        total_turn, joint_match = 0, 0
        total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0

        for dial_id in dials:

            if eval_dial_list and dial_id not in eval_dial_list:
                continue

            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num, turn in enumerate(dial):
                gen_cons = self._bspn_to_dict(turn['bspn_gen'])
                truth_cons = self._bspn_to_dict(turn['bspn'])

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                if eval_dial_list is None:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(truth_cons,
                                                                            gen_cons,
                                                                            slot_appear_num,
                                                                            slot_correct_num)
                else:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(truth_cons,
                                                                            gen_cons)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn + 1e-10) * 100

        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num


if __name__ == '__main__':
    pass
