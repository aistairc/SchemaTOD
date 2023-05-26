import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch

from reader import SGD_Reader
from eval import SGD_Evaluator
from config import global_config as cfg


def evaluate(mode, result_path):

    reader = SGD_Reader()
    evaluator = SGD_Evaluator(reader)

    result_collection = json.loads(open(result_path, 'r').read())

    serv_sets_dict = {i: serv_set for serv_set, d_ids in reader.service_set_ids.items() for i in d_ids}
    train_files = {l.strip(): 1 for l in open(cfg.train_list, 'r').readlines()}
    train_servs = {serv: 1 for dial_id in train_files.keys() for serv in serv_sets_dict[dial_id].split('-')}

    eval_files = reader.test_files if mode == 'test' else reader.dev_files

    dial_ids = []
    unseen_dial_ids = []
    if 'all' not in cfg.eval_domains:
        for dial_id in eval_files.keys():
            for dom in cfg.eval_domains:
                for serv in serv_sets_dict[dial_id].split('-'):
                    if serv.split('_')[0] == dom:
                        dial_ids.append(dial_id)
                        if not train_servs.get(serv):
                            unseen_dial_ids.append(dial_id)
                        break
    else:
        for dial_id in eval_files.keys():
            dial_ids.append(dial_id)
            for serv in serv_sets_dict[dial_id].split('-'):
                if not train_servs.get(serv):
                    unseen_dial_ids.append(dial_id)
                    break

    if 'seen' in cfg.eval_dials:
        eval_dial_ids = {dial_id: 1 for dial_id in list(set(dial_ids) - set(unseen_dial_ids))}

    elif 'unseen' in cfg.eval_dials:
        eval_dial_ids = {dial_id: 1 for dial_id in list(set(unseen_dial_ids))}

    else:
        eval_dial_ids = {dial_id: 1 for dial_id in list(set(dial_ids))}

    eval_collection = {}
    for dial_id, res in result_collection.items():
        if eval_dial_ids.get(dial_id):
            eval_collection[dial_id] = res

    logging.info('running metrics on {}'.format(result_path))
    btm = time.time()
    results, _ = reader.wrap_result_lm(eval_collection)
    eval_results = evaluator.validation_metric(results,
                                               same_eval_as_cambridge=cfg.same_eval_as_cambridge,
                                               use_true_dspn_for_ctr_eval=cfg.use_true_dspn_for_ctr_eval,
                                               use_true_curr_bspn=cfg.use_true_curr_bspn)

    soft_score = 0.5 * (eval_results['soft_succ_rate'] + eval_results['soft_match_rate']) + eval_results['bleu']
    hard_score = 0.5 * (eval_results['hard_succ_rate'] + eval_results['hard_match_rate']) + eval_results['bleu']

    packed_data = evaluator.pack_dial(results)
    dst_evals = evaluator.dialog_state_tracking_eval(packed_data)
    joint_goal, f1, accuracy, slot_appear_num, slot_correct_num = dst_evals

    logging.info('scoring time: {:.2f} min'.format((time.time() - btm) / 60))
    res_txt = 'validation [CTR] bleu: {:.2f}'.format(eval_results['bleu']) + \
              ', soft match: {:.2f}'.format(eval_results['soft_match_rate']) + \
              ', soft success: {:.2f}'.format(eval_results['soft_succ_rate']) + \
              ', soft score: {:.2f}'.format(soft_score) + \
              ', hard match: {:.2f}'.format(eval_results['hard_match_rate']) + \
              ', hard success: {:.2f}'.format(eval_results['hard_succ_rate']) + \
              ', hard score: {:.2f}'.format(hard_score) + \
              ', joint acc: {:.2f}'.format(joint_goal) + \
              ', acc: {:.2f}'.format(accuracy) + \
              ', f1: {:.2f}'.format(f1)
    logging.info(res_txt)

    inference_res = {'bleu': eval_results['bleu'],
                     'soft_match_rate': eval_results['soft_match_rate'],
                     'soft_succ_rate': eval_results['soft_succ_rate'],
                     'soft_score': soft_score,
                     'hard_match_rate': eval_results['hard_match_rate'],
                     'hard_succ_rate': eval_results['hard_succ_rate'],
                     'hard_score': hard_score,
                     'joint_acc': joint_goal,
                     'acc': accuracy,
                     'f1': f1,
                     'result': res_txt}

    inference_cfg = '/'.join(cfg.result_path.split('/')[:-1])
    res_file_name = cfg.result_path.replace('-res.json', '')

    # save output results
    if 'all' in cfg.eval_domains:
        outp_res_txt = []
        for dial, soft_match, soft_success, hard_match, hard_success in zip(packed_data,
                                                                            eval_results['soft_matches'],
                                                                            eval_results['soft_successes'],
                                                                            eval_results['hard_matches'],
                                                                            eval_results['hard_successes']
                                                                            ):
            outp_res_txt.append({'soft match': soft_match,
                                 'soft success': soft_success,
                                 'hard match': hard_match,
                                 'hard success': hard_success,
                                 'dial': packed_data[dial]})

        json.dump(outp_res_txt, open(res_file_name + '-txt.json', 'w'), indent=2)
        logging.info('save txt results to {}'.format(res_file_name + '-txt.json'))

    summary_filename = os.path.join(inference_cfg, 'summary.json')
    extra_key = '-{}'.format('_'.join(cfg.eval_domains)) if 'all' not in cfg.eval_domains else ''
    extra_key += '-{}'.format('_'.join(cfg.eval_dials)) if 'all' not in cfg.eval_dials else ''
    if os.path.exists(summary_filename):
        res_json = json.load(open(summary_filename, 'r'))
        res_json[res_file_name + extra_key] = inference_res
        json.dump(res_json, open(summary_filename, 'w'), indent=2)
    else:
        res_json = {cfg.inference_path + extra_key: inference_res}
        json.dump(res_json, open(summary_filename, 'w'), indent=2)

    logging.info('update eval results to {}'.format(summary_filename))


def parse_arg_cfg(args):
    if args.cfg:  # add args to cfg
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode

    parse_arg_cfg(args)
    cfg.construct_paths()

    if args.mode not in ['test', 'dev']:
        raise Exception("mode should be 'test' or 'dev'")

    if not cfg.inference_path:
        raise Exception("require 'inference_path'")

    if not cfg.result_path:
        raise Exception("require 'result_path'")

    cfg.model_path = cfg.inference_path
    cfg.tok_path = cfg.inference_path
    cfg.opt_path = cfg.inference_path

    cfg.init_logging_handler(args.mode)

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    evaluate(args.mode, cfg.result_path)


if __name__ == "__main__":
    main()
