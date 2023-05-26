import copy

import torch
import torch.nn as nn
import os
import random
import argparse
import time
import logging
import json
import concurrent.futures
import numpy as np
import warnings

from transformers.modeling_outputs import BaseModelOutput
from transformers.optimization import AdamW
from transformers.trainer_pt_utils import get_parameter_names

import utils
from eval import SGD_Evaluator
from config import global_config as cfg
from reader import SGD_Reader
from model import T5WithSchemaDependency

warnings.filterwarnings('ignore')


class Trainer(object):
    def __init__(self, device):

        self.device = device

        self.reader = SGD_Reader()

        self.model = T5WithSchemaDependency.from_pretrained(cfg.model_path,
                                                            cfg.domain_size,
                                                            cfg.slot_size)

        if cfg.mode == 'train' and not cfg.resume:
            self.model.resize_token_embeddings(cfg.vocab_size)
            self.model.initialize_additional_decoder()

        self.model.to(self.device)
        if cfg.cuda and cfg.mode in ['train', 'finetune']:
            self.model = nn.DataParallel(self.model)

        self.evaluator = SGD_Evaluator(self.reader)

    def step(self, inputs):

        stats = {'dspn_loss': 0., 'bspn_loss': 0., 'resp_loss': 0.,
                 'dspn_correct': 0., 'bspn_correct': 0., 'resp_correct': 0.,
                 'dspn_count': 0., 'bspn_count': 0., 'resp_count': 0.}

        dial_hidden_states = None
        loss = torch.tensor(0., device=self.device)
        inputs = utils.to_device(inputs, self.device)

        attention_mask = torch.where(inputs['dial_contexts'] == cfg.pad_token_id, 0, 1)

        _, _, num_c, num_c_tkn = inputs['slot_contexts'].size()
        context_ids = inputs['slot_contexts'].view(-1, num_c, num_c_tkn)
        context_lbls = inputs['slot_lbls'].view(-1, num_c)
        context_embs = self.model(context_ids=context_ids, encode_context=True) if cfg.use_context else None

        if cfg.train_dspn:
            dspn_inps = {'input_ids': inputs['dial_contexts'],
                         'attention_mask': attention_mask,
                         'labels': inputs['dspn_lbls'],
                         'lm_mask_labels': inputs['domain_lbls'],
                         'return_dict': False}

            dspn_outps = self.model(**dspn_inps)

            dspn_loss = cfg.dspn_coeff * self.average_loss(dspn_outps[0], dspn_inps["labels"])
            stats['dspn_correct'], stats['dspn_count'] = self.count_tokens(dspn_outps[1], labels=dspn_inps["labels"])

            dial_hidden_states = dspn_outps[-1]

            loss += dspn_loss
            stats['dspn_loss'] = dspn_loss.item()

        if cfg.train_bspn:
            serv_flag = torch.tensor(utils.flatten_list(inputs['slot_dial_servs_flag']), device=self.device)
            lm_mask_labels = torch.cat((inputs['domain_lbls'].repeat_interleave(inputs['dspn_len'], dim=0),
                                        context_lbls[serv_flag.bool()]), dim=-1)

            bspn_inps = {'attention_mask': attention_mask.repeat_interleave(inputs['dspn_len'], dim=0),
                         'labels': inputs['bspn_lbls'],  # predicted bspn labels
                         'lm_mask_labels': lm_mask_labels,  # all domain labels + related slot tokens
                         'context_ids': context_ids[serv_flag.bool()],
                         'context_embs': context_embs[serv_flag.bool()] if cfg.use_context else None,
                         'context_labels': context_lbls[serv_flag.bool()],
                         'decode_bspn': True,
                         'return_dict': False}

            if dial_hidden_states is None:
                bspn_inps['input_ids'] = inputs['dial_contexts'].repeat_interleave(inputs['dspn_len'], dim=0)
            else:
                bspn_inps['encoder_outputs'] = BaseModelOutput(
                    last_hidden_state=dial_hidden_states.repeat_interleave(inputs['dspn_len'], dim=0))

            bspn_outps = self.model(**bspn_inps)
            bspn_loss = cfg.bspn_coeff * self.average_loss(bspn_outps[0], bspn_inps["labels"])
            stats['bspn_correct'], stats['bspn_count'] = self.count_tokens(bspn_outps[1], labels=bspn_inps["labels"])

            loss += bspn_loss
            stats['bspn_loss'] = bspn_loss.item()

        if cfg.train_resp:
            serv_flag = torch.tensor(utils.flatten_list(inputs['slot_turn_servs_flag']), device=self.device)
            resp_inps = {'attention_mask': attention_mask,
                         'labels': inputs['resp_lbls'],
                         'lm_mask_labels': torch.cat((inputs['domain_lbls'], context_lbls[serv_flag.bool()]), dim=-1),
                         'context_ids': context_ids[serv_flag.bool()],
                         'context_embs': context_embs[serv_flag.bool()] if cfg.use_context else None,
                         'context_labels': context_lbls[serv_flag.bool()],
                         'decode_resp': True,
                         'return_dict': False}

            if dial_hidden_states is None:
                resp_inps['input_ids'] = inputs['dial_contexts']
            else:
                resp_inps['encoder_outputs'] = BaseModelOutput(last_hidden_state=dial_hidden_states)

            resp_outps = self.model(**resp_inps)

            resp_loss = cfg.resp_coeff * self.average_loss(resp_outps[0], resp_inps["labels"])
            stats['resp_correct'], stats['resp_count'] = self.count_tokens(resp_outps[1], labels=resp_inps["labels"])

            loss += resp_loss
            stats['resp_loss'] = resp_loss.item()

        return loss, stats

    def train(self):
        logging.info('***** getting train batches *****')
        train_batches = self.reader.get_batches('train')
        train_batches = self.reader.process_batches(train_batches)

        dev_batches = []
        if cfg.validate_during_training:
            logging.info('***** getting validate batches *****')
            dev_batches = self.reader.get_batches('dev')
            dev_batches = self.reader.process_batches(dev_batches)

        optimizer = self.get_optimizers()

        start_epoch_num = cfg.resume_from_epoch if cfg.resume_from_epoch else 0

        set_stats = self.reader.set_stats['train']
        num_training_steps_per_epoch = set_stats['num_training_steps_per_epoch']
        num_opt_steps_per_epoch = -(set_stats['num_training_steps_per_epoch'] // -cfg.gradient_accumulation_steps)
        num_opt_steps = num_opt_steps_per_epoch * (cfg.epoch_num - start_epoch_num)

        logging.info('***** training setting *****')
        logging.info('  num training steps per epoch = {}'.format(num_training_steps_per_epoch))
        logging.info('  num turns = {}'.format(set_stats['num_turns']))
        logging.info('  num dialogs = {}'.format(set_stats['num_dials']))
        logging.info('  num epochs = {}, starts from {}'.format(cfg.epoch_num, start_epoch_num + 1))
        logging.info('  num vocab = {}'.format(cfg.vocab_size))
        logging.info('  batch size = {}'.format(cfg.batch_size))
        logging.info('  learning rate = {:.5f}'.format(cfg.lr))
        logging.info('  gradient accumulation steps = {}'.format(cfg.gradient_accumulation_steps))
        logging.info('  total optimization steps = {}'.format(num_opt_steps))

        log_inputs = 3
        global_step = 0

        for epoch in range(start_epoch_num, cfg.epoch_num):
            self.model.train()
            self.model.zero_grad()

            epoch_step = 0
            stats = {'dspn_loss': 0., 'bspn_loss': 0., 'resp_loss': 0.,
                     'dspn_correct': 0., 'bspn_correct': 0., 'resp_correct': 0.,
                     'dspn_count': 0., 'bspn_count': 0., 'resp_count': 0.}
            logging_dspn_loss, logging_bspn_loss, logging_resp_loss = 0., 0., 0.

            btm = time.time()

            train_iterator = self.reader.get_data_iterator(train_batches)
            for batch, inputs in enumerate(train_iterator):
                if log_inputs > 0:  # log inputs for the very first two turns
                    self.log_first_inputs(inputs)
                    log_inputs -= 1

                loss, batch_stats = self.step(inputs)

                loss.backward()

                for k in stats:
                    stats[k] += batch_stats[k]

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                epoch_step += 1

                # step, wrt gradient_accumulation_steps, clip grad norm
                if (epoch_step % cfg.gradient_accumulation_steps == 0 or
                        epoch_step == set_stats['num_training_steps_per_epoch']):
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1  # global_step: actual step the optimizer took

                    if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                        loss_dspn_scalar = (stats['dspn_loss'] - logging_dspn_loss) / cfg.report_interval
                        loss_bspn_scalar = (stats['bspn_loss'] - logging_bspn_loss) / cfg.report_interval
                        loss_resp_scalar = (stats['resp_loss'] - logging_resp_loss) / cfg.report_interval
                        loss = loss_dspn_scalar + loss_bspn_scalar + loss_resp_scalar

                        acc, dspn_acc, bspn_acc, resp_acc = self.calculate_accuracy(stats)

                        log_txt = 'global opt. step: {}/{}'.format(global_step, num_opt_steps)
                        log_txt += ', epoch step: {}/{}, '.format(epoch_step, num_training_steps_per_epoch)
                        log_txt += self.report_stats({'loss': loss, 'acc': acc,
                                                      'dspn_loss': loss_dspn_scalar,
                                                      'bspn_loss': loss_bspn_scalar,
                                                      'resp_loss': loss_resp_scalar,
                                                      'dspn_acc': dspn_acc,
                                                      'bspn_acc': bspn_acc,
                                                      'resp_acc': resp_acc})
                        logging.info(log_txt)

                        logging_dspn_loss = stats['dspn_loss']
                        logging_bspn_loss = stats['bspn_loss']
                        logging_resp_loss = stats['resp_loss']

            loss = stats['dspn_loss'] + stats['bspn_loss'] + stats['resp_loss']
            acc, dspn_acc, bspn_acc, resp_acc = self.calculate_accuracy(stats)

            log_txt = ' ++ train epoch {}'.format(epoch + 1)
            log_txt += ', time: {:.2f} min, '.format((time.time() - btm) / 60)
            log_txt += self.report_stats({'loss': loss, 'acc': acc,
                                          'dspn_loss': stats['dspn_loss'],
                                          'bspn_loss': stats['bspn_loss'],
                                          'resp_loss': stats['resp_loss'],
                                          'dspn_acc': dspn_acc,
                                          'bspn_acc': bspn_acc,
                                          'resp_acc': resp_acc})
            logging.info(log_txt)

            if cfg.validate_during_training:  # validate
                self.validate(dev_batches, epoch)

            if (epoch + 1) % cfg.save_model_interval == 0:  # save model
                self.save_checkpoint(epoch + 1, optimizer)

    def validate(self, batches, epoch):
        self.model.eval()

        stats = {'dspn_loss': 0., 'bspn_loss': 0., 'resp_loss': 0.,
                 'dspn_correct': 0., 'bspn_correct': 0., 'resp_correct': 0.,
                 'dspn_count': 0., 'bspn_count': 0., 'resp_count': 0.}

        btm = time.time()
        with torch.no_grad():
            dev_iterator = self.reader.get_data_iterator(batches)
            for batch, inputs in enumerate(dev_iterator):
                _, batch_stats = self.step(inputs)
                for k in stats:
                    stats[k] += batch_stats[k]

        loss = stats['dspn_loss'] + stats['bspn_loss'] + stats['resp_loss']
        acc, dspn_acc, bspn_acc, resp_acc = self.calculate_accuracy(stats)

        log_txt = ' ++ validation epoch {}'.format(epoch + 1)
        log_txt += ', time: {:.2f} min, '.format((time.time() - btm) / 60)
        log_txt += self.report_stats({'loss': loss, 'acc': acc,
                                      'dspn_loss': stats['dspn_loss'],
                                      'bspn_loss': stats['bspn_loss'],
                                      'resp_loss': stats['resp_loss'],
                                      'dspn_acc': dspn_acc,
                                      'bspn_acc': bspn_acc,
                                      'resp_acc': resp_acc})
        logging.info(log_txt)

    def inference(self, data='dev'):
        # predict one dialog / one turn at a time
        self.model.eval()

        cfg.batch_size = cfg.batch_size / len(cfg.cuda_device) if cfg.cuda else cfg.batch_size

        logging.info('***** getting batches *****')
        batches = self.reader.get_batches(data)
        batches = [batch for batch in batches if batch]

        set_stats = self.reader.set_stats[data]
        logging.info('num dialogue = {}'.format(set_stats['num_dials']))
        logging.info('num turns = {}'.format(set_stats['num_turns']))

        replicas = nn.parallel.replicate(self.model, cfg.cuda_device) if cfg.cuda else [self.model]

        btm = time.time()
        result_collection = {}
        dial_finished = 0
        for batch_set_idx in range(0, len(batches), len(cfg.cuda_device)):

            batch_sets = batches[batch_set_idx: batch_set_idx + len(cfg.cuda_device)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = executor.map(self.inference_worker, [(rep, batch) for rep, batch in zip(replicas, batch_sets)])
                for future in futures:
                    result_collection.update(future)

            dial_finished += sum([len(batch) for batch in batch_sets])
            if dial_finished % 64 == 0:
                logging.info('** inference finished {} dialog'.format(dial_finished))

        logging.info('inference time: {:.2f} min'.format((time.time() - btm) / 60))

        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)

        eval_results = self.evaluator.validation_metric(results,
                                                        same_eval_as_cambridge=cfg.same_eval_as_cambridge,
                                                        use_true_dspn_for_ctr_eval=cfg.use_true_dspn_for_ctr_eval,
                                                        use_true_curr_bspn=cfg.use_true_curr_bspn)
        soft_score = 0.5 * (eval_results['soft_succ_rate'] + eval_results['soft_match_rate']) + eval_results['bleu']
        hard_score = 0.5 * (eval_results['hard_succ_rate'] + eval_results['hard_match_rate']) + eval_results['bleu']

        packed_data = self.evaluator.pack_dial(results)
        dst_evals = self.evaluator.dialog_state_tracking_eval(packed_data)
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

        model_mode, model_setting, inference_epoch = cfg.inference_path.split('/')[:3]

        inference_on = model_setting
        if 'Xdomain' in model_mode:
            inference_on = 'x-' + inference_on
        elif 'Fdomain' in model_mode:
            inference_on = 'f-' + inference_on
        inference_on = data + '-' + inference_on

        # save output results
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

        inference_cfg = inference_epoch
        inference_cfg += '-truedspn' if cfg.use_true_curr_dspn else ''
        inference_cfg += '-truebspn' if cfg.use_true_curr_bspn else ''
        inference_cfg += '-truedb' if cfg.use_true_db_pointer else ''

        log_path = os.path.join(cfg.log_path, inference_on)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        txt_filename = os.path.join(log_path, inference_cfg + '-txt.json')
        json.dump(outp_res_txt, open(txt_filename, 'w'), indent=2)
        logging.info('save txt results to {}'.format(txt_filename))

        res_filename = os.path.join(log_path, inference_cfg + '-res.json')
        json.dump(result_collection, open(res_filename, 'w'), indent=2)
        logging.info('save result collections to {}'.format(res_filename))

        summary_filename = os.path.join(log_path, 'summary.json')
        if os.path.exists(summary_filename):
            res_json = json.load(open(summary_filename, 'r'))
            res_json[log_path + '/' + inference_cfg] = inference_res
            json.dump(res_json, open(summary_filename, 'w'), indent=2)
        else:
            res_json = {log_path + '/' + inference_cfg: inference_res}
            json.dump(res_json, open(summary_filename, 'w'), indent=2)

        logging.info('update eval results to {}'.format(summary_filename))

        return inference_res

    def inference_worker(self, inps):
        module, dial_batch = inps
        module.eval()

        result_collection = {}

        gen_params = {'do_sample': cfg.do_sample,
                      'temperature': cfg.temperature,
                      'top_k': cfg.top_k,
                      'top_p': cfg.top_p,
                      'pad_token_id': cfg.pad_token_id,
                      'eos_token_id': cfg.eos_token_id,
                      'return_dict_in_generate': True,
                      'output_hidden_states': True,
                      'output_scores': True}

        num_dial = len(dial_batch)
        num_turn = len(dial_batch[0])

        batch_pv_turns = [{} for _ in range(num_dial)]
        batch_bspn_gen = [{} for _ in range(num_dial)]
        batch_offered = [{} for _ in range(num_dial)]
        batch_decoded = {}

        batch_serv_history = [[] for _ in range(num_dial)]

        for turn_idx in range(num_turn):
            batch_turns, batch_inputs = self.reader.process_batch_turn_eval(turn_idx, dial_batch, batch_pv_turns)

            batch_turn_serv = []
            for dial_idx, turn in enumerate(batch_turns):
                turn_serv = turn['turn_serv'][-1]
                batch_turn_serv.append(turn_serv)

            batch_pad_inputs = {}
            for k in batch_inputs:
                batch_pad_inputs[k] = utils.pad(batch_inputs[k], cfg.pad_token_id, trunc_len=cfg.max_context_length)
            batch_attention_mask = torch.where(batch_pad_inputs['dial_contexts'] == cfg.pad_token_id, 0, 1)

            # ############
            # predict dspn
            if not cfg.use_true_curr_dspn:

                decoder_input_ids = [[cfg.pad_token_id] + self.reader.encode('<sos_d>')] * num_dial
                dspn_inps = {'input_ids': batch_pad_inputs['dial_contexts'],
                             'attention_mask': batch_attention_mask,
                             'decoder_input_ids': decoder_input_ids,
                             'lm_mask_labels': batch_pad_inputs['domain_lbls']}

                dspn_inps = utils.to_tensor(dspn_inps)
                dspn_inps = utils.to_device(dspn_inps, device=module.device)

                gen_params['max_length'] = 5
                # gen_params['eos_token_id'] = self.reader.encode('<eos_d>')[0]

                dspn_outps = module.generate(**dspn_inps, **gen_params)
                batch_generated_ds = dspn_outps.sequences.cpu().numpy().tolist()

                batch_dspn_gen = []
                for generated_ds in batch_generated_ds:
                    batch_dspn_gen.append(self.decode_generated_dspn(generated_ds[1:]))

                dial_hidden_states = dspn_outps.encoder_hidden_states[-1]

                for dial_idx in range(num_dial):
                    # get turn dom
                    dspn_gen = batch_dspn_gen[dial_idx]
                    pred_doms = self.reader.decode(dspn_gen[1:-1]).split()  # remove start and end domain tags
                    pred_dom = self.get_turn_domain(pred_doms)
                    # use ground truth serv or random serv of the predicted dom
                    serv = batch_turn_serv[dial_idx]
                    serv = serv if serv.startswith(pred_dom[:-1]) else pred_dom[:-1] + '_1]'
                    batch_serv_history[dial_idx].append(serv)

                batch_decoded['dspn'] = batch_dspn_gen
                # print(self.reader.batch_decode(batch_decoded['dspn']))

            else:
                dial_hidden_states = None
                for dial_idx in range(num_dial):
                    batch_serv_history[dial_idx].append(batch_turn_serv[dial_idx])

            batch_slot_inps = {'slot_doms': [], 'slot_contexts': [], 'slot_lbls': [], 'lm_mask_lbls': []}
            for serv_hist in batch_serv_history:
                slot_inps = self.reader.dspn_to_slots(serv_hist[-1])
                for k in slot_inps:
                    batch_slot_inps[k].append(slot_inps[k])

            for k in batch_slot_inps:
                batch_slot_inps[k] = utils.pad(batch_slot_inps[k], cfg.pad_token_id)
            batch_slot_inps = utils.to_device(batch_slot_inps, module.device)

            batch_context_inps = {}
            if cfg.use_context:
                batch_context_inps = {'context_ids': batch_slot_inps['slot_contexts'],
                                      'context_embs': module(context_ids=batch_slot_inps['slot_contexts'],
                                                             encode_context=True),
                                      'context_labels': batch_slot_inps['slot_lbls']
                                      }
            batch_context_inps = {'lm_mask_labels': batch_slot_inps['lm_mask_lbls'],
                                  **batch_context_inps}

            # ############
            # predict bspn
            batch_db = [turn['db'] for turn in batch_turns]
            if not cfg.use_true_curr_bspn:
                decoder_input_ids = []
                for dial_idx in range(num_dial):
                    if cfg.use_true_curr_dspn:
                        dom = batch_serv_history[dial_idx][-1].split('_')[0] + ']'
                        decoder_input_ids.append([cfg.pad_token_id] + self.reader.encode('<sos_b> ' + dom))
                    else:
                        decoder_input_ids.append([cfg.pad_token_id] + self.reader.encode('<sos_b>'))

                bspn_inps = {'decoder_input_ids': decoder_input_ids,
                             'attention_mask': batch_attention_mask,
                             'decode_bspn': True,
                             **batch_context_inps}

                if dial_hidden_states is not None:
                    bspn_inps['encoder_outputs'] = BaseModelOutput(last_hidden_state=dial_hidden_states)
                else:
                    bspn_inps['input_ids'] = batch_pad_inputs['dial_contexts']

                bspn_inps = utils.to_tensor(bspn_inps)
                bspn_inps = utils.to_device(bspn_inps, module.device)

                gen_params['max_length'] = 60
                # gen_params['eos_token_id'] = self.reader.encode('<eos_b>')[0]
                start_token_idx = bspn_inps['decoder_input_ids'].size(-1) - 1

                bspn_outps = module.generate(**bspn_inps, **gen_params)
                batch_generated_bs = bspn_outps.sequences.cpu()[:, start_token_idx:].numpy().tolist()

                for dial_idx, generated_bs in enumerate(batch_generated_bs):
                    bspn_gen = self.decode_generated_bspn(generated_bs)
                    constraint = self.reader.bspn_to_constraint_dict(bspn_gen)

                    turn_serv = batch_serv_history[dial_idx][-1]
                    turn_dom = turn_serv.split('_')[0] + ']'
                    turn_offered = batch_turns[dial_idx]['offer']
                    pred_offered = batch_offered[dial_idx]
                    if constraint.get(turn_dom[1:-1]) and \
                            turn_offered.get(turn_serv) and \
                            pred_offered.get(turn_serv):

                        constraint[turn_dom[1:-1]] = self.map_offer_value(constraint[turn_dom[1:-1]],
                                                                          pred_offered[turn_serv],
                                                                          turn_offered[turn_serv])

                    bspn_gen = self.reader.constraint_dict_to_bspn(constraint)
                    bspn_gen = self.reader.encode('<sos_b> ' + bspn_gen + ' <eos_b>')

                    batch_bspn_gen[dial_idx][turn_serv] = bspn_gen
                    # print(self.reader.decode(bspn_gen))

                batch_decoded['bspn'] = []
                for bspns in batch_bspn_gen:
                    decoded_bspn = [b for s, b in bspns.items()]
                    batch_decoded['bspn'].append(decoded_bspn)

                # ###########
                # retrieve db
                if not cfg.use_true_db_pointer:
                    batch_db = []
                    batch_decoded['db'] = []
                    for dial_idx, bspn_gen in enumerate(batch_bspn_gen):
                        turn_serv = batch_serv_history[dial_idx][-1]
                        db_result = self.reader.bspn_to_DBpointer(bspn_gen[turn_serv], turn_serv)
                        db = self.reader.encode('<sos_db> ' + db_result + ' <eos_db>')
                        batch_db.append(db)
                        batch_decoded['db'].append(db)
                    # print(self.reader.batch_decode(batch_decoded['db']))

            # ############
            # predict resp
            decoder_input_ids = []
            for dial_idx in range(num_dial):
                if cfg.use_true_curr_dspn:
                    dom = batch_serv_history[dial_idx][-1].split('_')[0] + ']'
                    decoder_input_ids.append([cfg.pad_token_id] +
                                             batch_db[dial_idx] +
                                             self.reader.encode('<sos_a> ' + dom))
                else:
                    decoder_input_ids.append([cfg.pad_token_id] +
                                             batch_db[dial_idx] +
                                             self.reader.encode('<sos_a>'))

            decoder_input_ids = utils.pad_left(decoder_input_ids, cfg.pad_token_id)

            resp_inps = {'decoder_input_ids': decoder_input_ids,
                         'attention_mask': batch_attention_mask,
                         'decode_resp': True,
                         **batch_context_inps}

            if dial_hidden_states is not None:
                resp_inps['encoder_outputs'] = BaseModelOutput(last_hidden_state=dial_hidden_states)
            else:
                resp_inps['input_ids'] = batch_pad_inputs['dial_contexts']

            resp_inps = utils.to_tensor(resp_inps)
            resp_inps = utils.to_device(resp_inps, module.device)

            gen_params['max_length'] = 120
            # gen_params['eos_token_id'] = self.reader.encode('<eos_r>')[0]
            start_token_idx = resp_inps['decoder_input_ids'].size(-1) - 1

            resp_outps = module.generate(**resp_inps, **gen_params)

            batch_generated_ar = resp_outps.sequences.cpu()[:, start_token_idx:].numpy().tolist()

            batch_decoded['aspn'], batch_decoded['resp'] = [], []
            for dial_idx, generated_ar in enumerate(batch_generated_ar):
                decoded_ar = self.decode_generated_act_resp(generated_ar)

                turn_serv = batch_serv_history[dial_idx][-1]
                turn_dom = turn_serv.split('_')[0] + ']'

                if not batch_offered[dial_idx].get(turn_serv):
                    batch_offered[dial_idx][turn_serv] = {}

                constraint = self.reader.bspn_to_constraint_dict(batch_decoded['bspn'][dial_idx])
                serv_pred_offered = copy.deepcopy(batch_offered[dial_idx][turn_serv])
                serv_pred_offered = self.reader.update_system_offer(constraint[turn_dom] if constraint.get(turn_dom) else {},
                                                                    serv_pred_offered,
                                                                    self.reader.decode(decoded_ar['aspn']))
                batch_offered[dial_idx][turn_serv] = serv_pred_offered

                batch_decoded['aspn'].append(decoded_ar['aspn'])
                batch_decoded['resp'].append(decoded_ar['resp'])

            # print(self.reader.batch_decode(batch_decoded['aspn']))
            # print(self.reader.batch_decode(batch_decoded['resp']))
            # print()

            # ##########################
            # update turn and prev. turn
            for dial_idx in range(num_dial):
                turn = batch_turns[dial_idx]
                pv_turns = batch_pv_turns[dial_idx]

                turn['dspn_gen'] = turn['dspn'] if cfg.use_true_curr_dspn else batch_decoded['dspn'][dial_idx]
                turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else batch_decoded['bspn'][dial_idx]
                turn['db_gen'] = turn['db'] if cfg.use_true_db_pointer else batch_decoded['db'][dial_idx]
                turn['aspn_gen'] = batch_decoded['aspn'][dial_idx]
                turn['resp_gen'] = batch_decoded['resp'][dial_idx]

                pv_turns['hist'] = batch_inputs['hist'][dial_idx]  # all true previous context
                pv_turns['dspn'] = turn['dspn'] if cfg.use_true_prev_dspn else batch_decoded['dspn'][dial_idx]
                pv_turns['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else batch_decoded['bspn'][dial_idx]
                pv_turns['db'] = turn['db'] if cfg.use_true_db_pointer else batch_decoded['db'][dial_idx]
                pv_turns['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else batch_decoded['aspn'][dial_idx]
                pv_turns['resp'] = turn['resp'] if cfg.use_true_prev_resp else batch_decoded['resp'][dial_idx]

                batch_turns[dial_idx] = turn
                batch_pv_turns[dial_idx] = pv_turns

        for dial in dial_batch:
            result_collection.update(self.reader.inverse_transpose_turn(dial))

        return result_collection

    def get_optimizers(self):
        # set up the optimizer and the learning rate scheduler from transformers.Trainer
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

        if cfg.resume:
            checkpoint = torch.load(cfg.opt_path + '/opt.pt')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer

    def report_stats(self, stats):
        stat_txt = 'loss: {:.3f}'.format(stats['loss'])
        stat_txt += ' - dspn: {:.3f}'.format(stats['dspn_loss']) if cfg.train_dspn else ''
        stat_txt += ' - bspn: {:.3f}'.format(stats['bspn_loss']) if cfg.train_bspn else ''
        stat_txt += ' - resp: {:.3f}'.format(stats['resp_loss']) if cfg.train_resp else ''
        stat_txt += ', acc: {:.3f}'.format(stats['acc'] * 100)
        stat_txt += ' - dspn: {:.3f}'.format(stats['dspn_acc'] * 100) if cfg.train_dspn else ''
        stat_txt += ' - bspn: {:.3f}'.format(stats['bspn_acc'] * 100) if cfg.train_bspn else ''
        stat_txt += ' - resp: {:.3f}'.format(stats['resp_acc'] * 100) if cfg.train_resp else ''
        return stat_txt

    def log_first_inputs(self, inputs):
        logging.info('\n---- input examples:')
        logging.info('  dialogue context: {}'.format(self.reader.batch_decode(inputs['dial_contexts'][:2].tolist())))
        logging.info('  dspn label: {}'.format(self.reader.batch_decode(inputs['dspn_lbls'][:2].tolist())))
        logging.info('  bspn label: {}'.format(self.reader.batch_decode(inputs['bspn_lbls'][:2].tolist())))
        logging.info('  resp label: {}'.format(self.reader.batch_decode(inputs['resp_lbls'][:2].tolist())))
        for i, (contexts, lbls) in enumerate(zip(inputs['slot_contexts'][:2], inputs['slot_lbls'][:2])):
            decoded_c = self.reader.batch_decode(contexts[0])
            decoded_l = self.reader.decode(lbls[0]).split()
            logging.info('  slot context {}: {}'.format(i+1, [' '.join([tkn for tkn in slot.split() if tkn != '<pad>'])
                                                              for slot in decoded_c]))
            logging.info('  slot label {}: {}'.format(i+1, ' '.join([slot for slot in decoded_l if slot != '<pad>'])))

    def average_loss(self, loss, labels):
        not_ignore = labels.ne(cfg.pad_token_id)
        num_targets = not_ignore.long().sum().item()
        return loss.sum() / num_targets

    def count_tokens(self, lm_logits, labels):

        pred_lm = torch.argmax(lm_logits, dim=-1)
        pred_lm = pred_lm.view(-1)

        labels = labels.view(-1)

        num_count = pred_lm.ne(cfg.pad_token_id).long().sum()
        num_correct = torch.eq(pred_lm, labels).long().sum()

        return num_correct, num_count

    def calculate_accuracy(self, stats):
        acc = stats['dspn_correct'] + stats['bspn_correct'] + stats['resp_correct']
        acc = acc / (stats['dspn_count'] + stats['bspn_count'] + stats['resp_count'])

        dspn_acc = stats['dspn_correct'] / stats['dspn_count'] if stats['dspn_count'] > 0. else 0.
        bspn_acc = stats['bspn_correct'] / stats['bspn_count'] if stats['bspn_count'] > 0. else 0.
        resp_acc = stats['resp_correct'] / stats['resp_count'] if stats['resp_count'] > 0. else 0.

        return acc, dspn_acc, bspn_acc, resp_acc

    def save_checkpoint(self, epoch, optimizer):
        save_path = os.path.join(cfg.experiment_path, 'epoch{}'.format(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logging.info('saving checkpoint to {}'.format(save_path))

        model = self.model.module if cfg.multi_gpu else self.model
        model.save_pretrained(save_path)
        self.reader.tokenizer.save_pretrained(save_path)

        if cfg.save_optimizer:
            torch.save({'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(cfg.experiment_path, 'opt.pt'))

    def decode_generated_act_resp(self, generated):
        decoded = {'aspn': [], 'resp': []}

        # remove eos_token
        if cfg.eos_token_id in generated:
            generated = generated[: generated.index(cfg.eos_token_id)]

        # keep only aspn (cut until eos_a_token)
        eos_a_id = self.reader.encode('<eos_a>')[0]
        if eos_a_id in generated:
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx + 1]
            generated = generated[eos_a_idx + 1:]

        sos_r_id = self.reader.encode('<sos_r>')[0]
        if sos_r_id in generated:
            sos_r_idx = generated.index(sos_r_id)

            if not decoded['aspn']:
                decoded['aspn'] = generated[: sos_r_idx]
                decoded['aspn'] += [eos_a_id]

            generated = generated[sos_r_idx:]

        # keep only resp (cut until eos_r_token)
        eos_r_id = self.reader.encode('<eos_r>')[0]
        eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated) - 1
        decoded['resp'] = generated[: eos_r_idx + 1]

        return decoded

    def decode_generated_bspn(self, generated):
        # remove eos_token
        if cfg.eos_token_id in generated:
            generated = generated[: generated.index(cfg.eos_token_id)]

        # keep only bspn (cut until eos_b_token)
        eos_b_id = self.reader.encode('<eos_b>')[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
            generated = generated[: eos_b_idx + 1]

        return generated

    def decode_generated_dspn(self, generated):
        # remove eos_token
        if cfg.eos_token_id in generated:
            generated = generated[: generated.index(cfg.eos_token_id)]

        # keep only bspn (cut until eos_d_token)
        eos_d_id = self.reader.encode('<eos_d>')[0]
        if eos_d_id in generated:
            eos_d_idx = generated.index(eos_d_id)
            generated = generated[: eos_d_idx + 1]

        return generated

    def map_offer_value(self, serv_constraint, serv_pred_offered, serv_true_offered):
        for slot in serv_pred_offered:
            if not serv_constraint.get(slot[1:-1]):
                continue
            if not serv_true_offered.get(slot):
                continue
            serv_constraint[slot[1:-1]] = serv_true_offered[slot]
        return serv_constraint

    def get_turn_domain(self, domains):
        return domains[-1] if domains else ''


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
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    cfg.resume = args.resume

    parse_arg_cfg(args)
    cfg.construct_paths()

    if not os.path.exists(cfg.data_path):
        os.makedirs(cfg.data_path)
    if args.mode in ['train', 'finetune'] and not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)

    if args.mode == 'test' or args.mode == 'dev' or cfg.resume:
        if not cfg.inference_path:
            raise Exception("require 'inference_path'")

        cfg.inference_epoch = int(cfg.inference_path.split('epoch')[-1])

        cfg.model_path = cfg.inference_path
        cfg.tok_path = cfg.inference_path
        cfg.opt_path = '/'.join(cfg.inference_path.split('/')[:-1])

        cfg.resume_from_epoch = cfg.inference_epoch

    elif args.mode == 'finetune':
        cfg.model_path = cfg.inference_path
        cfg.tok_path = cfg.inference_path
        cfg.inference_path = cfg.experiment_path
        cfg.resume_from_epoch = 0

    else:  # train from the first epoch
        cfg.inference_path = cfg.experiment_path
        cfg.resume_from_epoch = 0

    cfg.init_logging_handler(args.mode)
    if cfg.cuda:
        device = torch.device('cuda')
        cfg.cuda_device = [i for i in range(torch.cuda.device_count())]
        cfg.multi_gpu = False if len(cfg.cuda_device) == 1 else True
    else:
        device = torch.device('cpu')
        cfg.cuda_device = [0]
        cfg.multi_gpu = False
    logging.info('run on {}{}'.format(len(cfg.cuda_device) if cfg.cuda else '',
                                      ' gpu(s)' if cfg.cuda else 'cpu'))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    trainer = Trainer(device)

    if args.mode in ['train', 'finetune']:
        trainer.train()
    else:  # test
        logging.info('generate setting: \n\t' +
                     ' use true_prev_dspn = {} \n\t'.format(cfg.use_true_prev_dspn) +
                     ' use true_prev_bspn = {} \n\t'.format(cfg.use_true_prev_bspn) +
                     ' use true_db_pointer = {} \n\t'.format(cfg.use_true_db_pointer) +
                     ' use true_prev_aspn = {} \n\t'.format(cfg.use_true_prev_aspn) +
                     ' use true_prev_resp = {} \n\t'.format(cfg.use_true_prev_resp) +
                     ' use true_curr_dspn = {} \n\t'.format(cfg.use_true_curr_dspn) +
                     ' use true_curr_bspn = {} \n\t'.format(cfg.use_true_curr_bspn) +
                     # ' use true_curr_aspn = {} \n\t'.format(cfg.use_true_curr_aspn) +
                     ' use_all_previous_context = {}'.format(cfg.use_all_previous_context)
                     )
        logging.info('load a model from: ' + cfg.inference_path)

        if args.mode == 'dev':
            trainer.inference()
        elif args.mode == 'test':
            trainer.inference('test')


#  train all: python train.py -mode train -cfg gradient_accumulation_steps=4 batch_size=4 epoch_num=1 exp_no=trial low_resource=True cuda=False validate_during_training=False
#  train except X domain: python train.py -mode train -cfg gradient_accumulation_steps=4 batch_size=4 epoch_num=1 exp_no=trial low_resource=True cuda=False exp_domains=events validate_during_training=False
#  finetune on F domain: python train.py -mode finetune -cfg gradient_accumulation_steps=4 batch_size=4 epoch_num=1 exp_no=trial low_resource=True cuda=False ft_domains=events validate_during_training=False inference_path=experiments-Xdomain/events_trial_bs4_ga4/epoch1
#  test on (train all): python train.py -mode test -cfg batch_size=4 inference_path=experiments/all_trial_bs4_ga4/epoch1 low_resource=True cuda=False
#  test on (train all) with true dspn: python train.py -mode test -cfg batch_size=4 inference_path=experiments/all_trial_bs4_ga4/epoch1 low_resource=True cuda=False use_true_curr_dspn=True use_true_prev_dspn=True
#  test on (train except X domain): python train.py -mode test -cfg batch_size=4 inference_path=experiments-Xdomain/events_trial_bs4_ga4/epoch1 low_resource=True cuda=False
#  test on (finetune on F domain): python train.py -mode test -cfg batch_size=4 inference_path=experiments-Fdomain/events_trial_bs4_ga4/epoch1 low_resource=True cuda=False

if __name__ == "__main__":
    main()
