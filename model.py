import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map


class T5WithSchemaDependency(T5ForConditionalGeneration):
    def __init__(self, config, domain_size, slot_size):
        super(T5WithSchemaDependency, self).__init__(config)
        config.context_size = domain_size + slot_size
        config.slot_size = slot_size

        # new decoders
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.bspn_decoder = type(self.decoder)(decoder_config, self.shared)
        self.bspn_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))

        self.encoder.parallelize(self.device_map)

        self.decoder.parallelize(self.device_map)
        self.bspn_decoder.parallelize(self.device_map)
        self.resp_decoder.parallelize(self.device_map)

        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.bspn_lm_head = self.bspn_lm_head.to(self.decoder.first_device)
        self.resp_lm_head = self.resp_lm_head.to(self.decoder.first_device)

        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()

        self.decoder.deparallelize()
        self.bspn_decoder.deparallelize()
        self.resp_decoder.deparallelize()

        self.encoder = self.encoder.to("cpu")

        self.decoder = self.decoder.to("cpu")
        self.bspn_decoder = self.bspn_decoder.to("cpu")
        self.resp_decoder = self.resp_decoder.to("cpu")

        self.lm_head = self.lm_head.to("cpu")
        self.bspn_lm_head = self.bspn_lm_head.to("cpu")
        self.resp_lm_head = self.resp_lm_head.to("cpu")

        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def initialize_additional_decoder(self):
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.bspn_decoder = type(self.decoder)(decoder_config, self.shared)
        self.bspn_decoder.load_state_dict(self.decoder.state_dict())

        self.bspn_lm_head = type(self.lm_head)(self.config.d_model, self.config.vocab_size, bias=False)
        self.bspn_lm_head.load_state_dict(self.lm_head.state_dict())

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_decoder.load_state_dict(self.decoder.state_dict())

        self.resp_lm_head = type(self.lm_head)(self.config.d_model, self.config.vocab_size, bias=False)
        self.resp_lm_head.load_state_dict(self.lm_head.state_dict())

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        self.bspn_decoder.set_input_embeddings(new_embeddings)
        self.resp_decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.bspn_lm_head = new_embeddings
        self.resp_lm_head = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                input_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                context_ids=None,  # new
                context_embs=None,  # new
                context_labels=None,  # new
                lm_mask_labels=None,  # new
                encode_context=None,  # new
                decode_bspn=None,  # new
                decode_resp=None,  # new
                ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # slot encoding (for belief, system action and response decoding)
        if encode_context and context_ids is not None and context_embs is None:
            batch_size, batch_n_c, batch_n_c_tkn = context_ids.size()
            context_input_ids = context_ids.contiguous().view(-1, batch_n_c_tkn)
            context_attention_mask = torch.where(context_input_ids == self.config.pad_token_id, 0, 1)
            context_enc_outps = self.encoder(input_ids=context_input_ids,
                                             attention_mask=context_attention_mask,
                                             output_hidden_states=output_hidden_states,
                                             return_dict=return_dict)
            # avg. of context token vectors from the last encoder layer
            # context_embs = context_enc_outps[0].mean(dim=1).view(batch_size, batch_n_c, -1)

            # first index which represents slot token
            # context_embs = context_enc_outps[0][:, 0].view(batch_size, batch_n_c, -1)

            # last index which represents slot token
            context_embs = context_enc_outps[0][:, -1].view(batch_size, batch_n_c, -1)

            return context_embs

        # dialogue encoding
        if encoder_outputs is None:
            dial_enc_outps = self.encoder(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          return_dict=return_dict)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            dial_enc_outps = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        else:
            dial_enc_outps = encoder_outputs

        dial_hidden_states = dial_enc_outps[0]

        batch_size, batch_dial_tkn, _ = dial_hidden_states.size()

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None:  # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            dial_hidden_states = dial_hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)

        if decode_bspn:  # dialogue state tracking (bspn)
            decoder = self.bspn_decoder
            lm_head = self.bspn_lm_head
        elif decode_resp:  # response (db, action, response)
            decoder = self.resp_decoder
            lm_head = self.resp_lm_head
        else:  # domain (dspn)
            decoder = self.decoder
            lm_head = self.lm_head

        if context_embs is not None:
            dial_hidden_states = torch.cat((context_embs, dial_hidden_states), dim=1)

        next_dial_dec_outps = decoder(input_ids=decoder_input_ids,
                                      encoder_hidden_states=dial_hidden_states,
                                      past_key_values=past_key_values,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict,
                                      use_cache=use_cache,
                                      output_attentions=True
                                      )
        next_dial_hidden_states = next_dial_dec_outps[0]

        # set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            lm_head = lm_head.to(self.encoder.first_device)
            next_dial_hidden_states = next_dial_hidden_states.to(lm_head.weight.device)

        batch_next_dial_tkn = next_dial_hidden_states.size(1)

        if self.config.tie_word_embeddings:
            # rescale output before projecting on vocab
            next_dial_hidden_states = next_dial_hidden_states * (self.model_dim ** -0.5)

        lm_logits = lm_head(next_dial_hidden_states)

        start_domain_id = self.config.vocab_size - self.config.context_size
        lm_mask = torch.ones(batch_size, self.config.vocab_size)
        if lm_mask_labels is not None:
            lm_mask = torch.sum(F.one_hot(lm_mask_labels, num_classes=self.config.vocab_size), dim=1)
            lm_mask[:, :start_domain_id] = 1.  # no mask for generic tokens

        lm_mask = lm_mask.unsqueeze(1).repeat_interleave(batch_next_dial_tkn, dim=1)
        lm_logits[~lm_mask.bool()] = lm_logits[~lm_mask.bool()] + float("-inf")  # masking irrelevant tokens

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="none")
            loss = loss_fct(lm_logits.contiguous().view(-1, lm_logits.size(-1)), labels.contiguous().view(-1))

        if not return_dict:
            output = (lm_logits,) + next_dial_dec_outps[1:] + (dial_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=next_dial_dec_outps.past_key_values,
            decoder_hidden_states=next_dial_dec_outps.hidden_states,
            decoder_attentions=next_dial_dec_outps.attentions,
            cross_attentions=next_dial_dec_outps.cross_attentions,
            encoder_last_hidden_state=dial_enc_outps.last_hidden_state,
            encoder_hidden_states=dial_enc_outps.hidden_states,
            encoder_attentions=dial_enc_outps.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self,
                                                       inputs_tensor,
                                                       model_kwargs,
                                                       model_input_name=None):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache",
                             "context_ids", "context_embs", "context_labels", "lm_mask_labels",
                             "encode_context", "decode_bspn", "decode_resp"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      encoder_outputs=None,
                                      past_key_values=None,
                                      **kwargs):

        return {"decoder_input_ids": input_ids,
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "context_ids": kwargs.get("context_ids"),
                "context_embs": kwargs.get("context_embs"),
                "context_labels": kwargs.get("context_labels"),
                "lm_mask_labels": kwargs.get("lm_mask_labels"),
                "encode_context": kwargs.get("encode_context"),
                "decode_bspn": kwargs.get("decode_bspn"),
                "decode_resp": kwargs.get("decode_resp")
                }

