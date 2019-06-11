#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:43:38 2017

@author: jiahuei

Differences:
    1. Added attention map dropout
    2. Possible RNN variational dropout
    3. Possible context layer
    4. Changed RNN init method
    5. Changed training scheme (LR, ADAM epsilon)


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, math
import tensorflow as tf
from tensorflow.python.layers.core import Dense
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..', 'common'))
#from natural_sort import natural_keys
from nets import nets_factory
import ops
import ops_rnn as rops
_shape = ops.shape
pjoin = os.path.join


class ModelBase(object):
    """
    Base for model implementations.
    """
    
    def __init__(self, config):
        self._config = c = config
        assert c.token_type in ['radix', 'word', 'char']
        if c.token_type == 'radix':
            self._softmax_size = c.radix_base + 2
        else:
            self._softmax_size = len(c.itow)
    
    
    # TODO: Bookmark
    #############################################
    # Encoder & Decoder functions               #
    #############################################
    
    ###############
    ### Encoder ###
    
    def _encoder(self):
        """
        Encoder CNN.
        
        Builds image CNN model specified by config.cnn_name.
        
        Setups and returns the following:
        self.im_embed: A tensor of shape [batch, image_embed_size].
        self.cnn_fmaps: A list of feature maps specified by 
            config.cnn_fm_attention, each of shape 
            [batch, map_height * map_width, channels].
        """
        c = self._config
        
        # Select the CNN
        with tf.variable_scope('cnn'):
            cnn_fn = nets_factory.get_network_fn(
                            c.cnn_name,
                            num_classes=None,
                            weight_decay=0.0,
                            is_training=False)
            net, end_points = cnn_fn(self._enc_inputs)
        
        # Produce image embeddings
        if c.legacy:
            net = ops.layer_norm_activate(
                            scope='LN_tanh',
                            inputs=tf.squeeze(net, axis=[1, 2]),
                            activation_fn=tf.nn.tanh,
                            begin_norm_axis=1)
            self.im_embed = ops.linear(
                                scope='im_embed',
                                inputs=net,
                                output_dim=1024,
                                bias_init=None,
                                activation_fn=None)
        else:
            self.im_embed = tf.squeeze(net, axis=[1, 2])
        
        # Feature maps
        # Reshape CNN feature map for RNNs
        # Must have fully defined inner dims
        cnn_fm = end_points[c.cnn_fm_attention]
        fm_ds = tf.shape(cnn_fm)                            # (n, h, w, c)
        fm_ss = _shape(cnn_fm)
        fm_s = tf.stack([fm_ss[0], fm_ds[1] * fm_ds[2], fm_ss[3]], axis=0)
        cnn_fm = tf.reshape(cnn_fm, fm_s)
        self.cnn_fmaps = cnn_fm
        return self.im_embed, cnn_fm
    
    ###############
    ### Decoder ###
    
    def _decoder_rnn(self):
        """
        RNN Decoder.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        im_embed = self.im_embed
        cnn_fmaps = self.cnn_fmaps
        
        align = c.attn_alignment_method
        prob = c.attn_probability_fn
        rnn_size = c.rnn_size
        att_keep_prob = c.attn_keep_prob if self.is_training() else 1.0
        batch_size = _shape(im_embed)[0]
        is_inference = self.mode == 'infer'
        beam_search = (is_inference and c.infer_beam_size > 1)
        
        if beam_search:
            beam_size = c.infer_beam_size
            # Tile the batch dimension in preparation for Beam Search
            im_embed = tf.contrib.seq2seq.tile_batch(im_embed, beam_size)
            cnn_fmaps = tf.contrib.seq2seq.tile_batch(cnn_fmaps, beam_size)
        
        if align == 'add_LN':
            att_mech = rops.MultiHeadAddLN
        elif align == 'dot':
            att_mech = rops.MultiHeadDot
        else:
            raise ValueError('Invalid alignment method.')
        
        if prob == 'softmax':
            prob_fn = tf.nn.softmax
        elif prob == 'sigmoid':
            prob_fn = self._signorm
        else:
            raise ValueError('Invalid alignment method.')
        
        with tf.variable_scope('rnn_decoder'):
            cell = self._get_rnn_cell(rnn_size)
            rnn_init = self._get_rnn_init(im_embed, cell)
            cnn_attention = att_mech(
                                    rnn_size,
                                    cnn_fmaps,
                                    c.cnn_fm_projection,
                                    c.attn_num_heads,
                                    memory_sequence_length=None,
                                    probability_fn=prob_fn)
            attention_cell = rops.MultiHeadAttentionWrapperV3(
                                        deep_output_layer=False,
                                        context_layer=c.attn_context_layer,
                                        alignments_keep_prob=att_keep_prob,
                                        cell=cell,
                                        attention_mechanism=cnn_attention,
                                        attention_layer_size=None,
                                        alignment_history=True,
                                        cell_input_fn=None,
                                        output_attention=False,
                                        initial_cell_state=rnn_init)
            
            self._build_word_projections()
            embeddings = self._get_embedding_var_or_fn(batch_size)
            
            rnn_raw_outputs = self._rnn_dynamic_decoder(
                                                attention_cell,
                                                embeddings,
                                                self.decoder_output_layer)
        
        with tf.name_scope('post_processing'):
            logits, output_ids, attn_maps = self._decoder_post_process(
                                                            rnn_raw_outputs,
                                                            top_beam=True)
        self.dec_preds = output_ids
        self.dec_logits = logits
        self.dec_attn_maps = attn_maps
        return logits, output_ids, attn_maps
    
    
    def _decoder_rnn_scst(self, beam_size=0):
        """
        RNN Decoder for SCST training.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        im_embed = self.im_embed
        cnn_fmaps = self.cnn_fmaps
        
        align = c.attn_alignment_method
        prob = c.attn_probability_fn
        rnn_size = c.rnn_size
        att_keep_prob = c.attn_keep_prob if self.is_training() else 1.0
        batch_size = _shape(im_embed)[0]
        
        sample = False
        if not self.is_training():
            if beam_size == 0:
                sample = True
            else:
                # Prepare beam search to sample candidates
                c.batch_size_infer = batch_size
                c.infer_beam_size = beam_size
                c.infer_max_length = 20
                c.infer_length_penalty_weight = 0
                # Tile the batch dimension in preparation for Beam Search
                im_embed = tf.contrib.seq2seq.tile_batch(im_embed, beam_size)
                cnn_fmaps = tf.contrib.seq2seq.tile_batch(cnn_fmaps, beam_size)
        
        if align == 'add_LN':
            att_mech = rops.MultiHeadAddLN
        elif align == 'dot':
            att_mech = rops.MultiHeadDot
        else:
            raise ValueError('Invalid alignment method.')
        
        if prob == 'softmax':
            prob_fn = tf.nn.softmax
        elif prob == 'sigmoid':
            prob_fn = self._signorm
        else:
            raise ValueError('Invalid alignment method.')
        
        with tf.variable_scope('rnn_decoder'):
            cell = self._get_rnn_cell(rnn_size)
            rnn_init = self._get_rnn_init(im_embed, cell)
            cnn_attention = att_mech(
                                    rnn_size,
                                    cnn_fmaps,
                                    c.cnn_fm_projection,
                                    c.attn_num_heads,
                                    memory_sequence_length=None,
                                    probability_fn=prob_fn)
            attention_cell = rops.MultiHeadAttentionWrapperV3(
                                        deep_output_layer=False,
                                        context_layer=c.attn_context_layer,
                                        alignments_keep_prob=att_keep_prob,
                                        cell=cell,
                                        attention_mechanism=cnn_attention,
                                        attention_layer_size=None,
                                        alignment_history=True,
                                        cell_input_fn=None,
                                        output_attention=False,
                                        initial_cell_state=rnn_init)
            
            self._build_word_projections()
            embeddings = self._get_embedding_var_or_fn(batch_size)
            
            rnn_raw_outputs = self._rnn_dynamic_decoder(
                                                attention_cell,
                                                embeddings,
                                                self.decoder_output_layer,
                                                sample=sample)
        
        with tf.name_scope('post_processing'):
            logits, output_ids, attn_maps = self._decoder_post_process(
                                                            rnn_raw_outputs,
                                                            top_beam=False)
        self.dec_preds = output_ids
        self.dec_logits = logits
        self.dec_attn_maps = attn_maps
        return logits, output_ids, attn_maps
    
    
    def _decoder_post_process(self, rnn_raw_outputs, top_beam=True):
        c = self._config
        is_inference = self.mode == 'infer'
        beam_search = (is_inference and len(_shape(rnn_raw_outputs[0])) > 2)
        
        if beam_search:
            predicted_ids, scores, dec_states = rnn_raw_outputs                 # (time, batch_size, beam_size)
            if top_beam:
                # Beams are sorted from best to worst according to prob
                top_sequence = predicted_ids[:, :, 0]
                top_score = scores[:, :, 0] 
                # (batch_size, seq_len)
                output_ids = tf.transpose(top_sequence, [1, 0])
                logits = tf.transpose(top_score, [1, 0])
            else:
                output_ids = tf.transpose(predicted_ids, [2, 1, 0])             # (beam_size, batch_size, time)
                logits = tf.transpose(scores, [2, 1, 0])
        else:
            output_ids, logits, dec_states = rnn_raw_outputs
            # (batch_size, seq_len, softmax_size)
            logits = tf.transpose(logits, [1, 0, 2])                        
            output_ids = tf.transpose(output_ids, [1, 0])
        
        ## Attention Map ##
        attn_map = dec_states.alignment_history
        # (seq_len, batch * beam, fm_size)
        if beam_search:
            assert not isinstance(attn_map, tf.TensorArray)
            beam_size = c.infer_beam_size
            # Select top beam
            map_s = tf.shape(attn_map)
            map_s = tf.stack([map_s[0], -1, beam_size, map_s[2]], axis=0)
            attn_map = tf.reshape(attn_map, map_s)
            attn_map = attn_map[:, :, 0, :]
            # (seq_len, batch, fm_size)
        else:
            attn_map = attn_map.stack()
        # Retrieve the attention maps (seq_len, batch, num_heads * fm_size)
        map_s = tf.shape(attn_map)
        map_s = tf.stack([map_s[0], map_s[1], c.attn_num_heads, -1], axis=0)
        attn_map = tf.reshape(attn_map, map_s)                # (seq_len, batch, num_heads, fm_size)
        attn_map = tf.transpose(attn_map, [1, 2, 0, 3])       # (batch, num_heads, seq_len, fm_size)
        return logits, output_ids, attn_map
    
    
    # TODO: Bookmark
    #############################################
    # Loss & Training & Restore functions       #
    #############################################
    
    #############################
    ### Loss fns & Optimisers ###
    
    def _train_caption_model(self, scst=False):
        """
        Calculates the average log-perplexity per word, and also the
        doubly stochastic loss of attention map.
        """
        if self.mode == 'infer': return None
        c = self._config
        
        with tf.name_scope('loss_decoder'):
            ### Sequence / Reconstruction loss ###
            with tf.name_scope('decoder'):
                if scst is False:
                    dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                                            logits=self.dec_logits,
                                            targets=self._dec_sent_targets,
                                            weights=self._dec_sent_masks)
                else:
                    dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                                            logits=self.dec_logits,
                                            targets=self._dec_sent_targets,
                                            weights=self._dec_sent_masks,
                                            average_across_batch=False)
                    dec_log_ppl = tf.reduce_mean(dec_log_ppl * self.rewards)
                tf.summary.scalar('loss', dec_log_ppl)
                tf.summary.scalar('perplexity', tf.exp(dec_log_ppl))
                self.dec_log_ppl = dec_log_ppl
            
            if not self.is_training():
                return dec_log_ppl
            
            # Attention map doubly stochastic loss
            map_loss = .0
            if c.rnn_map_loss_scale > 0:
                with tf.name_scope('attention_map'):
                    # Sum along time dimension
                    flat_cnn_maps = tf.reduce_sum(self.dec_attn_maps, axis=1)   # (batch, fm_size)
                    map_loss = tf.squared_difference(1.0, flat_cnn_maps)
                    map_loss = tf.reduce_mean(map_loss)
                    tf.summary.scalar('loss', map_loss)
                    map_loss *= c.rnn_map_loss_scale
                    tf.summary.scalar('loss_weighted', map_loss)
            
            # Maybe L2 regularisation
            tvars = self._get_trainable_vars()
            tvars_cnn = tf.contrib.framework.filter_variables(
                                    var_list=tvars,
                                    include_patterns=['Model/encoder/cnn'],
                                    exclude_patterns=None,
                                    reg_search=True)
            tvars_dec = tf.contrib.framework.filter_variables(
                                    var_list=tvars,
                                    include_patterns=['Model'],
                                    exclude_patterns=['Model/encoder/cnn'],
                                    reg_search=True)
            assert len(tvars) == len(tvars_cnn + tvars_dec)
            reg_loss = self._loss_regularisation(tvars)
            
            # Add losses
            loss = dec_log_ppl + map_loss + reg_loss
            tf.summary.scalar('total_loss', loss)
        
        # Training op for captioning model
        with tf.variable_scope('optimise/caption'):
            if c.cnn_grad_multiplier != 1.0:
                multipliers = dict(
                    zip(tvars_cnn, [c.cnn_grad_multiplier] * len(tvars_cnn)) +
                    zip(tvars_dec, [1.0] * len(tvars_dec)))
            else:
                multipliers = None
            train_cap = tf.contrib.slim.learning.create_train_op(
                            loss,
                            self._get_optimiser(self.lr, momentum=None),
                            global_step=self.global_step,
                            variables_to_train=tvars,
                            clip_gradient_norm=c.clip_gradient_norm,
                            summarize_gradients=c.add_grad_summaries,
                            gradient_multipliers=multipliers)
            
            with tf.control_dependencies([train_cap]):
                self.dec_log_ppl = tf.identity(self.dec_log_ppl)
        return self.dec_log_ppl
    
    
    def _loss_regularisation(self, var_list):
        """ Add L2 regularisation. """
        c = self._config
        loss = .0
        if c.l2_decay > 0:
            with tf.name_scope('regularisation'):
                for var in var_list:
                    loss += ops.l2_regulariser(var, c.l2_decay)
        tf.summary.scalar('regularisation_loss', loss)
        return loss
    
    #################
    ### Restoring ###
    
    def restore_model(self, session, saver, lr):
        """
        Helper function to restore model variables.    
        """
        c = self._config
        print('\n')
        
        if not c.checkpoint_path:
            print('INFO: Training entire model from scratch.')
        else:
            if os.path.isfile(c.checkpoint_path + '.index')  \
            or os.path.isfile(c.checkpoint_path):                               # V2 & V1 checkpoint
                checkpoint_path = c.checkpoint_path
            else:
                checkpoint_path = tf.train.latest_checkpoint(c.checkpoint_path)
            
            ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
            ckpt_vars = set(ckpt_reader.get_variable_to_shape_map().keys())
            model_vars = set([v.op.name for v in self._get_trainable_vars()])
            if c.checkpoint_exclude_scopes != '':
                exc_scopes = [sc.strip()
                            for sc in c.checkpoint_exclude_scopes.split(',')]
            else:
                exc_scopes = None
                        
            if model_vars.issubset(ckpt_vars):
                if exc_scopes == None and c.resume_training:
                    # Restore whole model (resume training)
                    saver.restore(session, checkpoint_path)
                    print('INFO: Resume training from checkpoint: {}'.format(
                        checkpoint_path))
                else:
                    # Restore whole model (fine-tune)
                    var_list = tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES,
                                scope='Model')
                    var_list = tf.contrib.framework.filter_variables(
                                var_list=var_list,
                                include_patterns='Model',
                                exclude_patterns=exc_scopes,
                                reg_search=True)
                    _saver = tf.train.Saver(var_list)
                    _saver.restore(session, checkpoint_path)
                    print('INFO: Restored `Model` from checkpoint: {}'.format(
                        checkpoint_path))
            else:
                # Restore CNN model
                cnn_scope = 'Model/encoder/cnn/'
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=cnn_scope)
                var_list = tf.contrib.framework.filter_variables(
                                var_list=var_list,
                                include_patterns=cnn_scope,
                                exclude_patterns=exc_scopes,
                                reg_search=True)                                # Use re.search
                var_name = [v.op.name.replace(cnn_scope, '') for v in var_list]
                cnn_variables = dict(zip(var_name, var_list))
                cnn_saver = tf.train.Saver(cnn_variables)
                cnn_saver.restore(session, checkpoint_path)
                print('INFO: Restored CNN model from checkpoint {}'.format(
                        checkpoint_path))
         
        if self.is_training():
            if lr is None:
                lr = session.run(self.lr)
            else:
                self.update_lr(session, lr)
                session.run(self.lr)
        return lr
    
    
    # TODO: Bookmark
    #############################################
    # Model helper functions                    #
    #############################################
    
    #####################
    ### Input helpers ###
            
    def _process_inputs(self):
        """
        Generates the necessary inputs, targets, masks.
        """
        c = self._config
        self._enc_inputs = self.batch_ops[0]
        
        if self.mode == 'infer':
            self._dec_sent_lens = None
        else:
            _dec_sent = self.batch_ops[1]           # Decoder sentences (word IDs)
            
            #self._dec_sent_masks = tf.sign(tf.to_float(_dec_sent[:, :-1] + 1))  # Exclude <EOS>
            self._dec_sent_masks = tf.sign(tf.to_float(_dec_sent[:, 1:] + 1))   # Exclude <GO>
            self._dec_sent_lens = tf.reduce_sum(self._dec_sent_masks, axis=1)
            self._dec_sent_lens = tf.to_int32(self._dec_sent_lens)
            
            if c.token_type == 'word':
                # Clip padding values at zero
                _dec_sent = tf.maximum(_dec_sent, 0)
                self._dec_sent_inputs = _dec_sent[:, :-1]
            else:
                # One-hot outputs zero for negative indices
                # We convert to one-hot so that baseN calculation can
                # take place entirely on GPU, since the vocab size is small
                self._dec_sent_inputs = _dec_sent[:, :-1]
                _dec_sent = tf.maximum(_dec_sent, 0)
            self._dec_sent_targets = _dec_sent[:, 1:]
    
    
    def _build_word_projections(self):
        """Helper to update word embedding and output projection variables."""
        c = self._config
        rnn_size = c.rnn_size
        word_size = c.rnn_word_size
        softmax_size = self._softmax_size
        token_type = c.token_type
        place_var_on_cpu = token_type == 'word'
        
        #with tf.variable_scope('decoder/rnn_decoder', reuse=tf.AUTO_REUSE):
        dec_out_layer = Dense(softmax_size, name='output_projection')
        dec_out_layer.build(rnn_size)
        self.decoder_output_layer = dec_out_layer
        print('INFO: Building separate embedding matrix.')
        kwargs = dict(name='embedding_map',
                      shape=[softmax_size, word_size],
                      dtype=tf.float32,
                      trainable=True)
        if place_var_on_cpu:
            with tf.device('/cpu:0'):
                self._word_embed_map = tf.get_variable(**kwargs)
        else:
            self._word_embed_map = tf.get_variable(**kwargs)
        return self._word_embed_map
    
    
    def _get_embedding_var_or_fn(self, batch_size):
        c = self._config
        token_type = c.token_type
        word_size = c.rnn_word_size
        softmax_size = self._softmax_size
        is_inference = self.mode == 'infer'
        beam_search = (is_inference and c.infer_beam_size > 1)
        
        if token_type == 'word':
            if is_inference:
                embeddings = self._word_embed_map
            else:
                with tf.device('/cpu:0'):
                    embeds = tf.nn.embedding_lookup(
                                self._word_embed_map, self._dec_sent_inputs)    # (batch_size, seq_len, word_size)
                embeddings = tf.transpose(embeds, [1, 0, 2])                    # (seq_len, batch_size, word_size)
        else:
            if is_inference:
                # As the softmax size is small, we perform matmul on gpu
                # instead of embedding_lookup on cpu to speed up the operation
                def _embed_fn(ids):
                    ids = tf.one_hot(ids, softmax_size, dtype=tf.float32)
                    ids = tf.reshape(ids, [-1, softmax_size])
                    res = tf.matmul(ids, self._word_embed_map)
                    if beam_search:
                        return tf.reshape(res, [batch_size, -1, word_size])
                    else:
                        return tf.reshape(res, [batch_size, word_size])
                embeddings = _embed_fn
            else:
                dec_inputs = tf.one_hot(self._dec_sent_inputs,
                                        self._softmax_size,
                                        dtype=tf.float32)
                dec_inputs = tf.reshape(dec_inputs, [-1, softmax_size])         # (batch * time, softmax_size)
                embeds = tf.matmul(dec_inputs, self._word_embed_map)
                embeds = tf.reshape(embeds, [batch_size, -1, word_size])
                embeddings = tf.transpose(embeds, [1, 0, 2])                    # (max_time, batch_size, word_size)
        return embeddings
    
    ###################
    ### RNN helpers ###
    
    def _signorm(self, tn):
        with tf.variable_scope('sig_norm'):
            tn = tf.nn.sigmoid(tn)
            tn_sum = tf.reduce_sum(tn, axis=-1, keepdims=True)
            return tn / tn_sum
    
    
    def _get_rnn_cell(self, rnn_size):
        """Helper to select RNN cell(s)."""
        c = self._config
        rnn = c.rnn_name
        
        if c.cnn_fm_projection is None and c.attn_context_layer is False:
            attn_size = _shape(self.cnn_fmaps)[-1]
        else:
            attn_size = c.rnn_size
        self._rnn_input_size = c.rnn_word_size + attn_size
        
        if rnn == 'LSTM':
            cells = tf.contrib.rnn.BasicLSTMCell(
                                        num_units=rnn_size,
                                        state_is_tuple=True,
                                        reuse=self.reuse)
        elif rnn == 'LN_LSTM':
            cells = tf.contrib.rnn.LayerNormBasicLSTMCell(
                                        num_units=rnn_size,
                                        reuse=self.reuse)
        elif rnn == 'GRU':
            cells = tf.contrib.rnn.GRUCell(
                                        num_units=rnn_size,
                                        reuse=self.reuse)
        else:
            raise ValueError('Only `LSTM`, `LN_LSTM` and `GRU` are accepted.')
        if c.rnn_layers > 1:
            raise ValueError('RNN layer > 1 not implemented.')
            #cells = tf.contrib.rnn.MultiRNNCell([cells] * self.config.num_layers)
        
        # Setup input and output dropouts
        input_keep = 1 - c.dropout_rnn_in
        output_keep = 1 - c.dropout_rnn_out
        if self.is_training() and (input_keep < 1 or output_keep < 1):
            print('INFO: Training using dropout.')
            cells = tf.contrib.rnn.DropoutWrapper(
                                    cells,
                                    input_keep_prob=input_keep,
                                    output_keep_prob=output_keep,
                                    variational_recurrent=c.rnn_recurr_dropout,
                                    input_size=self._rnn_input_size,
                                    dtype=tf.float32)
        return cells


    def _get_rnn_init(self, sent_embeddings, cell):
        """
        Helper to generate initial state of RNN cell.
        """
        c = self._config
        rnn = c.rnn_name
        rnn_init_method = c.rnn_init_method
        if rnn_init_method == 'project_hidden':
            if 'LSTM' in rnn:
                init_state_h = ops.linear(
                                    scope='rnn_initial_state',
                                    inputs=sent_embeddings,
                                    output_dim=cell.state_size[1],
                                    bias_init=None,
                                    activation_fn=None)
                initial_state = tf.contrib.rnn.LSTMStateTuple(
                                    tf.zeros_like(init_state_h), init_state_h)
            elif rnn == 'GRU':
                initial_state = ops.linear(
                                    scope='rnn_initial_state',
                                    inputs=sent_embeddings,
                                    output_dim=cell.state_size,
                                    bias_init=None,
                                    activation_fn=None)
        elif rnn_init_method == 'first_input':
            # Run the RNN cell once to initialise the hidden state
            with tf.variable_scope('rnn_init_input'):
                batch_size = _shape(sent_embeddings)[0]
                sent_embeddings = ops.linear(
                                scope='projection',
                                inputs=sent_embeddings,
                                output_dim=self._rnn_input_size,
                                bias_init=None,
                                activation_fn=None)
                initial_state = cell.zero_state(batch_size, dtype=tf.float32)
                _, initial_state = cell(sent_embeddings, initial_state)
        else:
            raise ValueError('Invalid RNN init method specified.')
        return initial_state
    
    
    def _rnn_dynamic_decoder(self,
                             cell,
                             embedding,
                             output_layer,
                             sample=False):
        
        c = self._config
        is_inference = self.mode == 'infer'
        swap_memory = True
        if c.token_type == 'radix':
            start_id = tf.to_int32(c.radix_base)
            end_id = tf.to_int32(c.radix_base + 1)
        else:
            start_id = tf.to_int32(c.wtoi['<GO>'])
            end_id = tf.to_int32(c.wtoi['<EOS>'])
        
        if is_inference:
            maximum_iterations = c.infer_max_length
            if c.token_type == 'radix':
                max_word_len = len(ops.number_to_base(len(c.wtoi), c.radix_base))
                maximum_iterations *= max_word_len
            elif c.token_type == 'char':
                maximum_iterations *= 5
            beam_search = (is_inference and c.infer_beam_size > 1)
            if sample:
                return rops.rnn_decoder_search(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size_infer,
                                        maximum_iterations,
                                        start_id,
                                        end_id,
                                        swap_memory,
                                        greedy_search=False)
            if beam_search:
                return rops.rnn_decoder_beam_search(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size_infer,
                                        c.infer_beam_size,
                                        c.infer_length_penalty_weight,
                                        maximum_iterations,
                                        start_id,
                                        end_id,
                                        swap_memory)
            else:
                return rops.rnn_decoder_search(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size_infer,
                                        maximum_iterations,
                                        start_id,
                                        end_id,
                                        swap_memory,
                                        greedy_search=True)
        else:
            return rops.rnn_decoder_training(
                                        cell,
                                        embedding,
                                        output_layer,
                                        _shape(embedding)[1],
                                        self._dec_sent_lens,
                                        swap_memory)
    
    ########################
    ### Training helpers ###
    
    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == 'train'
    
    
    def update_lr(self, session, lr_value):
        session.run(self._assign_lr, {self._new_lr: lr_value})
    
    
    def get_global_step(self, session):
        return session.run(self.global_step)
    
    
    def _create_gstep(self):
        """
        Helper to create global step variable.
        """
        #with tf.variable_scope('misc'):
        self.global_step = tf.get_variable(
                                tf.GraphKeys.GLOBAL_STEP,
                                shape=[],
                                dtype=tf.int32,
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                             tf.GraphKeys.GLOBAL_STEP])
        self._new_step = tf.placeholder(tf.int32, None, 'new_global_step')
        self._assign_step = tf.assign(self.global_step, self._new_step)
    
    
    def _create_lr(self):
        """
        Helper to create learning rate variable.
        """
        #with tf.variable_scope('misc'):
        self.lr = tf.get_variable(
                                'learning_rate',
                                shape=[],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        self._new_lr = tf.placeholder(tf.float32, None, 'new_lr')
        self._assign_lr = tf.assign(self.lr, self._new_lr)
        tf.summary.scalar('learning_rate', self.lr)
    
    
    def _create_cosine_lr(self, max_step):
        """
        Helper to anneal learning rate following a cosine curve.
        """
        c = self._config
        self._create_lr()
        with tf.variable_scope('learning_rate'):
            step = tf.to_float(self.global_step / max_step)
            step = 1.0 + tf.cos(tf.minimum(1.0, step) * math.pi)
            lr = (c.lr_start - c.lr_end) * step / 2 + c.lr_end
        self.lr = lr
        tf.summary.scalar('learning_rate', self.lr)
    
    
    def _get_initialiser(self):
        """Helper to select initialiser."""
        if self._config.initialiser == 'xavier':
            print('INFO: Using Xavier initialiser.')
            init = tf.contrib.slim.xavier_initializer()
        else:
            print('INFO: Using TensorFlow default initialiser.')
            init = None
        return init
    
    
    def _get_trainable_vars(self):
        """
        Helper to retrieve list of variables we want to train.
        """
        c = self._config
        tvars = tf.trainable_variables()
        
        if c.freeze_scopes:
            exc_scopes = [sc.strip() for sc in c.freeze_scopes.split(',')]
            tvars = tf.contrib.framework.filter_variables(
                        var_list=tvars,
                        include_patterns='Model',
                        exclude_patterns=exc_scopes,
                        reg_search=True)
            print('INFO: Scopes freezed: {}'.format(exc_scopes))
        return tvars
    
    
    def _get_optimiser(self, lr, momentum=None):
        c = self._config
        opt_type = c.optimiser
        if opt_type == 'adam':
            if momentum is None:
                print('INFO: Using ADAM with default momentum values.')
                opt = tf.train.AdamOptimizer(
                                            learning_rate=lr,
                                            beta1=0.9, beta2=0.999,
                                            epsilon=c.adam_epsilon)
            else:
                print('INFO: Using ADAM with momentum: {}.'.format(momentum))
                opt = tf.train.AdamOptimizer(
                                            learning_rate=lr,
                                            beta1=momentum, beta2=0.999,
                                            epsilon=c.adam_epsilon)
        elif opt_type == 'sgd':
            if momentum is None:
                print('INFO: Using SGD default momentum values.')
                opt = tf.train.MomentumOptimizer(
                                            learning_rate=lr,
                                            momentum=0.9,
                                            use_nesterov=False)
            else:
                print('INFO: Using SGD with momentum: {}.'.format(momentum))
                opt = tf.train.MomentumOptimizer(
                                            learning_rate=lr,
                                            momentum=momentum,
                                            use_nesterov=False)
        else:
            raise ValueError('Unknown optimiser.')
        return opt

