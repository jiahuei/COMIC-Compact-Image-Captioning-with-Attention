# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:12:32 2017

@author: jiahuei

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import base_model_v2 as base
from utility_functions import ops_v3 as my_ops
from utility_functions.captions import image_embedding_v0_4 as image_embed


class Model(base._BaseModel):

    def __init__(self,
                 config,
                 mode, 
                 batch_ops=None,
                 reuse=False,
                 name=None):

        assert mode in ['train', 'eval', 'inference']
        
        super(Model, self).__init__(config)
        self.mode = mode
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._attention = my_ops.SoftAttentionV3
        self._deep_out_wrapper = my_ops.AttentionDeepOutputWrapperV3
        
        if self._config.initialiser == 'xavier':
            init = tf.contrib.slim.xavier_initializer()
        else:
            init = None
        
        # Start to build the model        
        with tf.variable_scope("Model", reuse=reuse, initializer=init):
            self.inference                                                          
            self.loss
            if self.is_training():
                self.optimise
                self._add_vars_summary()
                self.summary_op = tf.summary.merge_all()
        print("INFO: Model initialisation complete.")
    
    
    @my_ops.lazy_property
    def inference(self):
        """
        Builds the core of the model.
        """
        c = self._config
        is_inference = self.mode == 'inference'
        
        ### Setups inputs ###
        
        if is_inference:
            self._enc_inputs = self.batch_ops
            self._seq_lengths = None
        else:
            dec_inputs = self.batch_ops[0]
            self._dec_targets = self.batch_ops[1]
            self._enc_inputs = self.batch_ops[2]
            if c.lang_model == 'word':
                self._dec_inputs = tf.maximum(dec_inputs, 0)
            else:
                self._dec_inputs = tf.one_hot(dec_inputs,
                                              self._softmax_size,
                                              dtype=tf.float32)
            self._dec_targets_masks = tf.sign(
                                        tf.to_float(self._dec_targets + 1))     # Shift by 1, because pad value is -1.
            seq_lengths = tf.reduce_sum(self._dec_targets_masks, axis=1)
            self._seq_lengths = tf.to_int32(seq_lengths)
        
        ### Encoder CNN ###
        
        with tf.variable_scope("encoder"):
            image_features, cnn_fm = self._encoder(self._enc_inputs)
        
        ### Decoder RNN ###
        
        if c.lang_model == 'word':
            decoder = self._decoder_word
        else:
            decoder = self._decoder
        
        with tf.variable_scope("decoder"):
            logits, output_ids, attention_maps = decoder(image_features,
                                                         cnn_fm)
        self._attention_maps = attention_maps
        if is_inference:
            if attention_maps is None:
                self.infer_output = [output_ids, tf.zeros([])]
            else:
                self.infer_output = [output_ids, attention_maps]
            return None
        
        # Log softmax temperature value
        t = tf.get_collection('softmax_temperatures')[0]
        tf.summary.scalar("softmax_temperature", t)
        
        return logits
    
    
    def _encoder(self, enc_inputs):
        """
        Encoder CNN.
        
        Builds image CNN model specified by config.image_model.
        
        Returns:
            image_features: A tensor of shape [batch, image_embed_size].
            cnn_fm: Feature maps specified by config.conv_fm list.
        """
        c = self._config
        image_features, end_points = image_embed.image_model(
                                                config=c,
                                                images=enc_inputs,
                                                spatial_squeeze=True,
                                                is_training=self.is_training())
        if c.conv_fm is not None:
            cnn_fm = end_points[c.conv_fm]
        else:
            cnn_fm = image_features
        
        # Image embedding
        image_features = self._pre_act_linear("image_embedding",
                                              image_features,
                                              c.image_embed_size,
                                              tf.nn.tanh)
        
        # Reshape CNN feature map for RNNs
        cnn_fm_shape = cnn_fm.get_shape().as_list()
        cnn_fm = tf.reshape(cnn_fm, [c.batch_size, -1, cnn_fm_shape[3]])
        
        return image_features, cnn_fm
    
    
    def _decoder(self,
                 image_features,
                 cnn_feature_map):
        """
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        batch_size = c.batch_size
        beam_size = c.beam_size
        rnn_size = c.decoder_rnn_size
        word_size = c.word_size
        softmax_size = self._softmax_size
        is_inference = self.mode == 'inference'
        beam_search = (is_inference and beam_size > 1)
        
        if beam_search:
            # Tile the batch dimension in preparation for Beam Search
            cnn_feature_map = tf.contrib.seq2seq.tile_batch(
                                        cnn_feature_map, beam_size)
            image_features = tf.contrib.seq2seq.tile_batch(
                                        image_features, beam_size)
        
        ### RNN decoder ###
        
        with tf.variable_scope("rnn_decoder"):
            cell = self._get_rnn_cell(rnn_size)
            rnn_init = self._get_rnn_init(image_features, cell)
            cnn_attention = self._attention(rnn_size,
                                            cnn_feature_map,
                                            c.fm_projection,
                                            c.num_heads)
            attention_cell = self._deep_out_wrapper(
                                        deep_output_layer=c.deep_output_layer,
                                        cell=cell,
                                        attention_mechanism=cnn_attention,
                                        attention_layer_size=None,
                                        alignment_history=not(beam_search),     # TArray incompatible with BeamSearchDec
                                        cell_input_fn=None,
                                        output_attention=False,
                                        initial_cell_state=rnn_init)
            if c.multi_softmax:
                raise ValueError
            else:
                output_layer = Dense(softmax_size, name="output_projection")
            
            if c.embedding_weight_tying:
                output_layer.build(word_size)
                embed_map = tf.transpose(output_layer.kernel, [1, 0])
            else:
                embed_map = tf.get_variable(
                                "embedding_map",
                                [softmax_size, word_size],
                                tf.float32,
                                trainable=c.train_lang_model)
            if is_inference:
                # As the softmax size is small, we perform matmul on gpu
                # instead of embedding_lookup on cpu to speed up the operation
                def _embed_fn(ids):
                    ids = tf.one_hot(ids, softmax_size, dtype=tf.float32)
                    ids = tf.reshape(ids, [-1, softmax_size])
                    res = tf.matmul(ids, embed_map)
                    if beam_search:
                        return tf.reshape(res, [batch_size, -1, word_size])
                    else:
                        return tf.reshape(res, [batch_size, word_size])
                embeddings = _embed_fn
            else:
                dec_inputs = tf.reshape(self._dec_inputs, [-1, softmax_size])    # (batch * time, softmax_size)
                embeddings = tf.matmul(dec_inputs, embed_map)
                embeddings = tf.reshape(embeddings, [batch_size, -1, word_size])
                embeddings = tf.transpose(embeddings, [1, 0, 2])                # (max_time, batch_size, word_size)
            
            rnn_raw_outputs = self._rnn_dynamic_decoder(
                                                attention_cell,
                                                embeddings,
                                                output_layer)
            
            # Do some processing on the outputs
            if beam_search:
                top_sequence, top_score, _ = rnn_raw_outputs
                output_ids = tf.transpose(top_sequence, [1, 0])                 # (batch_size, time)
                logits = tf.transpose(top_score, [1, 0])                        # (batch_size, time)
                attn_maps = None
            else:
                output_ids, logits, dec_states = rnn_raw_outputs
                logits = tf.transpose(logits, [1, 0, 2])                        # (batch_size, max_time, softmax_size)
                output_ids = tf.transpose(output_ids, [1, 0])                   # (batch_size, max_time)
                attn_maps = dec_states.alignment_history.stack()
                attn_maps = tf.transpose(attn_maps, [1, 2, 0, 3])           # (batch_size, num_heads, max_time, fm_size)
                map_shape = attn_maps.get_shape().as_list()
                attn_maps = tf.reshape(
                    attn_maps, [map_shape[0] * map_shape[1], -1, map_shape[3]])
            
        return logits, output_ids, attn_maps
    
    
    def _decoder_word(self,
                      image_features,
                      cnn_feature_map):
        """
        Decoder for baseline model, word-based.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        beam_size = self._config.beam_size
        rnn_size = c.decoder_rnn_size
        word_size = c.word_size
        softmax_size = self._softmax_size
        is_inference = self.mode == 'inference'
        beam_search = (is_inference and beam_size > 1)
        
        if beam_search:
            # Tile the batch dimension in preparation for Beam Search
            cnn_feature_map = tf.contrib.seq2seq.tile_batch(
                                        cnn_feature_map, beam_size)
            image_features = tf.contrib.seq2seq.tile_batch(
                                        image_features, beam_size)
        
        
        ### RNN decoder ###
        
        with tf.variable_scope("rnn_decoder"):
            cell = self._get_rnn_cell(rnn_size)
            rnn_init = self._get_rnn_init(image_features, cell)
            cnn_attention = self._attention(rnn_size,
                                            cnn_feature_map,
                                            c.fm_projection,
                                            c.num_heads)
            attention_cell = self._deep_out_wrapper(
                                        deep_output_layer=c.deep_output_layer,
                                        cell=cell,
                                        attention_mechanism=cnn_attention,
                                        attention_layer_size=None,
                                        alignment_history=not(beam_search),     # TArray incompatible with BeamSearchDec
                                        cell_input_fn=None,
                                        output_attention=False,
                                        initial_cell_state=rnn_init)
            if c.multi_softmax:
                raise ValueError
            else:
                output_layer = Dense(softmax_size, name="output_projection")
            
            if c.embedding_weight_tying:
                output_layer.build(word_size)
                word_embed_map = tf.transpose(output_layer.kernel, [1, 0])
                with tf.device("/cpu:0"):
                    word_embed_map = tf.identity(word_embed_map)
            else:
                with tf.device("/cpu:0"):
                    word_embed_map = tf.get_variable(
                                "embedding_map",
                                [softmax_size, word_size],
                                tf.float32,
                                trainable=c.train_lang_model)
            
            if is_inference:
                embeddings = word_embed_map
            else:
                with tf.device("/cpu:0"):
                    embeddings = tf.nn.embedding_lookup(word_embed_map,
                                                        self._dec_inputs)       # (batch_size, max_time, word_size)
                embeddings = tf.transpose(embeddings, [1, 0, 2])                # (max_time, batch_size, word_size)
            
            rnn_raw_outputs = self._rnn_dynamic_decoder(
                                                attention_cell,
                                                embeddings,
                                                output_layer)
            
            # Do some processing on the outputs
            if beam_search:
                top_sequence, top_score, _ = rnn_raw_outputs
                output_ids = tf.transpose(top_sequence, [1, 0])                 # (batch_size, time)
                logits = tf.transpose(top_score, [1, 0])                        # (batch_size, time)
                attn_maps = None
            else:
                output_ids, logits, dec_states = rnn_raw_outputs
                logits = tf.transpose(logits, [1, 0, 2])                        # (batch_size, max_time, softmax_size)
                output_ids = tf.transpose(output_ids, [1, 0])                   # (batch_size, max_time)
                attn_maps = dec_states.alignment_history.stack()
                attn_maps = tf.transpose(attn_maps, [1, 2, 0, 3])           # (batch_size, num_heads, max_time, fm_size)
                map_shape = attn_maps.get_shape().as_list()
                attn_maps = tf.reshape(
                    attn_maps, [map_shape[0] * map_shape[1], -1, map_shape[3]])
            
        return logits, output_ids, attn_maps
    
    
    def _decoder_word_old(self,
                          image_features,
                          cnn_feature_map):
        """
        Decoder for baseline model, word-based.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        beam_size = self._config.beam_size
        rnn_size = c.decoder_rnn_size
        word_size = c.word_size
        softmax_size = self._softmax_size
        is_inference = self.mode == 'inference'
        beam_search = (is_inference and beam_size > 1)
        
        if beam_search:
            # Tile the batch dimension in preparation for Beam Search
            cnn_feature_map = tf.contrib.seq2seq.tile_batch(
                                        cnn_feature_map, beam_size)
            image_features = tf.contrib.seq2seq.tile_batch(
                                        image_features, beam_size)
        
        if not c.embedding_weight_tying:
            with tf.device("/cpu:0"):
                word_embed_map = tf.get_variable(
                                "embedding_map",
                                [softmax_size, word_size],
                                tf.float32,
                                trainable=c.train_lang_model)
        
        ### RNN decoder ###
        
        with tf.variable_scope("rnn_decoder"):
            cell = self._get_rnn_cell(rnn_size)
            rnn_init = self._get_rnn_init(image_features, cell)
            cnn_attention = self._attention(rnn_size,
                                            cnn_feature_map,
                                            c.attention_type,
                                            c.reuse_keys_as_values,
                                            c.fm_projection,
                                            c.num_heads)
            attention_cell = self._deep_out_wrapper(
                                        deep_output_layer=c.deep_output_layer,
                                        cell=cell,
                                        attention_mechanism=cnn_attention,
                                        attention_layer_size=None,
                                        alignment_history=not(beam_search),     # TArray incompatible with BeamSearchDec
                                        cell_input_fn=None,
                                        output_attention=False,
                                        initial_cell_state=rnn_init)
            if c.multi_softmax:
                raise ValueError
            else:
                output_layer = Dense(softmax_size, name="output_projection")
            
            if c.embedding_weight_tying:
                output_layer.build(word_size)
                word_embed_map = tf.transpose(output_layer.kernel, [1, 0])
                with tf.device("/cpu:0"):
                    word_embed_map = tf.identity(word_embed_map)
            
            if is_inference:
                embeddings = word_embed_map
            else:
                with tf.device("/cpu:0"):
                    embeddings = tf.nn.embedding_lookup(word_embed_map,
                                                        self._dec_inputs)       # (batch_size, max_time, word_size)
                embeddings = tf.transpose(embeddings, [1, 0, 2])                # (max_time, batch_size, word_size)
            
            rnn_raw_outputs = self._rnn_dynamic_decoder(
                                                attention_cell,
                                                embeddings,
                                                output_layer)
            
            # Do some processing on the outputs
            if beam_search:
                top_sequence, top_score, _ = rnn_raw_outputs
                output_ids = tf.transpose(top_sequence, [1, 0])                 # (batch_size, time)
                logits = tf.transpose(top_score, [1, 0])                        # (batch_size, time)
                attn_maps = None
            else:
                output_ids, logits, dec_states = rnn_raw_outputs
                logits = tf.transpose(logits, [1, 0, 2])                        # (batch_size, max_time, softmax_size)
                output_ids = tf.transpose(output_ids, [1, 0])                   # (batch_size, max_time)
                attn_maps = dec_states.alignment_history.stack()
                if c.attention_type != 'single':
                    attn_maps = tf.transpose(attn_maps, [1, 2, 0, 3])           # (batch_size, num_heads, max_time, fm_size)
                    map_shape = attn_maps.get_shape().as_list()
                    attn_maps = tf.reshape(
                        attn_maps, [map_shape[0] * map_shape[1], -1, map_shape[3]])
                else:
                    attn_maps = tf.transpose(attn_maps, [1, 0, 2])              # (batch_size, max_time, fm_size)
            
        return logits, output_ids, attn_maps
    
    
    def _rnn_dynamic_decoder(self,
                             cell,
                             embedding,
                             output_layer):
        
        c = self._config
        is_inference = self.mode == 'inference'
        beam_search = (is_inference and c.beam_size > 1)
        swap_memory = True
        
        if c.lang_model == 'baseN' or c.lang_model == 'bpe':
            maximum_iterations = c.max_caption_length * 3
            start_id = tf.to_int32(c.base)
            end_id = tf.to_int32(c.base + 1)
        else:
            if c.lang_model == 'char':
                maximum_iterations = c.max_caption_length * 5
            else:
                maximum_iterations = c.max_caption_length
            start_id = tf.to_int32(c.wtoi['<GO>'])
            end_id = tf.to_int32(c.wtoi['<EOS>'])
        
        if is_inference:
            if beam_search:
                return my_ops.rnn_decoder_beam_search(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size,
                                        c.beam_size,
                                        c.length_penalty_weight,
                                        maximum_iterations,
                                        start_id,
                                        end_id,
                                        swap_memory)
            else:
                return my_ops.rnn_decoder_greedy_search(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size,
                                        maximum_iterations,
                                        start_id,
                                        end_id,
                                        swap_memory)
        else:
            return my_ops.rnn_decoder_training(
                                        cell,
                                        embedding,
                                        output_layer,
                                        c.batch_size,
                                        self._seq_lengths,
                                        swap_memory)
    
    
    @my_ops.def_name_scope
    def loss(self):
        """
        Calculates the average log-perplexity per word, and also the
        doubly stochastic loss of attention map.
        """
        if self.mode == 'inference': return None
        c = self._config
        # Sequence loss
        with tf.name_scope("decoder"):
            targets = tf.maximum(self._dec_targets, 0)
            dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                                                    self.inference,
                                                    targets,
                                                    self._dec_targets_masks)
            tf.summary.scalar("perplexity", tf.exp(dec_log_ppl))
        
        if not self.is_training():
            return dec_log_ppl
        
        # Attention map doubly stochastic loss
        with tf.name_scope("attention_map"):
            # Sum along time dimension
            flat_cnn_maps = tf.reduce_sum(self._attention_maps, axis=1)
            map_loss = tf.squared_difference(1.0, flat_cnn_maps)
            map_loss = tf.reduce_mean(map_loss)
            tf.summary.scalar("map_loss", map_loss)
            map_loss *= c.attention_map_loss_scale
            tf.summary.scalar("map_loss_weighted", map_loss)
        
        # Add losses
        tf.losses.add_loss(dec_log_ppl)
        tf.losses.add_loss(map_loss)
        
        # Add L2 regularisation
        for var in tf.trainable_variables():
            with tf.name_scope("regularisation/%s" % var.op.name):
                tf.losses.add_loss(self._regulariser(var),
                                   tf.GraphKeys.REGULARIZATION_LOSSES)
        return dec_log_ppl
    
    
    @my_ops.def_var_scope
    def optimise(self):
        """
        Builds the graph responsible for optimising the model parameters.
        """
        c = self._config
        self._create_lr_gstep()
        with tf.control_dependencies([self.loss]):
            loss = tf.add_n(tf.losses.get_losses())
            reg_losses = tf.losses.get_regularization_loss()
            total_loss = loss + reg_losses
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularisation_loss", reg_losses)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("learning_rate", self.lr)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           epsilon=1e-6)
        
        if c.train_image_model:
            print("INFO: Training image + language model.")
            var = None
        else:
            print("INFO: Training language model only.")
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='Model/encoder/image_embedding')
            var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='Model/decoder')
        
        train_step = tf.contrib.slim.learning.create_train_op(
                            total_loss,
                            optimizer,
                            global_step=self.global_step,
                            variables_to_train=var,
                            clip_gradient_norm=0,
                            summarize_gradients=c.add_grad_summary)
        return train_step


    def restore_model(self, session, saver, lr):
        """
        Helper function to restore model variables.    
        """
        c = self._config
        def _restore_image_model():
            # Restore image model
            cnn_scope_name = 'Model/encoder/%s' % c.image_model
            cnn_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=cnn_scope_name)
            cnn_vars_name = [v.op.name.replace("Model/encoder/", "") 
                                   for v in cnn_vars_list]
            cnn_variables = dict(zip(cnn_vars_name, cnn_vars_list))
            cnn_saver = tf.train.Saver(cnn_variables)
            cnn_saver.restore(session, c.cnn_checkpoint_path)
            print("INFO: Restored image model from checkpoint.")
        
        def _restore_lang_model():
            # Restore language model
            lang_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                               scope="Model/decoder")
            lang_saver = tf.train.Saver(lang_vars_list)
            checkpoint_path = tf.train.latest_checkpoint(c.checkpoint_path)
            lang_saver.restore(session, checkpoint_path)
            print("INFO: Restored language model from checkpoint.")
        
        if c.resume_training:
            checkpoint_path = tf.train.latest_checkpoint(c.checkpoint_path)
            saver.restore(session, checkpoint_path)
            print("INFO: Restored entire model from checkpoint.")
        else:
            if c.cnn_checkpoint_path is not None:
                _restore_image_model()
            if c.checkpoint_path is not None:
                _restore_lang_model()
            
        if c.lr_start is None:
            lr = session.run(self.lr)
        else:
            self.update_lr(session, lr)
            session.run(self.lr)
        return lr
    
    
    
    
    
    
    