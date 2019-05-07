#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:12:25 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptb_tokenizer import PTBTokenizer
from subprocess import call
from collections import deque
from tqdm import tqdm
import numpy as np
import h5py, json
import os, copy, string, re, tempfile, time


def _reverse_sequence(captions, wtoi):
    """Helper to reverse word sequence of captions."""
    if wtoi['<PAD>'] == 0:
        seq_len = np.sum(np.sign(captions), axis=1)
    elif wtoi['<PAD>'] == -1:
        seq_len = np.sum(np.sign(captions + 1), axis=1)
    reverse_captions = copy.deepcopy(captions)
    for row in range(captions.shape[0]):
        r_idx = np.arange(seq_len[row]-1, -1, -1)
        idx = np.arange(seq_len[row])
        reverse_captions[row, idx] = captions[row, r_idx]
    return reverse_captions


def _process_captions(captions, wtoi):
    """Helper to add <GO> and <EOS> symbols.
    
    Args:
        captions: batched captions / label Tensor.
        wtoi: Python dictionary mapping word to id.
    
    Returns:
        inputs: captions with <GO> appended.
        targets: captions with <EOS> at end of sequence.
        seq_len: sequence lengths of inputs and targets.
        reverse_targets: reversed captions with <EOS> at end of sequence.
    """
    
    reverse_captions = _reverse_sequence(captions, wtoi)
    go_sym = np.full([captions.shape[0], 1], wtoi['<GO>'], np.int32)
    inputs = np.concatenate((go_sym, captions), axis=1)
    
    pad_sym = np.full([captions.shape[0], 1], wtoi['<PAD>'], np.int32)
    targets = np.concatenate((captions, pad_sym), axis=1)
    reverse_targets = np.concatenate((reverse_captions, pad_sym), axis=1)
    idx = np.arange(0, captions.shape[0], 1)
    if wtoi['<PAD>'] == 0:
        seq_len = np.sum(np.sign(targets), axis=1)
    elif wtoi['<PAD>'] == -1:
        seq_len = np.sum(np.sign(targets + 1), axis=1)
    targets[idx, seq_len] = wtoi['<EOS>']
    reverse_targets[idx, seq_len] = wtoi['<EOS>']
    seq_len += 1
    return inputs.astype(np.int32), targets.astype(np.int32), \
            seq_len.astype(np.int32), reverse_targets.astype(np.int32)


def _process_captions_char(captions,
                           ctoi,
                           word_len_thres):
    """Helper to add <GO> and <EOS> symbols.
    
    NOTE: This is the character-based version of `_process_captions()`.
    
    Args:
        captions: batched captions / label Tensor.
        ctoi: Python dictionary mapping character to id.
        word_len_thres: Length of a word (number of characters). 
    
    Returns:
        inputs: captions with <GO> appended.
        targets: captions with <EOS> at end of sequence.
        seq_len: sequence lengths of inputs and targets.
        reverse_targets: reversed captions with <EOS> at end of sequence.
    """
    
    reverse_captions = _reverse_sequence(captions)
    pad_sym = np.zeros([captions.shape[0], word_len_thres], np.int32)
    go_sym = np.full([captions.shape[0], word_len_thres],
                     ctoi[unicode('<GO>')],
                     np.int32)
    inputs = np.concatenate((go_sym, captions), axis=1)
    
    targets = np.concatenate((captions, pad_sym), axis=1)
    reverse_targets = np.concatenate((reverse_captions, pad_sym), axis=1)
    seq_len = np.sum(np.sign(targets), axis=1)
    row_idx = np.arange(0, captions.shape[0], 1)
    for row in row_idx:
        col_idx = np.arange(seq_len[row], seq_len[row] + word_len_thres, 1)
        targets[row, col_idx] = ctoi[unicode('<EOS>')]
        reverse_targets[row, col_idx] = ctoi[unicode('<EOS>')]
    seq_len += word_len_thres
    return inputs.astype(np.int32), targets.astype(np.int32), \
            seq_len.astype(np.int32), reverse_targets.astype(np.int32)


def _split_check(split, include_restval=True):
    """Helper to check the splits. Returns Boolean."""
    if include_restval:
        return split == 'train' or split == 'restval'
    else:
        return split == 'train'


def number_to_base(n, base):
    """Function to convert any base-10 integer to base-N."""
    if base < 2:
        raise ValueError("Base cannot be less than 2.")
    if n < 0:
        sign = -1
        n *= sign
    elif n == 0:
        return [0]
    else:
        sign = 1
    digits = []
    while n:
        digits.append(sign * int(n % base))
        n //= base
    return digits[::-1]


def tokenise(dataset,
             image_id_key='cocoid',
             retokenise=False):
    """
    Tokenise captions (optional), remove non-alphanumerics.
    
    Args:
        dataset: Dictionary object loaded from Karpathy's dataset JSON file.
        image_id_key: String. Used to access `image_id` field.
        retokenise: Boolean. Whether to retokenise the raw captions using 
            Stanford-CoreNLP-3.4.1.
    
    Returns:
        A dictionary with tokenised captions.
    """
    if retokenise:
        print("\nINFO: Tokenising captions using PTB Tokenizer.\n")
        tokenizer = PTBTokenizer()
        
        raw_list = []
        for d in dataset['images']:
            for s in d['sentences']:
                raw_list.append(s['raw'])
        tokenised_cap, tokenised_cap_w_punc = tokenizer.tokenize(raw_list)
        
        tokenised_data = []
        cap_id = 0
        for d in dataset['images']:
            if 'filepath' in d.keys():
                filepath = os.path.join(d['filepath'], d['filename'])
            else:
                filepath = d['filename']
            temp_dict = dict(split=d['split'],
                             filepath=filepath,
                             image_id=d[image_id_key],
                             raw=[],
                             tokens=[])
            for s in d['sentences']:
                temp_dict['raw'].append(s['raw'])
                temp_dict['tokens'].append(
                        [unicode(w) for w in tokenised_cap[cap_id].split(' ')
                            if w != ''])
                cap_id += 1
            tokenised_data.append(temp_dict)
    else:
        print("\nINFO: Using tokenised captions.\n")
        #pattern = re.compile(r'([^\s\w]|_)+', re.UNICODE)      # matches non-alphanumerics
        pattern = re.compile(r'([^\w]|_)+', re.UNICODE)         # matches non-alphanumerics and whitespaces
        tokenised_data = []
        for d in dataset['images']:
            if 'filepath' in d.keys():
                filepath = os.path.join(d['filepath'], d['filename'])
            else:
                filepath = d['filename']
            temp_dict = dict(split=d['split'],
                             filepath=filepath,
                             image_id=d[image_id_key],
                             raw=[],
                             tokens=[])
            for s in d['sentences']:
                temp_dict['raw'].append(s['raw'])
                temp_list = []
                for w in s['tokens']:
                    w = re.sub(pattern, '', w.lower())
                    if w != '': temp_list.append(w)
                temp_dict['tokens'].append(temp_list)
            tokenised_data.append(temp_dict)
    return tokenised_data


def get_truncate_length(tokenised_dataset,
                        truncate_percentage,
                        include_restval=True):
    """
    Calculates the maximum caption length such that truncated captions makes
    up `truncate_precentage` of the training corpus.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        truncate_percentage: The percentage of truncated captions.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        The maximum caption length.
    """
    lengths = {}
    num_captions = 0
    for d in tokenised_dataset:
        split = d['split']
        if _split_check(split, include_restval):
            for s in d['tokens']:
                lengths[len(s)] = lengths.get(len(s), 0) + 1
                num_captions += 1
    truncate_length = 0
    percentage = .0
    for key, value in sorted(lengths.iteritems()):
        if percentage > (100.0 - truncate_percentage):
            truncate_length = key
            break
        percentage += lengths[key] / num_captions * 100
    print("INFO: Captions longer than {} words will be truncated.\n".format(truncate_length))
    return truncate_length


def build_vocab(tokenised_dataset,
                word_count_thres,
                caption_len_thres,
                vocab_size=None,
                include_restval=True,
                pad_value=0,
                include_GO_EOS_tokens=True):
    """
    Builds the word-to-id and id-to-word dictionaries.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        word_count_thres: Threshold for word occurrence. Words that appear
            less than this threshold will be converted to <UNK> token.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
        pad_value: Value assigned to <PAD> token.
    
    Returns:
        Word-to-id and id-to-word dictionaries.
    """
    print("INFO: Building vocabulary.\n")
    
    counts = {}
    for d in tokenised_dataset:
        split = d['split']
        if _split_check(split, include_restval):
            for s in d['tokens']:
                for w_count, w in enumerate(s):
                    if w_count < caption_len_thres:
                        counts[w] = counts.get(w, 0) + 1
    
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    
    if vocab_size is None:
        print("INFO: Vocab: Filtering out words with count less than {}.\n".format(
                word_count_thres))
        vocab = [w[1] for w in cw if counts[w[1]] >= word_count_thres]
    else:
        print("INFO: Vocab: Generating vocab with fixed size {}.\n".format(
                vocab_size))
        vocab = [w[1] for w in cw[:vocab_size]]
    #vocab_count = [w for w in cw if counts[w[1]] >= WORD_COUNT_THRES]
    #vocab_inv_freq = [1.0 - (w[0] / float(vocab_count[0][0])) for w in vocab_count]
    #vocab_weights = [0.5 + (f * 1.5) for f in vocab_inv_freq]
    
    wtoi = {}
    itow = {}
    idx = pad_value
    wtoi['<PAD>'] = idx
    itow[idx] = '<PAD>'
    idx += 1
    for w in vocab:
        wtoi[w] = idx
        itow[idx] = w
        idx += 1
    wtoi['<UNK>'] = idx
    itow[idx] = '<UNK>'
    idx += 1
    
    if include_GO_EOS_tokens:
        wtoi['<GO>'] = idx
        itow[idx] = '<GO>'
        idx += 1
        wtoi['<EOS>'] = idx
        itow[idx] = '<EOS>'
    
    time.sleep(0.5)
    
    return wtoi, itow


def build_vocab_char(include_space=False,
                     pad_value=0):
    """
    Builds the character-to-id and id-to-character dictionaries.
    
    The alphanumeric characters in the vocab are: a-z (lowercase), 0-9.
    A character pad value of 1 is used for easier sequence length calculation.
    
    Returns:
        Character-to-id and id-to-character dictionaries.
    """
    print("INFO: Building vocabulary.\n")
    
    char_list = list(string.digits + string.ascii_lowercase)
    
    ctoi = {}
    itoc = {}
    idx = pad_value
    ctoi['<PAD>'] = idx
    itoc[idx] = '<PAD>'
    idx += 1
    
    if include_space:
        ctoi[' '] = idx
        itoc[idx] = ' '
        idx += 1
    
    for c in char_list:
        ctoi[c] = idx
        itoc[idx] = c
        idx += 1
    ctoi['<GO>'] = len(ctoi)
    ctoi['<EOS>'] = len(ctoi)
    itoc[len(itoc)] = '<GO>'
    itoc[len(itoc)] = '<EOS>'
    
    return ctoi, itoc


def build_vocab_char_word():
    """
    Builds the character-to-id and id-to-character dictionaries.
    
    The alphanumeric characters in the vocab are: a-z (lowercase), 0-9.
    A character pad value of 1 is used for easier sequence length calculation.
    
    Returns:
        Character-to-id and id-to-character dictionaries.
    """
    print("INFO: Building vocabulary.\n")
    
    char_list = list(string.digits + string.ascii_lowercase)
    
    ctoi = {}
    itoc = {}
    ctoi[unicode('<PAD>')] = 0
    ctoi[unicode('<char_PAD>')] = 1
    itoc[0] = unicode('<PAD>')
    itoc[1] = unicode('<char_PAD>')
    idx = 2
    for c in char_list:
        ctoi[unicode(c)] = idx
        itoc[idx] = unicode(c)
        idx += 1
    ctoi[unicode('<GO>')] = len(ctoi)
    ctoi[unicode('<EOS>')] = len(ctoi)
    itoc[len(itoc)] = unicode('<GO>')
    itoc[len(itoc)] = unicode('<EOS>')
    
    return ctoi, itoc


def clean_raw_captions(tokenised_dataset):
    """
    Removes all non-alphanumeric characters with a single space.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
    
    Returns:
        The same dictionary as input, with the `raw` captions cleaned.
    """
    print("INFO: Cleaning raw captions.\n")
    
    pattern = re.compile(r'[^a-zA-Z0-9]+', re.UNICODE)
    eol_pattern = re.compile(r'[ ]+$', re.UNICODE)
    
    for i, d in enumerate(tokenised_dataset):
        for j, s in enumerate(d['raw']):
            s = re.sub(pattern, ' ', s.lower())
            s = re.sub(eol_pattern, '', s)
            d['raw'][j] = s
        tokenised_dataset[i]['raw'] = d['raw']
    return tokenised_dataset


def learn_and_apply_bpe(tokenised_dataset,
                        num_symbols,
                        output_dir,
                        vocabulary_threshold=2,
                        include_restval=True,
                        delete_files=True):
    """
    Use byte pair encoding (BPE) to learn a variable-length encoding of the
    vocabulary in a text. Then we use operations learned with learn_bpe.py to
    encode a new text. The text will not be smaller, but use only a fixed
    vocabulary, with rare words encoded as variable-length sequences of
    subword units.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        num_symbols: Create this many new symbols, each representing a
            character n-gram.
        output_dir: The directory for the output files.
        vocabulary_threshold: Any symbol with frequency < threshold will be
            treated as OOV. Defaults to 2.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
        delete_files: Boolean. Whether to delete the BPE output files.
    
    Returns:
        Updated `tokenised_dataset`.
    """
    subword_dir_name = 'subword-nmt-master'
    bpe_code_file = 'bpe_s{:d}_code.txt'.format(num_symbols)
    bpe_vocab_file = 'bpe_s{:d}_vocab.txt'.format(num_symbols)
    bpe_code_file = os.path.join(output_dir, bpe_code_file)
    bpe_vocab_file = os.path.join(output_dir, bpe_vocab_file)
    splits = ['train', 'val', 'test']
    
    print("INFO: Starting BPE, number of symbols: {:d}.".format(num_symbols))
    sentences = dict(train=[], val=[], test=[])
    for d in tokenised_dataset:
        if d['split'] == 'restval':
            split = 'train'
        else:
            split = d['split']
        for s in d['raw']:
            sentences[split].append(s)
    
    tmp_file = {}
    for split in splits:
        tmp_file[split] = tempfile.NamedTemporaryFile(delete=False, dir=output_dir)
        tmp_file[split].write('\r\n'.join(sentences[split]))
        tmp_file[split].close()
    
    print("INFO: Learning BPE.")
    cwd = os.getcwd()
    cmd = ['python', os.path.join(cwd, subword_dir_name, 'learn_bpe.py'),
           '-s', str(num_symbols),
           '-i', tmp_file['train'].name,
           '-o', bpe_code_file]
    call(cmd)
    
    print("INFO: Applying BPE, first pass.")
    # Apply BPE to train corpus
    cmd = ['python', os.path.join(cwd, subword_dir_name, 'apply_bpe.py'),
           '-c', bpe_code_file,
           '-i', tmp_file['train'].name,
           '-o', os.path.join(output_dir, 'bpe_s{:d}_train.txt'.format(num_symbols))]
    call(cmd)
    # Obtain vocabulary
    cmd = ['python', os.path.join(cwd, subword_dir_name, 'get_vocab.py')]
    fi = open(os.path.join(output_dir, 'bpe_s{:d}_train.txt'.format(num_symbols)), 'r')
    fo = open(bpe_vocab_file, 'w')
    call(cmd, stdin=fi, stdout=fo)
    fi.close()
    fo.close()
    
    print("INFO: Applying BPE, with vocabulary filter.")
    # Reapply BPE to train, val, test corpora
    for split in splits:
        cmd = ['python', os.path.join(cwd, subword_dir_name, 'apply_bpe.py'),
               '-c', bpe_code_file,
               '--vocabulary', bpe_vocab_file,
               '--vocabulary-threshold', str(vocabulary_threshold),
               '-i', tmp_file[split].name,
               '-o', os.path.join(output_dir, 'bpe_s{:d}_{}.txt'.format(num_symbols, split))]
        call(cmd)
        os.remove(tmp_file[split].name)
    # Obtain vocabulary
    cmd = ['python', os.path.join(cwd, subword_dir_name, 'get_vocab.py')]
    fi = open(os.path.join(output_dir, 'bpe_s{:d}_train.txt'.format(num_symbols)), 'r')
    fo = open(bpe_vocab_file, 'w')
    call(cmd, stdin=fi, stdout=fo)
    fi.close()
    fo.close()
    #with open(os.path.join(output_dir, bpe_vocab_file), 'r') as f:
    #    vocab = [l.split(' ')[0] for l in f.readlines()]
    
    print("INFO: Reading processed corpora.")
    corpora = dict(train=[], val=[], test=[])
    for split in splits:
        name = 'bpe_s{:d}_{}.txt'.format(num_symbols, split)
        with open(os.path.join(output_dir, name), 'r') as f:
            corpora[split] = deque([l.strip() for l in f.readlines()])
    
    for i, d in enumerate(tokenised_dataset):
        if d['split'] == 'restval':
            split = 'train'
        else:
            split = d['split']
        for j, s in enumerate(d['tokens']):
            d['tokens'][j] = corpora[split].popleft().split(' ')
        tokenised_dataset[i] = d
    
    if delete_files:
        for split in splits:
            os.remove(os.path.join(output_dir, 'bpe_s{:d}_{}.txt'.format(num_symbols, split)))
        os.remove(bpe_code_file)
        os.remove(bpe_vocab_file)
    
    print("INFO: BPE processing completed.\n")
    return tokenised_dataset


def tokenised_word_to_ids(tokenised_dataset,
                          wtoi,
                          caption_len_thres,
                          include_restval=True):
    """
    Builds the train, validation and test dictionaries.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        wtoi: Word-to-id dictionary.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    #print("INFO: Converting tokenised words to ids.\n")
    
    train = dict(inputs=[],
                 targets=[],
                 reverse_targets=[],
                 caption_id=[],
                 filepaths=[],
                 image_id=[],
                 seq_len=[])
    valid = copy.deepcopy(train)
    test = copy.deepcopy(train)
    num_UNK_words = 0
    num_total_words = 0
    cap_id = 0
    
    for i, d in enumerate(tqdm(tokenised_dataset, ncols=70, desc="Word-to-id")):
        split = d['split']
        if include_restval is False:
            if split == 'restval': continue
        sent_list = d['tokens']
        sent_list_new = []
        for s in sent_list:
            l = []
            for w_count, w in enumerate(s):
                if w_count == caption_len_thres: break
                w_id = wtoi.get(w, wtoi[unicode('<UNK>')])
                if w_id == wtoi[unicode('<UNK>')]: num_UNK_words += 1
                num_total_words += 1
                l.append(w_id)
            # Pad captions to predefined length with <PAD>.
            cap_np = np.pad(np.array(l),
                            (0, caption_len_thres - len(l)),
                            'constant',
                            constant_values=(0, wtoi['<PAD>']))
            sent_list_new.append(cap_np)
        # Add <GO> and <EOS> to captions.
        _ = _process_captions(np.array(sent_list_new), wtoi)
        inputs, targets, seq_len, reverse_targets = _
        
        if split == 'val':
            valid['inputs'].append(inputs)
            valid['targets'].append(targets)
            valid['reverse_targets'].append(reverse_targets)
            valid['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                valid['filepaths'].append(d['filepath'])
                valid['image_id'].append(d['image_id'])
                valid['caption_id'].append(cap_id)
                cap_id += 1
        elif split == 'test':
            test['inputs'].append(inputs)
            test['targets'].append(targets)
            test['reverse_targets'].append(reverse_targets)
            test['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                test['filepaths'].append(d['filepath'])
                test['image_id'].append(d['image_id'])
                test['caption_id'].append(cap_id)
                cap_id += 1
        else:
            train['inputs'].append(inputs)
            train['targets'].append(targets)
            train['reverse_targets'].append(reverse_targets)
            train['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                train['filepaths'].append(d['filepath'])
                train['image_id'].append(d['image_id'])
                train['caption_id'].append(cap_id)
                cap_id += 1

    # Sort training data according to caption length
    
    train['inputs'] = np.concatenate(train['inputs'], axis=0)
    train['targets'] = np.concatenate(train['targets'], axis=0)
    train['reverse_targets'] = np.concatenate(train['reverse_targets'], axis=0)
    train['caption_id'] = np.array(train['caption_id']).astype(np.int32)
    train['seq_len'] = np.concatenate(train['seq_len'], axis=0)
    sorted_seq_len_idx_list = sorted(range(len(train['seq_len'])),
                                     key=lambda x:train['seq_len'][x])
    train['inputs'] = train['inputs'][sorted_seq_len_idx_list, :]
    train['targets'] = train['targets'][sorted_seq_len_idx_list, :]
    train['reverse_targets'] = train['reverse_targets'][sorted_seq_len_idx_list, :]
    train['caption_id'] = train['caption_id'][sorted_seq_len_idx_list]
    train['seq_len'] = train['seq_len'][sorted_seq_len_idx_list]
    train['filepaths'] = [train['filepaths'][i] for i in sorted_seq_len_idx_list]
    
    valid['inputs'] = np.concatenate(valid['inputs'], axis=0)
    valid['targets'] = np.concatenate(valid['targets'], axis=0)
    valid['reverse_targets'] = np.concatenate(valid['reverse_targets'], axis=0)
    valid['caption_id'] = np.array(valid['caption_id']).astype(np.int32)
    valid['seq_len'] = np.concatenate(valid['seq_len'], axis=0)
    
    test['inputs'] = np.concatenate(test['inputs'], axis=0)
    test['targets'] = np.concatenate(test['targets'], axis=0)
    test['reverse_targets'] = np.concatenate(test['reverse_targets'], axis=0)
    test['caption_id'] = np.array(test['caption_id']).astype(np.int32)
    test['seq_len'] = np.concatenate(test['seq_len'], axis=0)
    
    # Ratio of UNK tokens and Distribution of caption lengths
    len_dist = {}
    for l in train['seq_len']:
        len_dist[l] = len_dist.get(l, 0) + 1
    UNK_ratio = num_UNK_words / num_total_words * 100
    
    print("\nINFO: Caption length distribution:\n%r" % len_dist)
    print("\nINFO: Ratio of <UNK> to total tokens: %1.3f %%" % UNK_ratio)
    
    return train, valid, test


def tokenised_word_to_baseN_ids(tokenised_dataset,
                                wtoi,
                                caption_len_thres,
                                base,
                                include_restval=True):
    """
    Builds the train, validation and test dictionaries.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        wtoi: Word-to-id dictionary.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        base: The number base to convert to.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    
    def _pad_left(values, max_length):
        # Function to apply padding on left
        if values == [-1]:
            pad_value = -1
        else:
            pad_value = 0
        return np.pad(values,
                      (max_length - len(values), 0),
                      'constant',
                      constant_values=(pad_value, 0))
    
    def _ndarray_to_base(array, base, max_length):
        # Function to convert nd-array to base-N, with fixed number of digits
        new_arr = []
        for row in array:
            new_row = []
            for element in row:
                result = _pad_left(number_to_base(element, base), max_length)
                new_row.append(result)
            new_arr.append(np.concatenate(new_row))
        return np.array(new_arr)
    
    def _process_cap(captions, wtoi, base, max_length):
        # Adds <GO> and <EOS> tokens
        reverse_captions = _reverse_sequence(captions, wtoi)
        go_sym = np.full([captions.shape[0], 1], base, np.int32)
        inputs = np.concatenate((go_sym, captions), axis=1)
        
        pad_sym = np.full([captions.shape[0], 1], wtoi['<PAD>'], np.int32)
        targets = np.concatenate((captions, pad_sym), axis=1)
        reverse_targets = np.concatenate((reverse_captions, pad_sym), axis=1)
        
        if wtoi['<PAD>'] == 0:
            seq_len = np.sum(np.sign(targets), axis=1)
        elif wtoi['<PAD>'] == -1:
            seq_len = np.sum(np.sign(targets + 1), axis=1)
        idx = np.arange(0, captions.shape[0], 1)
        targets[idx, seq_len] = base + 1
        reverse_targets[idx, seq_len] = base + 1
        seq_len += 1
        return inputs.astype(np.int32), targets.astype(np.int32), \
                seq_len.astype(np.int32), reverse_targets.astype(np.int32)
    
    #print("INFO: Converting tokenised words to base-N ids.\n")
    
    train = dict(inputs=[],
                 targets=[],
                 reverse_targets=[],
                 caption_id=[],
                 filepaths=[],
                 image_id=[],
                 seq_len=[])
    valid = copy.deepcopy(train)
    test = copy.deepcopy(train)
    num_UNK_words = 0
    num_total_words = 0
    cap_id = 0
    max_word_len = len(number_to_base(len(wtoi), base))
    
    for i, d in enumerate(tqdm(tokenised_dataset, ncols=70, desc="BaseN-to-id")):
        split = d['split']
        if include_restval is False:
            if split == 'restval': continue
        sent_list = d['tokens']
        sent_list_new = []
        for s in sent_list:
            line = []
            for w_count, w in enumerate(s):
                if w_count == caption_len_thres: break
                w_id = wtoi.get(w, wtoi[unicode('<UNK>')])
                if w_id == wtoi[unicode('<UNK>')]: num_UNK_words += 1
                num_total_words += 1
                w_id = _pad_left(number_to_base(w_id, base), max_word_len)
                line.append(w_id)
            # Pad captions to predefined length with <PAD>.
            line = np.concatenate(line, axis=0)
            cap_np = np.pad(line,
                            (0, caption_len_thres * max_word_len - len(line)),
                            'constant',
                            constant_values=(0, wtoi['<PAD>']))
            sent_list_new.append(cap_np)
        # Add <GO> and <EOS> to captions.
        _ = _process_cap(np.array(sent_list_new), wtoi, base, max_word_len)
        inputs, targets, seq_len, reverse_targets = _
        
        if split == 'val':
            valid['inputs'].append(inputs)
            valid['targets'].append(targets)
            valid['reverse_targets'].append(reverse_targets)
            valid['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                valid['filepaths'].append(d['filepath'])
                valid['image_id'].append(d['image_id'])
                valid['caption_id'].append(cap_id)
                cap_id += 1
        elif split == 'test':
            test['inputs'].append(inputs)
            test['targets'].append(targets)
            test['reverse_targets'].append(reverse_targets)
            test['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                test['filepaths'].append(d['filepath'])
                test['image_id'].append(d['image_id'])
                test['caption_id'].append(cap_id)
                cap_id += 1
        else:
            train['inputs'].append(inputs)
            train['targets'].append(targets)
            train['reverse_targets'].append(reverse_targets)
            train['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                train['filepaths'].append(d['filepath'])
                train['image_id'].append(d['image_id'])
                train['caption_id'].append(cap_id)
                cap_id += 1

    # Sort training data according to caption length
    
    train['inputs'] = np.concatenate(train['inputs'], axis=0)
    train['targets'] = np.concatenate(train['targets'], axis=0)
    train['reverse_targets'] = np.concatenate(train['reverse_targets'], axis=0)
    train['caption_id'] = np.array(train['caption_id']).astype(np.int32)
    train['seq_len'] = np.concatenate(train['seq_len'], axis=0)
    sorted_seq_len_idx_list = sorted(range(len(train['seq_len'])),
                                     key=lambda x:train['seq_len'][x])
    train['inputs'] = train['inputs'][sorted_seq_len_idx_list, :]
    train['targets'] = train['targets'][sorted_seq_len_idx_list, :]
    train['reverse_targets'] = train['reverse_targets'][sorted_seq_len_idx_list, :]
    train['caption_id'] = train['caption_id'][sorted_seq_len_idx_list]
    train['seq_len'] = train['seq_len'][sorted_seq_len_idx_list]
    train['filepaths'] = [train['filepaths'][i] for i in sorted_seq_len_idx_list]
    
    valid['inputs'] = np.concatenate(valid['inputs'], axis=0)
    valid['targets'] = np.concatenate(valid['targets'], axis=0)
    valid['reverse_targets'] = np.concatenate(valid['reverse_targets'], axis=0)
    valid['caption_id'] = np.array(valid['caption_id']).astype(np.int32)
    valid['seq_len'] = np.concatenate(valid['seq_len'], axis=0)
    
    test['inputs'] = np.concatenate(test['inputs'], axis=0)
    test['targets'] = np.concatenate(test['targets'], axis=0)
    test['reverse_targets'] = np.concatenate(test['reverse_targets'], axis=0)
    test['caption_id'] = np.array(test['caption_id']).astype(np.int32)
    test['seq_len'] = np.concatenate(test['seq_len'], axis=0)
    
    # Ratio of UNK tokens and Distribution of caption lengths
    len_dist = {}
    for l in train['seq_len']:
        len_dist[l] = len_dist.get(l, 0) + 1
    UNK_ratio = num_UNK_words / num_total_words * 100
    
    print("\nINFO: Caption length distribution:\n%r" % len_dist)
    print("\nINFO: Ratio of <UNK> to total tokens: %1.3f %%" % UNK_ratio)
    
    return train, valid, test


def tokenised_word_to_char_ids(tokenised_dataset,
                               ctoi,
                               word_len_thres,
                               caption_len_thres,
                               include_restval=True):
    """
    Builds the train, validation and test dictionaries.
    
    NOTE: This is the character-based version, which will break up words into
        characters. Each word will be padded to a length of `word_len_thres`
        with value of `1`.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        ctoi: Character-to-id dictionary.
        word_len_thres: Length of a word (number of characters). Words
            with longer lengths are truncated.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    #print("INFO: Converting tokenised words to ids.\n")
    
    train = dict(inputs=[],
                 targets=[],
                 reverse_targets=[],
                 caption_id=[],
                 filepaths=[],
                 image_id=[],
                 seq_len=[])
    valid = copy.deepcopy(train)
    test = copy.deepcopy(train)
    cap_id = 0
    
    # Get word length distribution statistics
    words = []
    word_len = []
    for d in tokenised_dataset:
        for s in d['tokens']:
            for w in s:
                words.append(w)
                word_len.append(len(w))
    word_len_dist = np.histogram(word_len, bins=range(1, word_len_thres + 3))
    word_len_dist = np.stack([word_len_dist[1][:-1], word_len_dist[0]], axis=1)
    word_len_dist = dict(word_len_dist)
    
    for i, d in enumerate(tqdm(tokenised_dataset, ncols=70, desc="CharWord-to-id")):
        split = d['split']
        if include_restval is False and split == 'restval': continue
        sent_list = d['tokens']
        sent_list_new = []
        for s in sent_list:
            l = []
            for w_count, w in enumerate(s):
                if w_count == caption_len_thres: break
                chars = list(w)
                chars = [ctoi[c] for c in chars[: word_len_thres]]
                assert len(chars) <= word_len_thres
                # Pad words to predefined length with <char_PAD>.
                l.append(np.pad(np.array(chars),
                                (0, word_len_thres - len(chars)),
                                'constant',
                                constant_values=(0, ctoi['<char_PAD>'])))
                assert len(l[-1]) == word_len_thres
            # Pad captions to predefined length with <PAD>.
            cap_np = np.concatenate(l, axis=0)
            cap_np = np.pad(cap_np,
                            (0, caption_len_thres * word_len_thres - len(cap_np)),
                            'constant',
                            constant_values=(0, ctoi['<PAD>']))
            sent_list_new.append(cap_np.astype(np.int32))
        
        # Add <GO> and <EOS> to captions.
        _ = _process_captions_char(np.array(sent_list_new), ctoi, word_len_thres)
        inputs, targets, seq_len, reverse_targets = _
        
        if split == 'val':
            valid['inputs'].append(inputs)
            valid['targets'].append(targets)
            valid['reverse_targets'].append(reverse_targets)
            valid['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                valid['filepaths'].append(d['filepath'])
                valid['image_id'].append(d['image_id'])
                valid['caption_id'].append(cap_id)
                cap_id += 1
        elif split == 'test':
            test['inputs'].append(inputs)
            test['targets'].append(targets)
            test['reverse_targets'].append(reverse_targets)
            test['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                test['filepaths'].append(d['filepath'])
                test['image_id'].append(d['image_id'])
                test['caption_id'].append(cap_id)
                cap_id += 1
        else:
            train['inputs'].append(inputs)
            train['targets'].append(targets)
            train['reverse_targets'].append(reverse_targets)
            train['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                train['filepaths'].append(d['filepath'])
                train['image_id'].append(d['image_id'])
                train['caption_id'].append(cap_id)
                cap_id += 1
    
    # Sort training data according to caption length
    train['inputs'] = np.concatenate(train['inputs'], axis=0)
    train['targets'] = np.concatenate(train['targets'], axis=0)
    train['reverse_targets'] = np.concatenate(train['reverse_targets'], axis=0)
    train['caption_id'] = np.array(train['caption_id']).astype(np.int32)
    train['seq_len'] = np.concatenate(train['seq_len'], axis=0)
    sorted_seq_len_idx_list = sorted(range(len(train['seq_len'])),
                                     key=lambda x:train['seq_len'][x])
    train['inputs'] = train['inputs'][sorted_seq_len_idx_list, :]
    train['targets'] = train['targets'][sorted_seq_len_idx_list, :]
    train['reverse_targets'] = train['reverse_targets'][sorted_seq_len_idx_list, :]
    train['caption_id'] = train['caption_id'][sorted_seq_len_idx_list]
    train['seq_len'] = train['seq_len'][sorted_seq_len_idx_list]
    train['filepaths'] = [train['filepaths'][i] for i in sorted_seq_len_idx_list]
    
    valid['inputs'] = np.concatenate(valid['inputs'], axis=0)
    valid['targets'] = np.concatenate(valid['targets'], axis=0)
    valid['reverse_targets'] = np.concatenate(valid['reverse_targets'], axis=0)
    valid['caption_id'] = np.array(valid['caption_id']).astype(np.int32)
    valid['seq_len'] = np.concatenate(valid['seq_len'], axis=0)
    
    test['inputs'] = np.concatenate(test['inputs'], axis=0)
    test['targets'] = np.concatenate(test['targets'], axis=0)
    test['reverse_targets'] = np.concatenate(test['reverse_targets'], axis=0)
    test['caption_id'] = np.array(test['caption_id']).astype(np.int32)
    test['seq_len'] = np.concatenate(test['seq_len'], axis=0)
    
    # Distribution of caption lengths
    len_dist = {}
    for l in np.sum(np.sign(train['targets']), axis=1):
        cap_len = int(l / word_len_thres)
        len_dist[cap_len] = len_dist.get(cap_len, 0) + 1
    
    print("\nINFO: Word length distribution:\n%r" % word_len_dist)
    print("\nINFO: Caption length distribution:\n%r" % len_dist)
    
    return train, valid, test


def raw_captions_to_char_ids(tokenised_dataset,
                             ctoi,
                             caption_len_thres,
                             include_restval=True):
    """
    Builds the train, validation and test dictionaries.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        ctoi: Character-to-id dictionary.
        caption_len_thres: Threshold for sentence length in characters.
            Captions with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    #print("INFO: Converting tokenised words to ids.\n")
    
    pattern = re.compile(r'[^a-zA-Z0-9]+', re.UNICODE)
    eol_pattern = re.compile(r'[ ]+$', re.UNICODE)
    train = dict(inputs=[],
                 targets=[],
                 reverse_targets=[],
                 caption_id=[],
                 filepaths=[],
                 image_id=[],
                 seq_len=[])
    valid = copy.deepcopy(train)
    test = copy.deepcopy(train)
    cap_id = 0
    
    for i, d in enumerate(tqdm(tokenised_dataset, ncols=70, desc="Char-to-id")):
        split = d['split']
        if include_restval is False and split == 'restval':
            continue
        sent_list_new = []
        for s in d['raw']:
            s = re.sub(pattern, ' ', s.lower())
            s = re.sub(eol_pattern, '', s)
            l = [ctoi[c] for c in s]
            l = l[:caption_len_thres]
            # Pad captions to predefined length with <PAD>.
            cap_np = np.pad(np.array(l),
                            (0, caption_len_thres - len(l)),
                            'constant',
                            constant_values=(0, ctoi['<PAD>']))
            sent_list_new.append(cap_np)
        # Add <GO> and <EOS> to captions.
        _ = _process_captions(np.array(sent_list_new), ctoi)
        inputs, targets, seq_len, reverse_targets = _
        
        if split == 'val':
            valid['inputs'].append(inputs)
            valid['targets'].append(targets)
            valid['reverse_targets'].append(reverse_targets)
            valid['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                valid['filepaths'].append(d['filepath'])
                valid['image_id'].append(d['image_id'])
                valid['caption_id'].append(cap_id)
                cap_id += 1
        elif split == 'test':
            test['inputs'].append(inputs)
            test['targets'].append(targets)
            test['reverse_targets'].append(reverse_targets)
            test['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                test['filepaths'].append(d['filepath'])
                test['image_id'].append(d['image_id'])
                test['caption_id'].append(cap_id)
                cap_id += 1
        else:
            train['inputs'].append(inputs)
            train['targets'].append(targets)
            train['reverse_targets'].append(reverse_targets)
            train['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                train['filepaths'].append(d['filepath'])
                train['image_id'].append(d['image_id'])
                train['caption_id'].append(cap_id)
                cap_id += 1

    # Sort training data according to caption length
    
    train['inputs'] = np.concatenate(train['inputs'], axis=0)
    train['targets'] = np.concatenate(train['targets'], axis=0)
    train['reverse_targets'] = np.concatenate(train['reverse_targets'], axis=0)
    train['caption_id'] = np.array(train['caption_id']).astype(np.int32)
    train['seq_len'] = np.concatenate(train['seq_len'], axis=0)
    sorted_seq_len_idx_list = sorted(range(len(train['seq_len'])),
                                     key=lambda x:train['seq_len'][x])
    train['inputs'] = train['inputs'][sorted_seq_len_idx_list, :]
    train['targets'] = train['targets'][sorted_seq_len_idx_list, :]
    train['reverse_targets'] = train['reverse_targets'][sorted_seq_len_idx_list, :]
    train['caption_id'] = train['caption_id'][sorted_seq_len_idx_list]
    train['seq_len'] = train['seq_len'][sorted_seq_len_idx_list]
    train['filepaths'] = [train['filepaths'][i] for i in sorted_seq_len_idx_list]
    
    valid['inputs'] = np.concatenate(valid['inputs'], axis=0)
    valid['targets'] = np.concatenate(valid['targets'], axis=0)
    valid['reverse_targets'] = np.concatenate(valid['reverse_targets'], axis=0)
    valid['caption_id'] = np.array(valid['caption_id']).astype(np.int32)
    valid['seq_len'] = np.concatenate(valid['seq_len'], axis=0)
    
    test['inputs'] = np.concatenate(test['inputs'], axis=0)
    test['targets'] = np.concatenate(test['targets'], axis=0)
    test['reverse_targets'] = np.concatenate(test['reverse_targets'], axis=0)
    test['caption_id'] = np.array(test['caption_id']).astype(np.int32)
    test['seq_len'] = np.concatenate(test['seq_len'], axis=0)
    
    # Distribution of caption lengths
    
    bins = range(0, max(train['seq_len']), 20) + [max(train['seq_len'])]
    hist, bin_edges = np.histogram(train['seq_len'], bins=bins)
    
    print("\nINFO: Caption length distribution (in characters):\n")
    print("Bin edges: %r" % list(bin_edges))
    print("Amounts: %r" % list(hist))
    
    return train, valid, test


def clean_captions_to_char_ids(tokenised_dataset,
                               ctoi,
                               caption_len_thres,
                               include_restval=True):
    """
    Builds the train, validation and test dictionaries.
    This function converts captions containing ONLY alphanumeric characters
    into character IDs.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        ctoi: Character-to-id dictionary.
        caption_len_thres: Threshold for sentence length in characters.
            Captions with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    #print("INFO: Converting tokenised words to ids.\n")
    
    train = dict(inputs=[],
                 targets=[],
                 reverse_targets=[],
                 caption_id=[],
                 filepaths=[],
                 image_id=[],
                 seq_len=[])
    valid = copy.deepcopy(train)
    test = copy.deepcopy(train)
    cap_id = 0
    
    for i, d in enumerate(tqdm(tokenised_dataset, ncols=70, desc="Char-to-id")):
        split = d['split']
        if include_restval is False and split == 'restval':
            continue
        sent_list_new = []
        for s in d['raw']:
            l = [ctoi[c] for c in s]
            l = l[:caption_len_thres]
            # Pad captions to predefined length with <PAD>.
            cap_np = np.pad(np.array(l),
                            (0, caption_len_thres - len(l)),
                            'constant',
                            constant_values=(0, ctoi['<PAD>']))
            sent_list_new.append(cap_np)
        # Add <GO> and <EOS> to captions.
        _ = _process_captions(np.array(sent_list_new), ctoi)
        inputs, targets, seq_len, reverse_targets = _
        
        if split == 'val':
            valid['inputs'].append(inputs)
            valid['targets'].append(targets)
            valid['reverse_targets'].append(reverse_targets)
            valid['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                valid['filepaths'].append(d['filepath'])
                valid['image_id'].append(d['image_id'])
                valid['caption_id'].append(cap_id)
                cap_id += 1
        elif split == 'test':
            test['inputs'].append(inputs)
            test['targets'].append(targets)
            test['reverse_targets'].append(reverse_targets)
            test['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                test['filepaths'].append(d['filepath'])
                test['image_id'].append(d['image_id'])
                test['caption_id'].append(cap_id)
                cap_id += 1
        else:
            train['inputs'].append(inputs)
            train['targets'].append(targets)
            train['reverse_targets'].append(reverse_targets)
            train['seq_len'].append(seq_len)
            for k in range(inputs.shape[0]):
                train['filepaths'].append(d['filepath'])
                train['image_id'].append(d['image_id'])
                train['caption_id'].append(cap_id)
                cap_id += 1

    # Sort training data according to caption length
    
    train['inputs'] = np.concatenate(train['inputs'], axis=0)
    train['targets'] = np.concatenate(train['targets'], axis=0)
    train['reverse_targets'] = np.concatenate(train['reverse_targets'], axis=0)
    train['caption_id'] = np.array(train['caption_id']).astype(np.int32)
    train['seq_len'] = np.concatenate(train['seq_len'], axis=0)
    sorted_seq_len_idx_list = sorted(range(len(train['seq_len'])),
                                     key=lambda x:train['seq_len'][x])
    train['inputs'] = train['inputs'][sorted_seq_len_idx_list, :]
    train['targets'] = train['targets'][sorted_seq_len_idx_list, :]
    train['reverse_targets'] = train['reverse_targets'][sorted_seq_len_idx_list, :]
    train['caption_id'] = train['caption_id'][sorted_seq_len_idx_list]
    train['seq_len'] = train['seq_len'][sorted_seq_len_idx_list]
    train['filepaths'] = [train['filepaths'][i] for i in sorted_seq_len_idx_list]
    
    valid['inputs'] = np.concatenate(valid['inputs'], axis=0)
    valid['targets'] = np.concatenate(valid['targets'], axis=0)
    valid['reverse_targets'] = np.concatenate(valid['reverse_targets'], axis=0)
    valid['caption_id'] = np.array(valid['caption_id']).astype(np.int32)
    valid['seq_len'] = np.concatenate(valid['seq_len'], axis=0)
    
    test['inputs'] = np.concatenate(test['inputs'], axis=0)
    test['targets'] = np.concatenate(test['targets'], axis=0)
    test['reverse_targets'] = np.concatenate(test['reverse_targets'], axis=0)
    test['caption_id'] = np.array(test['caption_id']).astype(np.int32)
    test['seq_len'] = np.concatenate(test['seq_len'], axis=0)
    
    # Distribution of caption lengths
    
    bins = range(0, max(train['seq_len']), 20) + [max(train['seq_len'])]
    hist, bin_edges = np.histogram(train['seq_len'], bins=bins)
    
    print("\nINFO: Caption length distribution (in characters):\n")
    print("Bin edges: %r" % list(bin_edges))
    print("Amounts: %r" % list(hist))
    
    return train, valid, test


def split_training_data(train,
                        splits=[10, 12, 14]):
    """
    Splits training data into 4 chunks according to caption length in words.
    
    Args:
        train: `train` dictionary from output of `tokenised_word_to_ids()`.
        splits: List of length 3. Lower-bounds of chunk 2 to chunk 4.
    
    Returns:
        A tuple of size 4, (train_0, train_1, train_2, train_3).
    """
    print("\nINFO: Splitting training data into 4 chunks.\n")
    
    chunk_0_cond = train['seq_len'] <= splits[0] - 1
    chunk_1_cond = ((splits[0] <= train['seq_len'])
                        & (train['seq_len'] <= splits[1] - 1))
    chunk_2_cond = ((splits[1] <= train['seq_len'])
                        & (train['seq_len'] <= splits[2] - 1))
    chunk_3_cond = train['seq_len'] >= splits[2]
    
    assert(chunk_0_cond.sum() + chunk_1_cond.sum() \
           + chunk_2_cond.sum() + chunk_3_cond.sum() == len(train['filepaths']))
    print("\tChunk 1 getting %8d caption pairs." % chunk_0_cond.sum())
    print("\tChunk 2 getting %8d caption pairs." % chunk_1_cond.sum())
    print("\tChunk 3 getting %8d caption pairs." % chunk_2_cond.sum())
    print("\tChunk 4 getting %8d caption pairs." % chunk_3_cond.sum())
    
    train_0 = {}
    train_1 = {}
    train_2 = {}
    train_3 = {}
    
    train_0['seq_len'] = train['seq_len'][chunk_0_cond]
    max_len = np.max(train_0['seq_len'])
    train_0['inputs'] = train['inputs'][chunk_0_cond, :max_len]
    train_0['targets'] = train['targets'][chunk_0_cond, :max_len]
    train_0['reverse_targets'] = train['reverse_targets'][chunk_0_cond, :max_len]
    train_0['caption_id'] = train['caption_id'][chunk_0_cond]
    train_0['filepaths'] = [path for i, path in enumerate(train['filepaths'])
                            if chunk_0_cond[i] == True]
    
    train_1['seq_len'] = train['seq_len'][chunk_1_cond]
    max_len = np.max(train_1['seq_len'])
    train_1['inputs'] = train['inputs'][chunk_1_cond, :max_len]
    train_1['targets'] = train['targets'][chunk_1_cond, :max_len]
    train_1['reverse_targets'] = train['reverse_targets'][chunk_1_cond, :max_len]
    train_1['caption_id'] = train['caption_id'][chunk_1_cond]
    train_1['filepaths'] = [path for i, path in enumerate(train['filepaths'])
                            if chunk_1_cond[i] == True]
    
    train_2['seq_len'] = train['seq_len'][chunk_2_cond]
    max_len = np.max(train_2['seq_len'])
    train_2['inputs'] = train['inputs'][chunk_2_cond, :max_len]
    train_2['targets'] = train['targets'][chunk_2_cond, :max_len]
    train_2['reverse_targets'] = train['reverse_targets'][chunk_2_cond, :max_len]
    train_2['caption_id'] = train['caption_id'][chunk_2_cond]
    train_2['filepaths'] = [path for i, path in enumerate(train['filepaths'])
                            if chunk_2_cond[i] == True]
    
    train_3['seq_len'] = train['seq_len'][chunk_3_cond]
    train_3['inputs'] = train['inputs'][chunk_3_cond, :]
    train_3['targets'] = train['targets'][chunk_3_cond, :]
    train_3['reverse_targets'] = train['reverse_targets'][chunk_3_cond, :]
    train_3['caption_id'] = train['caption_id'][chunk_3_cond]
    train_3['filepaths'] = [path for i, path in enumerate(train['filepaths'])
                            if chunk_3_cond[i] == True]
    print('\n')
    return (train_0, train_1, train_2, train_3)


def output_files(train, valid, test,
                 wtoi, itow,
                 output_path, output_prefix, output_suffix,
                 output_dtype,
                 split_train_data=True,
                 build_vocab=True):
    """
    Writes the dictionaries to disk.
    
    Args:
        train: `train` dictionary from output of `tokenised_word_to_ids()`,
            or tuple from output of `split_training_data()`.
        valid: `valid` dictionary from output of `tokenised_word_to_ids()`.
        test: `test` dictionary from output of `tokenised_word_to_ids()`.
        wtoi: Word-to-id dictionary.
        itow: Id-to-word dictionary.
        word_count_thres: For naming purposes.
        caption_len_thres: For naming purposes.
        output_path: Directory in which to write the files.
        output_prefix: String. Prefix for the output h5 file name.
        output_dtype: One of 'int8', 'int16' or 'int32'.
        split_train_data: Boolean. Whether training data has been splitted.
        retokenise: Boolean. Whether the raw captions have been retokenised 
            using Stanford-CoreNLP-3.4.1.
        build_vocab: Boolean. Whether to rebuild vocabulary.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    """
    #print("\nINFO: Saving h5 file.\n")
    
    assert output_dtype in ['int8', 'int16', 'int32']
    dtype = output_dtype
    num_val = len(valid['filepaths'])
    num_test = len(test['filepaths'])
    if split_train_data:
        train_0, train_1, train_2, train_3 = train
        num_train = sum([len(t['filepaths']) for t in train])
    else:
        num_train = len(train['filepaths'])
    
    if build_vocab:
        with open('%s/%s_wtoi_%s.json' %
                  (output_path, output_prefix, output_suffix), 'w') as f:
            json.dump(wtoi, f)
        with open('%s/%s_itow_%s.json' %
                  (output_path, output_prefix, output_suffix), 'w') as f:
            json.dump(itow, f)
    
    with h5py.File('%s/%s_%s.h5' %
                   (output_path, output_prefix, output_suffix), 'w') as f:
        dt = h5py.special_dtype(vlen=unicode)
        
        f.create_dataset("info/vocab_size", dtype='int32',
                         data=len(itow))
        f.create_dataset("info/top_1k_words_ids", dtype='int32',
                         data=np.arange(15, 1015))
        
        f.create_dataset("valid/inputs", dtype=dtype,
                         data=valid['inputs'])
        f.create_dataset("valid/targets", dtype=dtype,
                         data=valid['targets'])
        f.create_dataset("valid/reverse_targets", dtype=dtype,
                         data=valid['reverse_targets'])
        #f.create_dataset("valid/caption_id", dtype=dtype, data=valid['caption_id'])
        
        f.create_dataset("test/inputs", dtype=dtype,
                         data=test['inputs'])
        f.create_dataset("test/targets", dtype=dtype,
                         data=test['targets'])
        f.create_dataset("test/reverse_targets", dtype=dtype,
                         data=test['reverse_targets'])
        #f.create_dataset("test/caption_id", dtype=dtype, data=test['caption_id'])
        
        if split_train_data:
            f.create_dataset("train_0/inputs", dtype=dtype,
                             data=train_0['inputs'])
            f.create_dataset("train_0/targets", dtype=dtype,
                             data=train_0['targets'])
            f.create_dataset("train_0/reverse_targets", dtype=dtype,
                             data=train_0['reverse_targets'])
            #f.create_dataset("train_0/caption_id", dtype=dtype, data=train_0['caption_id'])
            
            f.create_dataset("train_1/inputs", dtype=dtype,
                             data=train_1['inputs'])
            f.create_dataset("train_1/targets", dtype=dtype,
                             data=train_1['targets'])
            f.create_dataset("train_1/reverse_targets", dtype=dtype,
                             data=train_1['reverse_targets'])
            #f.create_dataset("train_1/caption_id", dtype=dtype, data=train_1['caption_id'])
            
            f.create_dataset("train_2/inputs", dtype=dtype,
                             data=train_2['inputs'])
            f.create_dataset("train_2/targets", dtype=dtype,
                             data=train_2['targets'])
            f.create_dataset("train_2/reverse_targets", dtype=dtype,
                             data=train_2['reverse_targets'])
            #f.create_dataset("train_2/caption_id", dtype=dtype, data=train_2['caption_id'])
            
            f.create_dataset("train_3/inputs", dtype=dtype,
                             data=train_3['inputs'])
            f.create_dataset("train_3/targets", dtype=dtype,
                             data=train_3['targets'])
            f.create_dataset("train_3/reverse_targets", dtype=dtype,
                             data=train_3['reverse_targets'])
            #f.create_dataset("train_3/caption_id", dtype=dtype, data=train_3['caption_id'])
        else:
            f.create_dataset("train/inputs", dtype=dtype,
                             data=train['inputs'])
            f.create_dataset("train/targets", dtype=dtype,
                             data=train['targets'])
            f.create_dataset("train/reverse_targets", dtype=dtype,
                             data=train['reverse_targets'])
            #f.create_dataset("train/caption_id", dtype=dtype, data=train['caption_id'])
        
        dset_valid = f.create_dataset("valid/image_paths", (num_val,), dtype=dt)
        dset_test = f.create_dataset("test/image_paths", (num_test,), dtype=dt)
        if split_train_data:
            dset_tr_0 = f.create_dataset("train_0/image_paths",
                                         (len(train_0['filepaths']),), dtype=dt)
            dset_tr_1 = f.create_dataset("train_1/image_paths",
                                         (len(train_1['filepaths']),), dtype=dt)
            dset_tr_2 = f.create_dataset("train_2/image_paths",
                                         (len(train_2['filepaths']),), dtype=dt)
            dset_tr_3 = f.create_dataset("train_3/image_paths",
                                         (len(train_3['filepaths']),), dtype=dt)
        else:
            dset_tr = f.create_dataset("train/image_paths", (num_train,), dtype=dt)
        
        for i, fp in enumerate(tqdm(valid['filepaths'], ncols=70, desc="Saved validation")):
            dset_valid[i] = fp
        for i, fp in enumerate(tqdm(test['filepaths'], ncols=70, desc="Saved test")):
            dset_test[i] = fp
        
        if split_train_data:
            for i, fp in enumerate(tqdm(train_0['filepaths'], ncols=70, desc="Saved train-0")):
                dset_tr_0[i] = fp
            for i, fp in enumerate(tqdm(train_1['filepaths'], ncols=70, desc="Saved train-1")):
                dset_tr_1[i] = fp
            for i, fp in enumerate(tqdm(train_2['filepaths'], ncols=70, desc="Saved train-2")):
                dset_tr_2[i] = fp
            for i, fp in enumerate(tqdm(train_3['filepaths'], ncols=70, desc="Saved train-3")):
                dset_tr_3[i] = fp
        else:
            for i, fp in enumerate(tqdm(train['filepaths'], ncols=70, desc="Saved train")):
                dset_tr[i] = fp
    
    print("\n\nINFO: File writing completed.\n\n")


def output_files_char(train, valid, test,
                      ctoi, itoc,
                      word_len_thres, caption_len_thres,
                      output_path, output_prefix,
                      split_train_data=True,
                      retokenise=False,
                      build_vocab=True,
                      include_restval=True):
    """
    Writes the dictionaries to disk.
    
    NOTE: This is the character-based version of `output_files()`.
    All data will be saved as `uint8`.
    
    Args:
        train: `train` dictionary from output of `tokenised_word_to_ids()`,
            or tuple from output of `split_training_data()`.
        valid: `valid` dictionary from output of `tokenised_word_to_ids()`.
        test: `test` dictionary from output of `tokenised_word_to_ids()`.
        ctoi: Character-to-id dictionary.
        itoc: Id-to-character dictionary.
        word_len_thres: Length of a word (number of characters).
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        output_path: Directory in which to write the files.
        output_prefix: String. Prefix for the output h5 file name.
        split_train_data: Boolean. Whether training data has been splitted.
        retokenise: Boolean. Whether the raw captions have been retokenised 
            using Stanford-CoreNLP-3.4.1.
        build_vocab: Boolean. Whether to rebuild vocabulary.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    """
    #print("\nINFO: Saving h5 file.\n")
    num_val = len(valid['filepaths'])
    num_test = len(test['filepaths'])
    if split_train_data:
        train_0, train_1, train_2, train_3 = train
        num_train = sum([len(t['filepaths']) for t in train])
    else:
        num_train = len(train['filepaths'])
    
    suffix = []
    suffix.append('c%d_s%d' % (word_len_thres, caption_len_thres))
    if include_restval:
        suffix.append('include_restval')
    if split_train_data:
        suffix.append('split')
    if retokenise:
        suffix.append('retokenised')
    suffix = '_'.join(suffix)
    
    if build_vocab:
        with open('%s/%s_ctoi_%s.json' %
                  (output_path, output_prefix, suffix), 'w') as f:
            json.dump(ctoi, f)
        with open('%s/%s_itoc_%s.json' %
                  (output_path, output_prefix, suffix), 'w') as f:
            json.dump(itoc, f)
    
    with h5py.File('%s/%s_%s.h5' %
                   (output_path, output_prefix, suffix), 'w') as f:
        dt = h5py.special_dtype(vlen=unicode)
        
        f.create_dataset("info/vocab_size", dtype='uint8', data=len(itoc))
        
        f.create_dataset("valid/inputs", dtype='uint8', data=valid['inputs'])
        f.create_dataset("valid/targets", dtype='uint8', data=valid['targets'])
        f.create_dataset("valid/reverse_targets", dtype='uint8',
                         data=valid['reverse_targets'])
        f.create_dataset("valid/caption_id", dtype='uint8', data=valid['caption_id'])
        
        f.create_dataset("test/inputs", dtype='uint8', data=test['inputs'])
        f.create_dataset("test/targets", dtype='uint8', data=test['targets'])
        f.create_dataset("test/reverse_targets", dtype='uint8',
                         data=test['reverse_targets'])
        f.create_dataset("test/caption_id", dtype='uint8', data=test['caption_id'])
        
        if split_train_data:
            f.create_dataset("train_0/inputs", dtype='uint8', data=train_0['inputs'])
            f.create_dataset("train_0/targets", dtype='uint8', data=train_0['targets'])
            f.create_dataset("train_0/reverse_targets", dtype='uint8',
                             data=train_0['reverse_targets'])
            f.create_dataset("train_0/caption_id", dtype='uint8', data=train_0['caption_id'])
            
            f.create_dataset("train_1/inputs", dtype='uint8', data=train_1['inputs'])
            f.create_dataset("train_1/targets", dtype='uint8', data=train_1['targets'])
            f.create_dataset("train_1/reverse_targets", dtype='uint8',
                             data=train_1['reverse_targets'])
            f.create_dataset("train_1/caption_id", dtype='uint8', data=train_1['caption_id'])
            
            f.create_dataset("train_2/inputs", dtype='uint8', data=train_2['inputs'])
            f.create_dataset("train_2/targets", dtype='uint8', data=train_2['targets'])
            f.create_dataset("train_2/reverse_targets", dtype='uint8',
                             data=train_2['reverse_targets'])
            f.create_dataset("train_2/caption_id", dtype='uint8', data=train_2['caption_id'])
            
            f.create_dataset("train_3/inputs", dtype='uint8', data=train_3['inputs'])
            f.create_dataset("train_3/targets", dtype='uint8', data=train_3['targets'])
            f.create_dataset("train_3/reverse_targets", dtype='uint8',
                             data=train_3['reverse_targets'])
            f.create_dataset("train_3/caption_id", dtype='uint8', data=train_3['caption_id'])
        else:
            f.create_dataset("train/inputs", dtype='uint8', data=train['inputs'])
            f.create_dataset("train/targets", dtype='uint8', data=train['targets'])
            f.create_dataset("train/reverse_targets", dtype='uint8',
                             data=train['reverse_targets'])
            f.create_dataset("train/caption_id", dtype='uint8', data=train['caption_id'])
        
        dset_valid = f.create_dataset("valid/image_paths", (num_val,), dtype=dt)
        dset_test = f.create_dataset("test/image_paths", (num_test,), dtype=dt)
        if split_train_data:
            dset_tr_0 = f.create_dataset("train_0/image_paths",
                                         (len(train_0['filepaths']),), dtype=dt)
            dset_tr_1 = f.create_dataset("train_1/image_paths",
                                         (len(train_1['filepaths']),), dtype=dt)
            dset_tr_2 = f.create_dataset("train_2/image_paths",
                                         (len(train_2['filepaths']),), dtype=dt)
            dset_tr_3 = f.create_dataset("train_3/image_paths",
                                         (len(train_3['filepaths']),), dtype=dt)
        else:
            dset_tr = f.create_dataset("train/image_paths", (num_train,), dtype=dt)
        
        for i, fp in enumerate(tqdm(valid['filepaths'], ncols=70, desc="Saved validation")):
            dset_valid[i] = fp
        for i, fp in enumerate(tqdm(test['filepaths'], ncols=70, desc="Saved test")):
            dset_test[i] = fp
        
        if split_train_data:
            for i, fp in enumerate(tqdm(train_0['filepaths'], ncols=70, desc="Saved train-0")):
                dset_tr_0[i] = fp
            for i, fp in enumerate(tqdm(train_1['filepaths'], ncols=70, desc="Saved train-1")):
                dset_tr_1[i] = fp
            for i, fp in enumerate(tqdm(train_2['filepaths'], ncols=70, desc="Saved train-2")):
                dset_tr_2[i] = fp
            for i, fp in enumerate(tqdm(train_3['filepaths'], ncols=70, desc="Saved train-3")):
                dset_tr_3[i] = fp
        else:
            for i, fp in enumerate(tqdm(train['filepaths'], ncols=70, desc="Saved train")):
                dset_tr[i] = fp
    
    print("\n\nINFO: File writing completed.\n\n")


