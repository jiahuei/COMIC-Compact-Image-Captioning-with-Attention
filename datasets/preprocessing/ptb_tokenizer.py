#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
#import sys
import subprocess
import tempfile
import re
#import itertools

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

class PTBTokenizer:
    """
    Python wrapper of Stanford PTBTokenizer.
    
    The tokeniser will split " xxx's " into " xxx 's ".
        Thus, words like " chef's " (one word)
         will become " chef 's " (two words).
    """

    def tokenize(self, sentences_list, remove_non_alphanumerics=False):
        pattern = re.compile(r'([^\s\w]|_)+', re.UNICODE)                       # matches non-alphanumerics
        #pattern = re.compile(r'([^\w]|_)+', re.UNICODE)                        # matches non-alphanumerics and whitespaces
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        sentences = '\n'.join([s.replace('\n', ' ') for s in sentences_list])

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences)
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        tokenized_caption_w_punc = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # remove punctuations and optionally non-alphanumerics
        # ======================================================
        tokenized_caption = []
        if remove_non_alphanumerics:
            for line in tokenized_caption_w_punc:
                line = ' '.join([w for w in line.rstrip().split(' ') \
                        if w not in PUNCTUATIONS])
                line = re.sub(pattern, '', line)
                tokenized_caption.append(line)
        else:
            for line in tokenized_caption_w_punc:
                tokenized_caption.append(' '.join([w for w in line.rstrip().split(' ') \
                        if w not in PUNCTUATIONS]))

        return tokenized_caption, tokenized_caption_w_punc
