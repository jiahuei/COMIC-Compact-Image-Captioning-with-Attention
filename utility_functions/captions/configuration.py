# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
from time import localtime, strftime


class Config(object):
    """ Configuration object."""
    def __init__(self,
                 data_paths,
                 **kwargs):
        
        self.attr_list = []
        for key, value in sorted(kwargs.iteritems()):
            self._attr(key, value)
        for key, value in sorted(data_paths.iteritems()):
            self._attr(key, value)


    def _attr(self, name, value):
        setattr(self, name, value)
        self.attr_list.append(name)


    def save_config_to_file(self):
        f_dump = ["%s = %s" % (k, self.__dict__[k]) for k in self.attr_list]
        config_name = 'config___%s.txt' % strftime("%m-%d_%H-%M", localtime())
        with open(os.path.join(self.log_path, config_name), 'w') as f:
            f.write('\r\n'.join(f_dump))
        with open(os.path.join(self.log_path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


    def overwrite_safety_check(self, overwrite):
        """ Exits if log_path exists but 'overwrite' is set to 'False'."""
        path_exists = os.path.exists(self.log_path)
        if path_exists: 
            if not overwrite:
                print("\nINFO: log_path already exists. "
                      "Set `overwrite` to True? Exiting now.")
                raise SystemExit
            else: print("\nINFO: log_path already exists. "
                        "The directory will be overwritten.")
        else: 
            print("\nINFO: log_path does not exist. "
                  "The directory will be created.")
            #os.mkdir(self.log_path)
            os.makedirs(self.log_path)

    '''

        
        self.data_paths = []
        for key, value in sorted(kwargs.iteritems()):
            setattr(self, key, value)
        for key, value in sorted(data_paths.iteritems()):
            setattr(self, key, value)
            self.data_paths.append(key)


    def save_config_to_file(self):
        data_paths = set(self.data_paths)
        attributes = set(self.__dict__.keys()) - data_paths
        f_dump = ["%s = %s" % (k, self.__dict__[k]) for k in attributes]
        f_dump += ["%s = %s" % (k, self.__dict__[k]) for k in data_paths]
        config_name = 'config___%s.txt' % strftime("%m-%d_%H-%M", localtime())
        with open(os.path.join(self.log_path, config_name), 'w') as f:
            f.write('\r\n'.join(f_dump))
        with open(os.path.join(self.log_path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    '''

