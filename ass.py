# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:19:57 2019

@author: dio
"""
import json 

with open('RC_2018-03') as fp:
    for line in fp:
        comment = json.loads(line)
        print(comment)