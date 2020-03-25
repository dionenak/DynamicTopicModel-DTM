# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:02:07 2020

@author: dio
"""
#Open/Load our dictionary and corpus
import pickle
fileObject = open("dict.pkl",'rb')  ##!!! Prepei na to kleisw meta!!
b = pickle.load(fileObject)
file=open("corp.pkl","rb")
c=pickle.load(file)

