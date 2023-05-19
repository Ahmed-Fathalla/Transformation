# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: writting the output to file
"""
from ._config import output_path

def write_to_file(file_name, *str_):
    s = ''
    for arg in str_:
        s += str(arg)
        if str(arg) == '\n':pass
        else:s = s  + ' '
    with open(output_path+'%s.txt'%file_name, 'a') as myfile:myfile.write(s+'\n')
    print(s)