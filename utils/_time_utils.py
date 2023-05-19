# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: utils for time

"""

import datetime

def get_timestamp(option_ = 0):
    now = datetime.datetime.now()
    h = int(now.strftime("%H"))
    m = int(now.strftime("%M"))
    ss = int(now.strftime("%S"))
    s = now.strftime("%Y/%2m/%2d")
    s = str(s)
    if m<10:
        m = '0' + str(m)
    if h>12:
        s += "*%2d_%2s_%2dPM"%(h - 12,m,ss)
    elif h == 12:
        s += "*%2d_%2s_%2dPM"%(h ,m,ss)
    else:
        s += "*%2d_%2s_%2dAM"%(h,m,ss)
    s = s.replace(' ','0')
    if option_==0:return s.replace('/','_').replace(':','_').replace(' ','_')
    elif option_ == 1:return s
    elif option_ == 2:return ' ***running at: '+ str(s)