# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:29:31 2018

@author: Dwyane Wade
"""


import  time
from progressbar import *
 
total = 1000
 
def dosomework():
    time.sleep(0.01)
 
widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=10*total).start()
for i in range(total):
    # do something
    pbar.update(10 * i + 1)
    dosomework()
