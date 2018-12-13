# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:28:14 2018

@author: Dwyane Wade
"""

import sph_stub as sphClass

# Initialise grid
domain = sphClass.SPH_main()
domain.set_values()
domain.initialise_grid()
domain.place_points()
domain.allocate_to_grid()
print("Done before simulation")
count = domain.simulate(10)
print("Done simulation")

file = open('countnum.txt','w')
file.write(str(count))
file.close()

print("Done savefile")