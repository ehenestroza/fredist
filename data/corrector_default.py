#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
Template: A user-modifiable file to quickly generate feature vectors for
parsing and correction.
"""

import sys
from dtbutils import *

class Template(object):

    def __init__(self):

        return


    #
    # Obtain feature vector given features corresponding to items s_X and n_X.
    #
    def feature_vector(self,s_0,s_1,s_2,n_0,n_1,n_2,n_3,d_0,c_0,c_d,fpos):

        return c_d[ISPR]+\
               d_0[LEM]+\
               d_0[FPS]+\
               d_0[OFPS]+\
               c_0[LEM]+\
               c_0[FPS]+\
               c_0[HFPS]+\
               c_d[LDIS]+\
               c_d[LDIR]+\
               c_d[GDIS]+\
               c_d[NDEP]+\
               c_d[MFPS]+\
               c_d[MLAB]+\
               c_d[PFPS]+\
               c_d[PLAB]+\
               c_d[PUNC]

