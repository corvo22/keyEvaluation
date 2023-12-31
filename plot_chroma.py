# coding: utf-8
"""
===================================
Enhanced chroma and chroma variants
===================================

This notebook demonstrates a variety of techniques for enhancing chroma features and 
also, introduces chroma variants implemented in librosa.
"""


###############################################################################################
#  
# Enhanced chroma
# ^^^^^^^^^^^^^^^
# Beyond the default parameter settings of librosa's chroma functions, we apply the following 
# enhancements:
#
#    1. Harmonic-percussive-residual source separation to eliminate transients.
#    2. Nearest-neighbor smoothing to eliminate passing tones and sparse noise.  This is inspired by the
#       recurrence-based smoothing technique of
#       `Cho and Bello, 2011 <https://zenodo.org/record/1417557>`_.
#    3. Local median filtering to suppress remaining discontinuities.

# Code source: Brian McFee
# License: ISC
# sphinx_gallery_thumbnail_number = 5

import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa


#######################################################################
# We'll use a track that has harmonic, melodic, and percussive elements

y, sr = librosa.load('mp3_files/californiacation+Key_Am.mp3')

y_harm = librosa.effects.harmonic(y=y, margin=8)
chroma_harm = librosa.feature.chroma_stft(y=y_harm, sr=sr)

chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))

print(np.mean(chroma_filter,axis=1))