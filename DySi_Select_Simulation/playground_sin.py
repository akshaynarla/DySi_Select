# !/usr/bin/env python3
# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# playground python module is used to test any interfaces and
# is purely for debugging purposes. This is not a part of the SIN module.
# ==============================================================================
"""playground_sin python module is used to test the prediction of situation class from 
a bev input image. This provides usage example of the situation class prediction 
from the created interfaces in SIN
"""

from SIN.src.utils.data_utils import load_image
from SIN.src.predict_situation import predict_situation, get_network_sin

# ==============================================================
# Usage Example
# ==============================================================
img = load_image('SIN/examples/v_0_0013000.png')  # here location of predicted BEV image
sin = get_network_sin()
output = predict_situation(model=sin, bev_img=img)
print("Prediction from SIN:", output)