# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# playground python module is used to test the module and debug.
# ==============================================================================
"""playground_sem python module is used to test the prediction of semantic segmentation from 
a RGB input image. This provides usage example of the semantically segmented image 
generation from the created interfaces in SemSeg module
"""

import cv2
from SemSeg import sem_seg

# semantic = sem_seg.get_semseg('examples/000031.png')
nw = sem_seg.get_network_semseg()
semantic, loc = sem_seg.predict_semseg(model=nw, original_img='SemSeg/examples/000031.png')
cv2.imshow("Predicted BEV", semantic)  # here RGB to BGR not done. So cars are red. 
# But fixed in actual saved location which will be used for evaluation or further processing
print(loc)
cv2.waitKey(0)