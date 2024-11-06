# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# playground python module is used to test the prediction of bev from 
# a semantically segmented input image.
# ==============================================================================
"""playground_bev python module is used to test the prediction of bev from 
a semantically segmented input image. This provides usage example of the bev 
generation from the created interfaces in Cam2BEV
"""
import cv2
from Cam2BEV import predict_bev
from Cam2BEV import get_ipm

# ==============================================================================
# usage examples
# ==============================================================================
model, label = predict_bev.get_bev_network(backbone='unetxst', 
                                           weights_dir='Cam2BEV/model/output/unetxst_singlecam/finetuned/Checkpoints/best_weights.hdf5')
bev, loc = predict_bev.predict_bev(model=model, one_hotd_label=label, 
                                   sem_img='Cam2BEV/examples/SemSegLog-014.png',
                                   backbone='unetxst')   # works super well with deeplab-mobilenet
cv2.imshow("Predicted BEV", bev)  # test predict_bev.py
print(loc)
cv2.waitKey(0)

model, label = predict_bev.get_bev_network()
bev2, loc2 = predict_bev.predict_bev(model=model, one_hotd_label=label,
                                     sem_img='Cam2BEV/examples/SemSegLog-014.png')
cv2.imshow("Predicted BEV", bev2)  # test predict_bev.py
print(loc2)
cv2.waitKey(0)
# ==============================================================================

# ==============================================================================
# ipm, loc = get_ipm.get_homography('SemSeg/seg_output/2023-08-22/SemSegLog-012.png')
# ipm, loc = get_ipm.get_homography('Cam2BEV/examples/SemSegLog-014.png')
# ipm, loc = get_ipm.get_homography('Cam2BEV/examples/v_0_0049500.png')
# cv2.imshow("Homography", ipm)
# print(loc)
# cv2.waitKey(0)
# ==============================================================================