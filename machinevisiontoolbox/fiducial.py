

# import cv2

# _aruco_dict = {
#     "4x4_50", cv2.DICT_4X4_50 = 0, 
#     "4x4_100", cv2.DICT_4X4_100, 
#     "4x4_250", cv2.DICT_4X4_250, 
#     "4x4_1000", cv2.DICT_4X4_1000, 
#     "5x5_50", cv2.DICT_5X5_50, 
#     "5x5_100", cv2.DICT_5X5_100, 
#     "5x5_250", cv2.DICT_5X5_250, 
#     "5x5_1000", cv2.DICT_5X5_1000, 
#     "6x6_50", cv2.DICT_6X6_50, 
#     "6x6_100", cv2.DICT_6X6_100, 
#     "6x6_250", cv2.DICT_6X6_250, 
#     "6x6_1000", cv2.DICT_6X6_1000, 
#     "7x7_50", cv2.DICT_7X7_50, 
#     "7x7_100", cv2.DICT_7X7_100, 
#     "7x7_250", cv2.DICT_7X7_250, 
#     "7x7_1000", cv2.DICT_7X7_1000, 
#     original, cv2.DICT_ARUCO_ORIGINAL,
# }

# _april_dict = {
#     "16h5", cv2.DICT_APRILTAG_16h5, 
#     "25h9", cv2.DICT_APRILTAG_25h9, 
#     "36h10", cv2.DICT_APRILTAG_36h10, 
#     "36h11", cv2.DICT_APRILTAG_36h11,
# }




# class Fiducial2dMixin(Feature2d):


#     def arucotag(self, dict="4x4_1000", K=None, side=None):

#         dictionary = cv2.aruco.getPredefinedDictionary(_aruco_dict[dict])
#         markers, ids, _ = cv2.aruco.detectMarkers(scene.image, dictionary)

#         if K is not None and side is not None:
#             rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markers, side, K, None)
#             for rvec, tvec in zip(rvec, tvecs):
#                 T = SE3(tvec) * SE3.EulerVec(rvec.flatten())


#     def apriltag(self):
#         pass