# HandDetectionAndGestureUnderstanding
This project has working code example of hand detection and gesture understanding work done during my master thesis.

## Hand Position detection 
The project works with Kinect camera and uses color and depth image segmentations techniques to identify both hand's center of palm position in 3D space and saves these points for later gesture detection part.  

## Gesture Understanding
Saved 3D points were used to identify basic gestures from 0 to 9 digits and shape identifications like square, rectangle, triangle or circle. SVM classifier was used for classification and proper training data was recorded during the project work which is also part of the code! The prediction was measured by preconditions identified for each individual gesture and the classification/identification part was backed by SVM classifier.

## Getting started
`prerequisite` 
- OpenCV_2.x 
- OpenNI_2.x 
- KinectSDK_1.8

