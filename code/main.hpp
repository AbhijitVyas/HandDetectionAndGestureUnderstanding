#ifndef _MAIN_
#define _MAIN_

#define NOMINMAX
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>
#include <iostream>
#include <ctype.h>

#include <sys/timeb.h>
#include "HandDetection.hpp"
const float scaleFactor = 0.05f;

using namespace cv;
using namespace std;
IplImage* new_image;


//for kinect data 
cv::vector<cv::Mat> dataStream;

//global frame : scale down images
Mat colorImage;
Mat depthImage;

Point returnPoint;

// depth region grow
Rect selection;
bool selectObject = false;
float frameRate;

class main  {

public:
	
};
#endif