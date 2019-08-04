#ifndef _HelperFunctions_
#define _HelperFunctions_
 
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv\cv.h>

#include <iostream>
#include <ctype.h>

#include "myImage.hpp"
 
#include "BackGroundSubtraction.hpp"
#include "Kalman.hpp"
using namespace std;
using namespace cv;

class HelperFunctions{
	int MINNOOFPIXELS ;
	HelperFunctions();
	void preFiltering(cv::Mat& srcMat);
	void methodForCOunterDetectionAndThresholding(cv::Mat& srcColorMat,cv::Mat& returnGrayMat);
	 
	int findBiggestContour(vector<vector<Point> > contours);
	 
	void stdDeviationModified(vector<Mat>& mask,vector<Mat>& filteredMatVec,int& retHueMin1,int& retHueMax1);
	void indAvgOfIntVector(vector<int>& srcVec,int& avg);
	void calHist(cv::Mat& srcMat,cv::Mat& mask);
	void calAndDrawHist(cv::Mat& srcMat,cv::Mat& mask);
	float findBiggestValueInHist(cv::Mat& srcMat);
	void mainFunction(cv::Mat& image, KalmanFilter& KF);
	 
	void createTrackbar();
};
#endif