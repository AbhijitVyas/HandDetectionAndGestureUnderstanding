#ifndef _KALMAN_FILTER_
#define _KALMAN_FILTER_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv\cv.h>
#include <Windows.h>


using namespace cv;
using namespace std;

const Scalar COLOR_YELLOW_AS_MEASURMENT      = Scalar(0,128,200);
const Scalar COLOR_BLACK_AS_PREDICTION         = Scalar(0,0,0);

 
#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

class Kalman{
	 
	 // Image to show mouse tracking
	cv::Mat img;
	vector<Point> measurementRight,kalmanvRight;
	vector<Point> measurementLeft,kalmanvLeft;

	// >>>> Kalman Filter initialization parameters
	int stateSize;
	int measSize ;
	int contrSize; 
	unsigned int type;

	cv::Mat stateLeft;
	cv::Mat stateRight;
	cv::Mat measLeft;
	cv::Mat measRight;
	cv::Mat imageLeft;
	cv::Mat imageRight;

	// for transition matrix
	double ticksLeft;
	double ticksRight;

	//not found countor
	int notFoundCountLeft;
	int notFoundCountRight;

	bool firstTimeFlagLeft;
	bool firstTimeFlagRight;
public:
	void getKalmanPosition(KalmanFilter& KF,cv::Point& center);
	Kalman();
	Kalman(KalmanFilter& kfLeft,KalmanFilter& kfRight);

	void runLeft(KalmanFilter& kf,Point& measuredPoint,Rect& boundingBox,cv::Mat& imageForTracking,Rect& returnedRect,Point& returedKalmanPoint);
	void runRight(KalmanFilter& kf,Point& measuredPoint,Rect& boundingBox,cv::Mat& imageForTracking,Rect& returnedRect,Point& returedKalmanPoint);
};
#endif