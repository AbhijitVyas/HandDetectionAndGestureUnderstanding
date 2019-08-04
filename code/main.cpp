#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>
#include <iostream>
#include <ctype.h>

#include "main.hpp"
#include "OpenCVKinect.h"
  
using namespace cv;
using namespace std;

#include "opencv2/opencv.hpp"
#include <string>

using namespace cv;
using namespace std;

#include "MainMethodClass.h"
 
#include "Kalman.hpp"
#include "GestureRecognizer.hpp"

/**
* This main method initialize endless for loop and initialize variables for starting kinect based color and depth image based hand
* segmentations and later tracking hand positions in 2D space and undnerstanding gestures created based on those points, classified by SVM
* classifier
*/ 
int main( int argc, char** argv )
{
	   
	//Kinect data 
	OpenCVKinect myDataCap;
	 
	if(!myDataCap.init())
	{
		std::cout << "Error initializing" << std::endl;
		return 1;
	}
	 
	//timer initialization
	clock_t initial, final, intemediate, intemediate1,frameRateInitial,frameRateInitialFinal;
  
	// >>>> Kalman Filter
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
 
	unsigned int type = CV_32F;
	cv::KalmanFilter kfLeft(stateSize, measSize, contrSize, type);
	cv::KalmanFilter kfRight(stateSize, measSize, contrSize, type);
	MainMethodClass mmc(kfLeft,kfRight);
	mmc.createTrackbars();
	frameRateInitial = clock();
	int count = 0;
	for(;;)
    {
		initial = clock();
	 	dataStream = myDataCap.getData();
		Point3i leftHandCoordinates, rightHandCoordinates;
		mmc.run(dataStream[C_COLOR_STREAM],dataStream[C_DEPTH_STREAM],myDataCap,frameRate,leftHandCoordinates, rightHandCoordinates);
		cout<<"Left : "<<leftHandCoordinates<<endl;
		cout<<"Right : "<<rightHandCoordinates<<endl;
		final = clock() - initial; 

		if(final != 0)
		frameRate = 1000 / final;
		 
		cout<< "final time : "<<(double)final<<endl<<endl;
        char c = (char)waitKey(1);
       
    }	  
    return 0;
}
