
#include "Kalman.hpp"

Kalman::Kalman(KalmanFilter& kfLeft,KalmanFilter& kfRight){
	

    firstTimeFlagLeft = false;
	firstTimeFlagRight = false;

	img = Mat::zeros(120,160,CV_8UC3);

   // >>>> Kalman Filter
   stateSize = 6;
   measSize = 4;
   contrSize = 0;
   type = CV_32F;

   stateLeft.create(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
   measLeft.create(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

   stateRight.create(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
   measRight.create(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

   //cv::Mat procNoise(stateSize, 1, type)
   // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]
 
   // Transition State Matrix A
   // Note: set dT at each processing step!
   // [ 1 0 dT 0  0 0 ]
   // [ 0 1 0  dT 0 0 ]
   // [ 0 0 1  0  0 0 ]
   // [ 0 0 0  1  0 0 ]
   // [ 0 0 0  0  1 0 ]
   // [ 0 0 0  0  0 1 ]
   cv::setIdentity(kfLeft.transitionMatrix);
   cv::setIdentity(kfRight.transitionMatrix);
   // Measure Matrix H
   // [ 1 0 0 0 0 0 ]
   // [ 0 1 0 0 0 0 ]
   // [ 0 0 0 0 1 0 ]
   // [ 0 0 0 0 0 1 ]
   kfLeft.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
   kfLeft.measurementMatrix.at<float>(0) = 1.0f;
   kfLeft.measurementMatrix.at<float>(7) = 1.0f;
   kfLeft.measurementMatrix.at<float>(16) = 1.0f;
   kfLeft.measurementMatrix.at<float>(23) = 1.0f;
	
   kfRight.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
   kfRight.measurementMatrix.at<float>(0) = 1.0f;
   kfRight.measurementMatrix.at<float>(7) = 1.0f;
   kfRight.measurementMatrix.at<float>(16) = 1.0f;
   kfRight.measurementMatrix.at<float>(23) = 1.0f;
 
   // Process Noise Covariance Matrix Q
   // [ Ex 0  0    0 0    0 ]
   // [ 0  Ey 0    0 0    0 ]
   // [ 0  0  Ev_x 0 0    0 ]
   // [ 0  0  0    1 Ev_y 0 ]
   // [ 0  0  0    0 1    Ew ]
   // [ 0  0  0    0 0    Eh ]
   //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
   kfLeft.processNoiseCov.at<float>(0) = 1e-2;
   kfLeft.processNoiseCov.at<float>(7) = 1e-2;
   kfLeft.processNoiseCov.at<float>(14) = 2.0f;
   kfLeft.processNoiseCov.at<float>(21) = 1.0f;
   kfLeft.processNoiseCov.at<float>(28) = 1e-2;
   kfLeft.processNoiseCov.at<float>(35) = 1e-2;
 
   kfRight.processNoiseCov.at<float>(0) = 1e-2;
   kfRight.processNoiseCov.at<float>(7) = 1e-2;
   kfRight.processNoiseCov.at<float>(14) = 2.0f;
   kfRight.processNoiseCov.at<float>(21) = 1.0f;
   kfRight.processNoiseCov.at<float>(28) = 1e-2;
   kfRight.processNoiseCov.at<float>(35) = 1e-2;

   // Measures Noise Covariance Matrix R
   cv::setIdentity(kfLeft.measurementNoiseCov, cv::Scalar(1e-1));
   cv::setIdentity(kfRight.measurementNoiseCov, cv::Scalar(1e-1));
   // <<<< Kalman Filter
}

void Kalman::runLeft(KalmanFilter& kf,Point& measuredPoint,Rect& boundingBox,cv::Mat& imageForTracking,Rect& returnedRect,Point& returedKalmanPoint){
	//imageForTracking.copyTo(imageLeft);

	double precTick = ticksLeft;
    ticksLeft = (double) cv::getTickCount();
 
    double dT = (ticksLeft - precTick) / cv::getTickFrequency(); //seconds

	if (firstTimeFlagLeft)
      {
		// >>>> Matrix A
         kf.transitionMatrix.at<float>(2) = dT;
         kf.transitionMatrix.at<float>(9) = dT;
         // <<<< Matrix A
 
         //cout << "dT:" << endl << dT << endl;
 
         stateLeft = kf.predict();
         //cout << "State post:" << endl << state << endl;
 
         cv::Rect predRect;
         predRect.width = stateLeft.at<float>(4);
		 predRect.height = stateLeft.at<float>(5);
         predRect.x = stateLeft.at<float>(0) - predRect.width / 2;
         predRect.y = stateLeft.at<float>(1) - predRect.height / 2;
 
         cv::Point center;
         center.x = stateLeft.at<float>(0);
         center.y = stateLeft.at<float>(1);
         
		 //return this kalman prediction center point 
		 returedKalmanPoint = center;

		//intialize returnedRect with predRect
		returnedRect = predRect;

		// Draw a line for both prediction and measurement points... 
		Point measuredPointLarge;
		measuredPointLarge.x = measuredPoint.x*4;
		measuredPointLarge.y = measuredPoint.y*4;
		measurementLeft.push_back(measuredPointLarge);
		Point centerLarge;
		centerLarge.x = center.x*4;
		centerLarge.y = center.y*4;
		kalmanvLeft.push_back(centerLarge); 
			
		for (int i = 0; i < measurementLeft.size()-1; i++) 
			line(imageForTracking, measurementLeft[i], measurementLeft[i+1], COLOR_YELLOW_AS_MEASURMENT, 3);
 
		for (int i = 0; i < kalmanvLeft.size()-1; i++) 
		line(imageForTracking, kalmanvLeft[i], kalmanvLeft[i+1], COLOR_BLACK_AS_PREDICTION, 3);
 
		if(measurementLeft.size() >= 20){
			measurementLeft.clear();
		}
		
		if(kalmanvLeft.size() >= 20){
			kalmanvLeft.clear();
		}
	}
		 
	 // >>>>> Kalman Update

      if (measuredPoint.x == 0 && measuredPoint.y == 0)
      {
         notFoundCountLeft++;
         cout << "notFoundCount:" << notFoundCountLeft << endl;
         if( notFoundCountLeft >= 10 )
         {
            firstTimeFlagLeft = false;
         }
         else
            kf.statePost = stateLeft;
      }
      else
      {
         notFoundCountLeft = 0;
 
         measLeft.at<float>(0) = (float)measuredPoint.x ;
         measLeft.at<float>(1) = (float)measuredPoint.y ;
         measLeft.at<float>(2) = (float)boundingBox.width;
         measLeft.at<float>(3) = (float)boundingBox.height;
 
         if (!firstTimeFlagLeft) // First detection!
         {
            // >>>> Initialization
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px
 
            stateLeft.at<float>(0) = measLeft.at<float>(0);
            stateLeft.at<float>(1) = measLeft.at<float>(1);
            stateLeft.at<float>(2) = 0;
            stateLeft.at<float>(3) = 0;
            stateLeft.at<float>(4) = measLeft.at<float>(2);
            stateLeft.at<float>(5) = measLeft.at<float>(3);
            // <<<< Initialization
 
            firstTimeFlagLeft = true;
         }
         else
            kf.correct(measLeft); // Kalman Correction
 
         //cout << "Measure matrix:" << endl << meas << endl;
      }
      // <<<<< Kalman Update 
}

void Kalman::runRight(KalmanFilter& kf,Point& measuredPoint,Rect& boundingBox,cv::Mat& imageForTracking,Rect& returnedRect,Point& returedKalmanPoint){
	//imageForTracking.copyTo(imageRight);

	double precTick = ticksRight;
    ticksRight = (double) cv::getTickCount();
 
    double dT = (ticksRight - precTick) / cv::getTickFrequency(); //seconds

	if (firstTimeFlagRight)
      {
		// >>>> Matrix A
         kf.transitionMatrix.at<float>(2) = dT;
         kf.transitionMatrix.at<float>(9) = dT;
         // <<<< Matrix A
 
         //cout << "dT:" << endl << dT << endl;
 
         stateRight = kf.predict();
         //cout << "State post:" << endl << state << endl;
 
         cv::Rect predRect;
         predRect.width = stateRight.at<float>(4);
         predRect.height = stateRight.at<float>(5);
         predRect.x = stateRight.at<float>(0) - predRect.width / 2;
         predRect.y = stateRight.at<float>(1) - predRect.height / 2;
 
         cv::Point center;
         center.x = stateRight.at<float>(0);
         center.y = stateRight.at<float>(1);
         
		 //return this kalman prediction center point 
		 returedKalmanPoint = center;
	
		//intialize returnedRect with predRect
		returnedRect = predRect;

		// Draw a line for both prediction and measurement points... 
		Point measuredPointLarge;
		measuredPointLarge.x = measuredPoint.x*4;
		measuredPointLarge.y = measuredPoint.y*4;
		measurementRight.push_back(measuredPointLarge);
		Point centerLarge;
		centerLarge.x = center.x*4;
		centerLarge.y = center.y*4;
		kalmanvRight.push_back(centerLarge);
		/*drawCross( center, COLOR_BLACK_AS_PREDICTION, 5 );
		drawCross( measuredPoint, COLOR_YELLOW_AS_MEASURMENT, 5 );*/

		for (int i = 0; i < measurementRight.size()-1; i++) 
			line(imageForTracking, measurementRight[i], measurementRight[i+1], COLOR_YELLOW_AS_MEASURMENT, 3);
     
			for (int i = 0; i < kalmanvRight.size()-1; i++) 
			line(imageForTracking, kalmanvRight[i], kalmanvRight[i+1], COLOR_BLACK_AS_PREDICTION, 3);
     
			if(measurementRight.size() >= 10){
				measurementRight.clear();
			}
			
			if(kalmanvRight.size() >= 10){
				kalmanvRight.clear();
			}
	}
		 
	 // >>>>> Kalman Update

      if (measuredPoint.x == 0 && measuredPoint.y == 0)
      {
         notFoundCountRight++;
         cout << "notFoundCount Right:" << notFoundCountRight << endl;
         if( notFoundCountRight >= 10 )
         {
            firstTimeFlagRight = false;
         }
         else
            kf.statePost = stateRight;
      }
      else
      {
         notFoundCountRight = 0;
 
         measRight.at<float>(0) = (float)measuredPoint.x ;
         measRight.at<float>(1) = (float)measuredPoint.y ;
         measRight.at<float>(2) = (float)boundingBox.width;
         measRight.at<float>(3) = (float)boundingBox.height;
 
         if (!firstTimeFlagRight) // First detection!
         {
            // >>>> Initialization
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px
 
            stateRight.at<float>(0) = measRight.at<float>(0);
            stateRight.at<float>(1) = measRight.at<float>(1);
            stateRight.at<float>(2) = 0;
            stateRight.at<float>(3) = 0;
            stateRight.at<float>(4) = measRight.at<float>(2);
            stateRight.at<float>(5) = measRight.at<float>(3);
            // <<<< Initialization
 
            firstTimeFlagRight = true;
         }
         else
            kf.correct(measRight); // Kalman Correction
 
         //cout << "Measure matrix:" << endl << meas << endl;
      }
      // <<<<< Kalman Update
	  
} 

Kalman::Kalman(){
	firstTimeFlagRight = false;
	firstTimeFlagLeft = false;  
}
