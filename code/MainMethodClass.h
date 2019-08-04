#ifndef _MAIN_METHOD_CLASS_
#define _MAIN_METHOD_CLASS_

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>
 
#include <cmath>
#include "myImage.hpp"
 
#include "HandDetection.hpp"
 
#include "Kalman.hpp"
#include "GrowRegion.hpp"
#include "OpenCVKinect.h"
#include "GestureRecognizerModified.h";
using namespace cv;
using namespace std;

// colors
const Scalar COLOR_BLUE        = Scalar(240,40,0);
const Scalar COLOR_DARK_GREEN  = Scalar(0, 128, 0);
const Scalar COLOR_LIGHT_GREEN = Scalar(0,255,0);
const Scalar COLOR_YELLOW      = Scalar(0,128,200);
const Scalar COLOR_RED         = Scalar(0,0,255);

 
// grasping threshold
const double GRASPING_THRESH = 0.90;
class MainMethodClass{

public:
	void run(cv::Mat& colorImageOrgSize,cv::Mat& depthImageOrgSize,OpenCVKinect& openCVKinect,int frameRate,
		Point3i& leftHandCoordinates,Point3i& rightHandCoordinates);
	MainMethodClass(KalmanFilter& kfLeft,KalmanFilter& kfRight);
	~MainMethodClass();
	void createTrackbars();
	
private:
	void onMouse( int event, int x, int y, int flags );
	static void onMouseStatic( int event, int x, int y, int flags, void* that );
	clock_t intemediate, intemediate1,intemediate2,initialRG,finalRG;

	// hand detection methods
	// conversion from cvConvexityDefect
	struct ConvexityDefect
	{
		Point start;
		Point end;
		Point depth_point;
		float depth;
	};
	void findHand(cv::Mat& backprojThreshold, Point& returnPoint,bool& isHand,String flag); 
	void eleminateDefects(vector<ConvexityDefect>& convexDefects);
	float distanceP2P(Point a, Point b);
	float getAngle(Point s, Point f, Point e);
	void removeRedundantEndPoints(vector<ConvexityDefect>& newDefects,vector<Point>& contour);
	void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<ConvexityDefect>& convexDefects);
	int findBiggestContour(vector<vector<Point> > contours);
	/// Not usefull methods right now...... But are in above list of hand detection methods
	void myDrawContours(Mat& srcMat,HandDetection& hg);
	void checkHandInImage(cv::Mat srcMat, bool& isHand,HandDetection& hg);
	void makeContours(cv::Mat& srcMat, HandDetection& hg);
	
	//////////////////// VARIABLES DECLARATION RELATED TO HAND DETECTION .................../////////////////////

	// hand detection variabels ////////////////////////////
	int bRect_height;
	int bRect_width;
	int nrOfDefects;
	
	//Report
	int counter;
	string str;

	 Mat reportMask1;

	Rect bRect;

	//////////////////// HAND DETECTION SECTION ENDS.........////////////////

	/* method to track left hand */
	void trackLeftHand(Point3i& handCoordinates);
	/* method to track right hand */
	void trackRightHand(Point3i& handCoordinates);
	
	void calculateBackProjForRight();
	void calculateBackProjForLeft( );
	
	void preProcessingBeforeCamshiftRight(Rect& trackWindow);
	void preProcessingBeforeCamshiftLeft(Rect& trackWindow);
	
	bool findMeadianOftheTrakWindowPixelsInDepthImageLeft(cv::Mat& depthImage,short& distance);
	bool findMeadianOftheTrakWindowPixelsInDepthImageRight(cv::Mat& depthImage,short& distance);
	
	void findMeadianOftheTrakWindowPixelsInDepthImage(IplImage* depthIplImage,Rect& trackWindow,short& distance);
	bool isCvMatEmpty(IplImage* srcMat);
	bool isCvMatEmpty(Mat& srcMat);
	bool isCvMatEmpty(Mat& srcMat,Rect& limitWindow);
	void displayHistogram(Mat& srcHistMat,String hand);
	void HistThresholding(cv::Mat& srcHistMat,int& minRange1, int& maxRange1,int& minRange2, int& maxRange2);
	void findFirstTwoBiggestPeakInHist(Mat& srcHistMat, float& index1, float& index2 );
	void thresholdIplImageBasedOnDepthValue(IplImage* srcMat,int depth,IplImage* tempImage);
	void thresholdCvMatBasedOnDepthValue(Mat& srcMat,Mat& tempImage);
	void expandTrackWindow(Rect& inputTrackWindow, Rect& returnTrackWindow, int margin,int MaxColVal,int MaxRowVal);
	void initialFiltering(cv::Mat& image);
	bool createDepthMaskBeforeCamShiftLeft();
	bool createDepthMaskBeforeCamShiftRight();
	void findXYCoordinatesFromCountors(vector<Point>& contour,vector<Point>& returnedContour);
	void checkDepthRegionContinuity(Point& seedPoint,int direction,IplImage* depthIplImage);
	void findContinuity(vector<Point>& srcVecNPs,int direction,String flag,IplImage* depthIplImage);
	void getNeighbourhoodPoints(Point& seedPoint,int direction,vector<Point>& returnedNPs);
	void findTwoPointsFromContours(vector<vector<Point>>& smallCountersPoints,vector<Point>& returnedPoints);
	void findEdgeDepthPixelWithMeadianDepthRange(IplImage* edgeDepthImage,vector<Point>& srcVec);
	void playWIthShowLeft(IplImage* edgeDepthImage,IplImage* depthIplImage,vector<Point>& returnedPoints,Point& centerPoint,Mat& depthShow);
	void playWIthShowRight(IplImage* edgeDepthImage,IplImage* depthIplImage,vector<Point>& returnedPoints,Point& centerPoint,Mat& depthShow);
	 
	void growEdgeImageUntillNextPoint(IplImage* edgeDepthImage,vector<Point>& srcVec);
	void sortVectorPointsLeft(vector<Point>& srcVecPoints,vector<Point>& sortedPoints,Point startPoint,Point stopPoint,String& flag,cv::Mat& show);
	void sortVectorPointsRight(vector<Point>& srcVecPoints,vector<Point>& sortedPoints,Point startPoint,Point stopPoint,String& flag,cv::Mat& show);
	void findTheTopPointOfThePalm(vector<Point>& sortedPoints,Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,String flag);
	void drawRectOnShowLeft(Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,Mat& show,String& flag);
	void drawRectOnShowRight(Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,Mat& show,String& flag);
	void initializeVariables(cv::Mat& colorImage,cv::Mat& depthImage);
	void pyrDownManually(IplImage* srcMat,IplImage* returnedIplImage);
	void removeBackGroundBasedOnDepthAndCreateEdgeImage(IplImage* srcMat,int depth,IplImage* depthBackGrndSubMask,IplImage* edgeDepthImage);
	void findcorrospondindPointsInDepthImageLeft(vector<Point>& returnedPoints,Point& centerPoint);
	void findcorrospondindPointsInDepthImageRight(vector<Point>& returnedPoints,Point& centerPoint);
	void skinThresholding();
	void checkPointInRange(Point& point,int maxColsRange,int maxRowSize);
	void initializeTrackingWindowLeft();
	void initializeTrackingWindowRight();
	void manualThreshold(Mat& srcMat, int thresholdVal,Mat& dstMat);
	Mat deflicker(Mat Mat1,int strengthcutoff );
	void MainMethodClass::deflicker(IplImage* Mat1,int strengthcutoff);
	void understandGesture(vector<Point3i>& originalTrajectory,vector<Point3i>& srcVector5x5Scattered);
	void copyIplImageToOtherIplImage(IplImage* srcImage,IplImage* destImage);
	void writeMatToFile(Mat& srcMat,String filename,String MatName);

	/// Not usefull methods right now...... But are in above list of Color + Depth Hand tracking/
	void divideDepthImage(Mat& srcMat,vector<Mat>& depthdataSubImages);
	void manualThreshold(Mat& srcMat, int thresholdVal,Mat& returnNearMask,Mat& returnFarMask);
	void averageDepthImageValue(Mat srcMat,Rect roi, float& avgDepthValue);
	void mainWaterShade(Mat& srcMat,Mat& srcColorMat);
	void processTheImage(cv::Mat& backprojThreshold);
	int countNoOfPixelsInTrackWindow(cv::Mat& srcMat);
	short depthDistanceLeft;
	short prevDepthDistanceLeft;
	short depthDistanceRight;
	
	//////////////////// VARIABLES DECLARATION RELATED TO HAND TRACKING .................../////////////////////
	Rect selectionRight;
	Rect selectionLeft;
	Rect selection;
	Rect trackWindowHead;
	Rect trackWindowLeft;
	Rect trackWindowRight;
	Rect trackWindowFromDepthLeft; //This is depth made rect from toppoint and two other points. very very important
	Rect trackWindowFromDepthRight; //This is depth made rect from toppoint and two other points. very very important
	Rect kalmanReturnedWindowLeft;		// use this kalman returned window when trackWindow == null..
	Rect kalmanReturnedWindowRight;		// use this kalman returned window when trackWindow == null..
	RotatedRect trackBox;
	RotatedRect trackBoxRight;
	RotatedRect trackBoxHead;
	

	String flagForDirectionLeft;
	String flagForDirectionRight;
	String gestureText;				//this would be the global text for a gesture, it will indicate if any gesture is detected?

	bool gestureFlag;				//this flag will be global flag indicating if any gesture has been previously detected or not?
	bool firstTimeDirectionFlag;

	bool firstTimeFlagForGlobalDepthMedianDistance ; //very very imp
	bool selectObject;
	bool showHist ;
	bool flagForFirstTime;		//this flag is for cuurent depth frame and previous depth frame and operation 
	bool flagForGesture;		//this flag is used for gesture start stop indication..

	int countForStartStopGesture;	//this is the counter which will indicate start stop after certain value.
	int trackObjectLeft ;
	int trackObjectRight ;
	int vmin , vmax , smin ;
	int dynamicHueMin1 ;
	int dynamicHueMax1 ;
	int dynamicHueMin2 ;
	int dynamicHueMax2 ;

	//global hue min & max ranges for initial filtering
	int staticHueMin1 ;
	int staticHueMax1 ;
	int staticHueMin2 ;
	int staticHueMax2 ;
	int satMin ;
	int satMax ;
	int valMin ;
	int valMax ;
	int Y_MIN;
	int Y_MAX;
	int Cr_MIN;
	int Cr_MAX;
	int Cb_MIN;
	int Cb_MAX;
	int medianBlurCnt ;
	int erodeAmt ;
	int dilateAmt;
	int backGroundSubtractionFlag ;
	int flagForImage ;
	int flagForKalman;
	int writeToFileFlag ;
	int flagForWaterShade;
	int globalCounter ;
	int displayHistogramFlag;
	int handDetectionFlag;
	int MINNOOFPIXELS ;
	int hsize ;
	int gesture;
	int flickering;
	int ORGCOLS;
	int ORGROWS;

	int PYRDOWNCOLS;
	int PYRDOWNROWS;
	
	int indexOfGestures;
	int counterForLeftTWNull;		//this is counter for no of frames the left TW is null, if its more than certain value reinitialize the left TW..
	int counterForRightTWNull;		//this is counter for no of frames the Right TW is null, if its more than certain value reinitialize the Right TW..
	int findFingersFlag;			//this is the flag for finger detection algorithm.

	int dynamicThresholding ;

	float widthHeightRationFactor;	//this is very imp for creating new Rect from width or height info..
	
	vector<Point3i> trajectory50Points;  
	vector<Point3i> trajectory;  
	vector<Point3i> individualTrajectory;		//this  is set of points which are at specific distance to each other.
	vector<Point3i> originalTrajectory;			//this  is set of points with original values
	
	cv::Mat mask1,mask1PreviousFrame,mask2,mask3,mask4,mask5;
	Mat newBackGrngThrImage;
	Mat hsv, hue,sat,val, mask, histLeft,histRight, histimg, backprojLeft,backprojRight,YCrCbMat;
	Mat trackWindowImage;
	cv::Mat processBackProjThrMat;
	Mat mask1Right;
	Mat mask1Left;
	Mat depthMatOrgSizePreviousFrame;		//global frames, previous depth frames
	Mat prevdeflicker;
	

	Mat depthMatPyrDown;		//global frames
	Mat colorMatPyrDown;		//global frames
	Mat colorMatOriginalSize;	//global frames
	Mat depthMatOriginalSize;	//global frames
	Mat show;					//global frames, for depht image show with CV_8U 
	Mat depthImageForHandFingers;	//global frames, for depht image show with CV_8U, used for hand finger numbers
	Mat returnedBorderMatRight;		//this will be the return mat for edge grow method
	Mat returnedBorderMatLeft;		//this will be the return mat for edge grow method

	IplImage* prevdeflickerIplImage;
	IplImage* tempImageLeft;				//global frames, imp for depth median mask 
	IplImage* tempImageRight;				//global frames, imp for depth median mask 
	IplImage* edgeMaskImageLeft;				//global frames, imp for edge depth median mask 
	IplImage* edgeMaskImageRight;				//global frames, imp for edge depth median mask 

	IplImage* new_image1;
	IplImage* depthIplImageOriginalSize;	//global frames
	IplImage* depthIplImagePyrDown;			//global frames
	IplImage* edgeIplDepthImage;			//global frames
	IplImage* depthIplBackGrndSubMask;		//global frames
	IplImage* tempImage;					//global frames
	IplImage* tempImageEdgeImage;			//global frames
	Point topPointPalm,leftPointPalm,rightPointPalm; //very imp..

	Point centerPointLeft;					//this point will the center point of left hand..
	Point centerPointRight;					//this point will the center point of right hand..

	Point returnedKalmanPointLeft;			//this point will the retured point of kalman, use it in case of trackWindow == null..
	Point returnedKalmanPointRight;			//this point will the retured point of kalman, use it in case of trackWindow == null..

	Point3i prevTrajectoryPoint;			//used for gesture reco.
	//mouse points origin
	Point origin;
	// kalman stuff
	cv::Point center;
	Kalman kalman;
	KalmanFilter kfLeft;
	KalmanFilter kfRight;
	//KalmanOld kal;
	  
	
	//GestureRecognizer gestureRecognizer; 
	GestureRecognizerModified gestureRecognizerModified; 
	//////////////////////////////////////   HAND TRACKING SECTION END ******* ///////////////////
	 
};
#endif