#ifndef _GROWREGION_
#define _GROWREGION_
 
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv\cv.h>
 

#include <iostream>
#include <ctype.h>

using namespace std;
using namespace cv;
  



class GrowRegion{
	int globalCounter;
	 
	int ROWS;
	int COLS;

 
	Rect rect; 
	Mat depthImage;

	Mat cloneLookUpDepthImage;
	IplImage* cloneLookUpDepthImageDifferently;
	IplImage* borderImage;
	vector<Point> countor;
	 //IplImage* tempImage;

	Mat onesMat;
	
	int rowsVal;
	int colsVal;
	void getNeighbourhoodPixels(Point SeedPoint, vector<Point>& neighbourPixels);
	/*void getNewSeedPointsDifferently(vector<Point> SeedPoints);*/
	int findBiggestContour(CvSeq* contours);
	void DrawTextOnIPLImage(IplImage* srcImage,Point& position,String text);
	 
	void getNewSeedPoints(vector<Point> SeedPoints);
	CvMemStorage* storage;

	int XMIN, XMAX, YMIN, YMAX;
	 

	void thresholdIplImageBasedOnDepthValue(IplImage* srcMat,unsigned short depth);
	clock_t intemediate, intemediate1,intemediate2,initialRG,finalRG;

	Point stopPoint; //very imp
	Point startPoint; //very imp  
	IplImage* srcMat; //very imp

public:

	/*void run(Mat& srcMat, Point SeedPoint,Rect& trackWindowForDepthImage);*/
	void run(IplImage* srcMatInput, Point SeedPointInput,int distance,Point& stopPointInput,vector<Point>& borderPoints,Mat& returnedBorderMat);
	//constructor
	~GrowRegion();
	GrowRegion(IplImage* srcImage);
};

#endif