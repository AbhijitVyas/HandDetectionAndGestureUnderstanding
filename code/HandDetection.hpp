#ifndef _HAND_DETECTION_
#define _HAND_DETECTION_

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>
 
 

using namespace cv;
using namespace std;
#define PI 3.14159
class HandDetection{
	public:
 
		HandDetection();
		vector<vector<Point> > contours;
		vector<vector<int> >hullI;
		vector<vector<Point> >hullP;
		vector<vector<Vec4i> > defects;	
		vector <Point> fingerTips;
		Rect rect;
		void printGestureInfo(Mat src);
		int cIdx;
		int frameNumber;
		int mostFrequentFingerNumber;
		int nrOfDefects;
		Rect bRect;
		double bRect_width;
		double bRect_height;
		bool isHand;
		bool detectIfHand();
		void initVectors();
		void getFingerNumber(Mat& srcMat);
		void eleminateDefects();
		void getFingerTips(Mat& srcMat);
		void drawFingerTips(Mat& srcMat);
	private:
		string bool2string(bool tf);
		int fontFace;
		int prevNrFingerTips;
		void checkForOneFinger(Mat& srcMat);
		float getAngle(Point s,Point f,Point e);	
		vector<int> fingerNumbers;
		void analyzeContours();
		string intToString(int number);
		void computeFingerNumber();
		void drawNewNumber(Mat& srcMat);
		void addNumberToImg(Mat& srcMat);
		vector<int> numbers2Display;
		void addFingerNumberToVector();
		Scalar numberColor;
		int nrNoFinger;
		float distanceP2P(Point a,Point b);
		void removeRedundantEndPoints(vector<Vec4i> newDefects);
		void removeRedundantFingerTips();
};




#endif
