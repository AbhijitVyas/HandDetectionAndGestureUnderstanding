#ifndef _GESTURE_RECOGNIZED_MODIFIED_
#define _GESTURE_RECOGNIZED_MODIFIED_
  
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>
 #include <opencv2/ml/ml.hpp>

#include <iostream>     // std::cout
#include <functional>   // std::multiplies
#include <numeric>      // std::adjacent_difference
#include <math.h>       /* acos */
#include <fstream>
#include <iterator>
#include <map>
#include <algorithm>    // std::adjacent_find
#include <vector>       // std::vector

using namespace cv;
using namespace std;
 
#define PI 3.14159265
const float RANGEM = 5;
const int noOfTrainingSamplesCircleM = 95;
const int noOfPVeTrainingSamplesCircleM = 29;
  
const int noOfTrainingSamplesLetterSM = 37;
const int noOfPVeTrainingSamplesLetterSM = 16;

const int noOfTrainingSamplesDigit1M = 83;
const int noOfPVeTrainingSamplesDigit1M = 14;

const int noOfTrainingSamplesDigit2M = 61;
const int noOfPVeTrainingSamplesDigit2M = 10;

const int noOfTrainingSamplesDigit3M = 71;
const int noOfPVeTrainingSamplesDigit3M = 20;

const int noOfTrainingSamplesDigit4M = 84;
const int noOfPVeTrainingSamplesDigit4M = 10;

const int noOfTrainingSamplesDigit5M = 61;
const int noOfPVeTrainingSamplesDigit5M = 9;

const int noOfTrainingSamplesDigit6M = 67;
const int noOfPVeTrainingSamplesDigit6M = 35;

const int noOfTrainingSamplesDigit7M = 118;
const int noOfPVeTrainingSamplesDigit7M = 66;

const int noOfTrainingSamplesDigit8M = 134;
const int noOfPVeTrainingSamplesDigit8M = 9;

const int noOfTrainingSamplesDigit9M = 87;
const int noOfPVeTrainingSamplesDigit9M = 13;




class GestureRecognizerModified
{ 
	 
	bool gestureStartStopFlag;

	int findMedianOfVector(vector<int>& srcVector);
	int findAdjacentDifference(vector<Point3i>& srcVector);
	void analyzeGesturePoints(vector<Point3i>& srcGesturePoints);		//this method helps to find extream Left,Right,Top and bottom points of gesture trajectory.
	void findRepetedPatternsInASrcVec(vector<int>& srcVec,int& returnFirst,int& returnSecond);				//this method finds similar pattern in a angle chain vector
	void findRepetedPatternsInASrcVecLeft(vector<int>& srcVec,vector<String>& returnPattern,int limit);
	void findRepetedPatternsInASrcVecRight(vector<int>& srcVec,vector<String>& returnPattern,int limit);
	void findRepetedPatternsInASrcVecUp(vector<int>& srcVec,vector<String>& returnPattern,int limit);
	void findRepetedPatternsInASrcVecDown(vector<int>& srcVec,vector<String>& returnPattern,int limit);
	//bool myfunction (int i, int j);
	//gesture rules variables..
	int widthOriginal;		//width of Original gesture ..
	int heightOriginal;		//height of Original gesture..
	int indexOfGesturesGlobal;
	int closenessCriteria;	//it checks if the start point and stop point are close enough to say its a close gesture or not?
							//sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20

	float rationForGestureReduced;	//this ratio will be assigned when any one gesture is reduced for understanding purpose
	
	Rect boundingRectAroundGestureOriginal;		//this would be the bounding rect around original size gesture.

	Point3i startPointOriginal;		//start point of Original gesture..
	Point3i stopPointOriginal;		//start point of Original gesture..
	Point3i extreamLeftOriginal;	//extreamLeft point of Original gesture..
	Point3i extreamRightOriginal;	//extreamRight point of Original gesture..
	Point3i extreamTopOriginal;		//extreamTop point of Original gesture..
	Point3i extreamBottomOriginal;	//extreamBottom point of Original gesture..

	Point3i startPointScaledDownVec;	//start point of ScaledDownVec gesture..
	Point3i stopPointScaledDownVec;		//start point of ScaledDownVec gesture..
	Point3i extreamLeftScaledDownVec;	//extreamLeft point of ScaledDownVec gesture..
	Point3i extreamRightScaledDownVec;	//extreamRight point of ScaledDownVec gesture..
	Point3i extreamTopScaledDownVec;	//extreamTop point of ScaledDownVec gesture..
	Point3i extreamBottomScaledDownVec;	//extreamBottom point of ScaledDownVec gesture..

	vector<int> angleVectorOriginal;	//this would be the angle vec(sequence of codes) for Original trajectory
	
	
	vector<int> angleVecForDisplayPurpose;	//this vector will help us to visualize angles
	vector<Point> pointsVecForDisplayPurpose;	//this vector will help us to visualize Points
	vector<Point3i> points3iVecForDisplayPurpose;	//this vector will help us to visualize Points

	vector<Point> approx;	//this vector hold values of an angle with Approx curve..

	vector<Mat> angleMatVec;			//this would be the vector of angle Mats(sequence of codes), utilized mainly for writing 10 gestures to a text file.			
	
	vector<String> gestureTextVec;		//used for wrting gesture data to file.
	Mat trainingDataMatCircle,trainingDataMatSquare,trainingDataMatTriangle,trainingDataMatLetterS,
		trainingDataMatDigit1,trainingDataMatDigit2,trainingDataMatDigit3,trainingDataMatDigit4,trainingDataMatDigit5,
		trainingDataMatDigit6,trainingDataMatDigit7,trainingDataMatDigit8,trainingDataMatDigit9;
	
	//vector<Mat> trainingDataTotal;


	//SVM initialization
	CvSVMParams paramsCircle;
	CvSVM SVMCircle;

	CvSVMParams paramsSquare;
	CvSVM SVMSquare;

	CvSVMParams paramsTriangle;
	CvSVM SVMTriangle;

	CvSVMParams paramsLetterS;
	CvSVM SVMLetterS;

	CvSVMParams paramsDigit1,paramsDigit2,paramsDigit3,paramsDigit4,paramsDigit5,paramsDigit6,paramsDigit7,paramsDigit8,paramsDigit9;
	CvSVM SVMDigit1,SVMDigit2,SVMDigit3,SVMDigit4,SVMDigit5,SVMDigit6,SVMDigit7,SVMDigit8,SVMDigit9;

	
	void findAngleVector(vector<Point3i>& srcVector,vector<int>& returnedAngleVector);
	void findAngleVector(vector<Point>& srcVector,vector<int>& returnedAngleVector);
	void removeDuplicateElements(vector<int>& srcVector,vector<int>& returnVector);
	void convertAngleCodeVectorToQuarterVec(vector<int>& returnReducedVector,vector<int>& quarterVec);
	double findAngleBetweenTwoPoints(Point3i& a,Point3i& b);
	double findAngleBetweenTwoPoints(Point& A,Point& B);
	int findNoOfStreightSidesInVector(vector<int>& srcVec);
	//method for finding streight line
	void findStreightLine();
	void trainingDataForCircle();
	void triangleTrainingData();
	void LetterSTrainingData();
	void Digit1TrainingData();
	void Digit2TrainingData();
	void Digit3TrainingData();
	void Digit4TrainingData();
	void Digit5TrainingData();
	void Digit6TrainingData();
	void Digit7TrainingData();
	void Digit8TrainingData();
	void Digit9TrainingData();

	void trainSVMCircle();		//this can work as circle, ellipse and or zero SVM
	void trainSVMDigit1();
	void trainSVMDigit2();
	void trainSVMDigit3();
	void trainSVMDigit4();
	void trainSVMDigit5();
	void trainSVMDigit6();
	void trainSVMDigit7();
	void trainSVMDigit8();
	void trainSVMDigit9();

	void trainSVMLetterS();


	void writeCircleAngleStreamInFile(vector<Mat>& angleMatVecSrc,vector<String>& gestureTextVecSrc); 
	
	void circleTrainingData();
	void squareTrainingData();
	bool detectSquareWithSimpleTechniqueModified(vector<int>& srcVector);
	void copyArrayDataToVec();
	struct by_second
{
    template <typename Pair>
    bool operator()(const Pair& a, const Pair& b)
    {
        return a.second < b.second;
    }
};
	template <typename Fwd> 
	typename std::map<typename std::iterator_traits<Fwd>::value_type, int>::value_type most_frequent_element(Fwd begin, Fwd end);
	bool detectUp(vector<Point3i>& srcVector,String& returnString);
	bool detectUp(vector<Point3i>& srcVector);
	bool detectDown(vector<Point3i>& srcVector,String& returnString);
	bool detectDown(vector<Point3i>& srcVector);
	bool detectLeft(vector<Point3i>& srcVector,String& returnString);
	bool detectLeft(vector<Point3i>& srcVector);
	bool detectRight(vector<Point3i>& srcVector,String& returnString);
	bool detectRight(vector<Point3i>& srcVector);
	bool detectCircle(vector<int>& returnReducedVector,String& returnString);
	bool detectEllipse(vector<int>& returnReducedVector,String& returnString);
	bool detectCharacterS(vector<int>& returnReducedVector,String& returnString);
	bool detectLater5(vector<int>& returnReducedVector,String& returnString);
	
	void drawTrajectoryOnMat(vector<Point3i>& srcVecPoints,vector<int>& angleVec,Mat& drawMat,double ratio);
	void scaledDownSrcVec(vector<Point3i>& srcVectorOriginalSize,vector<Point3i>& scaledDownVecReturn,
		vector<int>& angleVecSDReturn,Mat& returnSDMat,Mat& srcMatForVisualization);
	//svm methods
	bool findCircle(vector<Point3i>& srcVectorOriginalSize, String& returnString,Mat& srcMat,int index);
	bool findCircle(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit1(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit2(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit3(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit4(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit5(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit6(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit7(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit8(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findDigit9(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);
	bool findLetterS(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index);

	
public: 
	GestureRecognizerModified();
	~GestureRecognizerModified();
	
	
	bool ifVectorValuesAreInRange(vector<Point3i>& srcVector); 
	bool understandTheGesture(vector<Point3i>& srcVectorOriginal,vector<Point3i>& srcVector5x5Scattered,Mat& srcMatForShapes,
		Rect& returnedGestureBoundingRect,int indexOfGestures,String& returnGestureText);
	
	String understandTheGestureWithShapeDescriptor(vector<Point3i>& srcVector);
	bool checkForGesture(vector<Point3i>& srcVector);
	
	void drawTrajectoryOnMat(Mat& drawMat);
	void writeCircleAngleStreamInFile(vector<int>& returnReducedVector,int& indexOfGestures,String gestureText);
	void storeVectors(float xW,float yW, float zW);
	void drawTrajectoryBasedOnVecPoints(vector<Point3i>& originalTrajectory,int rows,int cols,String ImageName,Rect boundingRect);
	void findEdgePointsForTrajectory(vector<Point3i>& srcVector,Point3i& XL,Point3i& XH,Point3i& YL,Point3i& YH,Rect& returnRectAroundGesture
		,int& gestureWidth,int& gestureHeight);
	
	
	 
	
};



#endif