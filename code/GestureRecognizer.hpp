#pragma once
  
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
const float RANGE = 20;
const int noOfTrainingSamplesCircle = 198;
const int noOfPVeTrainingSamplesCircle = 153;
 
const int noOfTrainingSamplesSquare = 247;
const int noOfPVeTrainingSamplesSquare = 48;
 
const int noOfTrainingSamplesTriangle = 174;
const int noOfPVeTrainingSamplesTriangle = 20;

const int noOfTrainingSamplesLetterS = 121;
const int noOfPVeTrainingSamplesLetterS = 100;

const int noOfTrainingSamplesDigit1 = 149;
const int noOfPVeTrainingSamplesDigit1 = 80;

const int noOfTrainingSamplesDigit2 = 171;
const int noOfPVeTrainingSamplesDigit2 = 66;

const int noOfTrainingSamplesDigit3 = 125;
const int noOfPVeTrainingSamplesDigit3 = 74;

const int noOfTrainingSamplesDigit4 = 143;
const int noOfPVeTrainingSamplesDigit4 = 69;

const int noOfTrainingSamplesDigit5 = 118;
const int noOfPVeTrainingSamplesDigit5 = 66;

const int noOfTrainingSamplesDigit6 = 145;
const int noOfPVeTrainingSamplesDigit6 = 52;

const int noOfTrainingSamplesDigit7 = 118;
const int noOfPVeTrainingSamplesDigit7 = 66;

const int noOfTrainingSamplesDigit8 = 186;
const int noOfPVeTrainingSamplesDigit8 = 41;

const int noOfTrainingSamplesDigit9 = 121;
const int noOfPVeTrainingSamplesDigit9 = 47;




class GestureRecognizer
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
	int width;		//width of gesture..
	int height;		//height of gesture..
	int indexOfGesturesGlobal;
	int closenessCriteria;	//it checks if the start point and stop point are close enough to say its a close gesture or not?
							//sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20

	Point3i startPoint;	//start point of gesture..
	Point3i stopPoint;	//start point of gesture..
	Point3i extreamLeft;	//extreamLeft point of gesture..
	Point3i extreamRight;	//extreamRight point of gesture..
	Point3i extreamTop;		//extreamTop point of gesture..
	Point3i extreamBottom;	//extreamBottom point of gesture..

	vector<int> angleVector;

	vector<Mat> circleAngleVec;
	
	vector<String> gestureTextVec;
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

	
	void findAngleVector(vector<Point3i>& srcVector);
	void removeDuplicateElements(vector<int>& srcVector,vector<int>& returnVector);
	void convertAngleCodeVectorToQuarterVec(vector<int>& returnReducedVector,vector<int>& quarterVec);
	double findAngleBetweenTwoPoints(Point3i& a,Point3i& b);
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


	void writeCircleAngleStreamInFile(vector<Mat>& circleAngleVec,vector<String>& gestureTextVecSrc); 
	
	void circleTrainingData();
	void squareTrainingData();
	bool GestureRecognizer::detectSquareWithSimpleTechnique(vector<int>& srcVector);
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
public: 
	GestureRecognizer();
	~GestureRecognizer();
	String checkForGesture(vector<Point3i>& srcVector);
	bool ifVectorValuesAreInRange(vector<Point3i>& srcVector);
	void storeVectors(float xW,float yW, float zW);
	void findEdgePointsOnTrajectory(vector<Point3i>& srcVector,Point& XL,Point& XH,Point& YL,Point& YH,Rect& returnRectAroundGesture,
		vector<int>& angleReturnedVec,vector<int>& returnReducedVector);
	 
	String understandTheGesture(vector<Point3i>& srcVectorScattered,vector<int>& srcReducedVec,vector<Point3i>& srcVectorOriginal,Mat& srcMatForShapes);
	String understandTheGestureWithShapeDescriptor(vector<Point3i>& srcVector);
	void writeCircleAngleStreamInFile(vector<int>& returnReducedVector,int& indexOfGestures,String gestureText);
};