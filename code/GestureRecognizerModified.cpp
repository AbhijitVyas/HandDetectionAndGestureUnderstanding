#include "GestureRecognizerModified.h"

GestureRecognizerModified::GestureRecognizerModified(){
	//counterForLeft = 0;
	gestureStartStopFlag = false;
	indexOfGesturesGlobal = 0;
	  
	rationForGestureReduced = 0;	//this ratio will be assigned when any one gesture is reduced for understanding purpose

	//////////////////////////////////////////////////// Circle Taining Phase //////////////////////////////////////////////
	trainSVMCircle();

	//////////////////////////////////////////////////// Letter S Taining Phase //////////////////////////////////////////////
	trainSVMLetterS();
	 
	//////////////////////////////////////////////////// Digit 1 Taining Phase //////////////////////////////////////////////
	trainSVMDigit1();
	
	//////////////////////////////////////////////////// Digit 2 Taining Phase //////////////////////////////////////////////
	trainSVMDigit2();
	 
	//////////////////////////////////////////////////// Digit 3 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit3();

	//////////////////////////////////////////////////// Digit 4 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit4();

	//////////////////////////////////////////////////// Digit 5 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit5();

	//////////////////////////////////////////////////// Digit 6 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit6();

	//////////////////////////////////////////////////// Digit 7 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit7();

	//////////////////////////////////////////////////// Digit 8 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit8();

	//////////////////////////////////////////////////// Digit 9 Taining Phase //////////////////////////////////////////////
	 trainSVMDigit9();
 
}

GestureRecognizerModified::~GestureRecognizerModified(){
	 
}

void GestureRecognizerModified::trainSVMLetterS(){
	//////////////////////////////////////////////////// Letter S Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Letter S
	LetterSTrainingData();
	  
	 // Set up training data
	float labelsLetterS[noOfTrainingSamplesLetterSM];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesLetterSM ; i6++){
		labelsLetterS[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesLetterSM; i7 < noOfTrainingSamplesLetterSM ; i7++){
			labelsLetterS[i7] = -1.0;
	}

	 
    Mat labelsMatLetterS(1, noOfTrainingSamplesLetterSM, CV_32FC1, labelsLetterS);
	// Set up SVM's parameters
    paramsLetterS.svm_type    = CvSVM::C_SVC;
    paramsLetterS.kernel_type = CvSVM::LINEAR;
    paramsLetterS.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMLetterS.train(trainingDataMatLetterS, labelsMatLetterS, Mat(), Mat(), paramsLetterS);
}
void GestureRecognizerModified::trainSVMDigit1(){
	//////////////////////////////////////////////////// Digit 1 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 1
	Digit1TrainingData();
	  
	 // Set up training data
	float labelsDigit1[noOfTrainingSamplesDigit1M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit1M ; i6++){
		labelsDigit1[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit1M; i7 < noOfTrainingSamplesDigit1M; i7++){
			labelsDigit1[i7] = -1.0;
	}

	 
    Mat labelsMatDigit1(1, noOfTrainingSamplesDigit1M, CV_32FC1, labelsDigit1);
	// Set up SVM's parameters
    paramsDigit1.svm_type    = CvSVM::C_SVC;
    paramsDigit1.kernel_type = CvSVM::LINEAR;
    paramsDigit1.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit1.train(trainingDataMatDigit1, labelsMatDigit1, Mat(), Mat(), paramsDigit1);
}

void GestureRecognizerModified::trainSVMDigit2(){
//////////////////////////////////////////////////// Digit 2 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 2
	Digit2TrainingData();
	  
	 // Set up training data
	float labelsDigit2[noOfTrainingSamplesDigit2M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit2M ; i6++){
		labelsDigit2[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit2M; i7 < noOfTrainingSamplesDigit2M; i7++){
			labelsDigit2[i7] = -1.0;
	}

	 
    Mat labelsMatDigit2(1, noOfTrainingSamplesDigit2M, CV_32FC1, labelsDigit2);
	// Set up SVM's parameters
    paramsDigit2.svm_type    = CvSVM::C_SVC;
    paramsDigit2.kernel_type = CvSVM::LINEAR;
    paramsDigit2.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit2.train(trainingDataMatDigit2, labelsMatDigit2, Mat(), Mat(), paramsDigit2);
}

void GestureRecognizerModified::trainSVMDigit3(){
	//////////////////////////////////////////////////// Digit 3 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 3
	Digit3TrainingData();
	  
	 // Set up training data
	float labelsDigit3[noOfTrainingSamplesDigit3M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit3M ; i6++){
		labelsDigit3[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit3M; i7 < noOfTrainingSamplesDigit3M; i7++){
			labelsDigit3[i7] = -1.0;
	}

	 
    Mat labelsMatDigit3(1, noOfTrainingSamplesDigit3M, CV_32FC1, labelsDigit3);
	// Set up SVM's parameters
    paramsDigit3.svm_type    = CvSVM::C_SVC;
    paramsDigit3.kernel_type = CvSVM::LINEAR;
    paramsDigit3.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit3.train(trainingDataMatDigit3, labelsMatDigit3, Mat(), Mat(), paramsDigit3);
}

void GestureRecognizerModified::trainSVMDigit4(){
	//////////////////////////////////////////////////// Digit 4 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 4
	Digit4TrainingData();
	  
	 // Set up training data
	float labelsDigit4[noOfTrainingSamplesDigit4M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit4M ; i6++){
		labelsDigit4[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit4M; i7 < noOfTrainingSamplesDigit4M; i7++){
			labelsDigit4[i7] = -1.0;
	}

	 
    Mat labelsMatDigit4(1, noOfTrainingSamplesDigit4M, CV_32FC1, labelsDigit4);
	// Set up SVM's parameters
    paramsDigit4.svm_type    = CvSVM::C_SVC;
    paramsDigit4.kernel_type = CvSVM::LINEAR;
    paramsDigit4.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit4.train(trainingDataMatDigit4, labelsMatDigit4, Mat(), Mat(), paramsDigit4);
}

void GestureRecognizerModified::trainSVMDigit5(){
	//////////////////////////////////////////////////// Digit 5 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 5
	Digit5TrainingData();
	  
	 // Set up training data
	float labelsDigit5[noOfTrainingSamplesDigit5M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit5M ; i6++){
		labelsDigit5[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit5M; i7 < noOfTrainingSamplesDigit5M; i7++){
			labelsDigit5[i7] = -1.0;
	}

	 
    Mat labelsMatDigit5(1, noOfTrainingSamplesDigit5M, CV_32FC1, labelsDigit5);
	// Set up SVM's parameters
    paramsDigit5.svm_type    = CvSVM::C_SVC;
    paramsDigit5.kernel_type = CvSVM::LINEAR;
    paramsDigit5.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit5.train(trainingDataMatDigit5, labelsMatDigit5, Mat(), Mat(), paramsDigit5);
}

void GestureRecognizerModified::trainSVMDigit6(){
	//////////////////////////////////////////////////// Digit 6 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 6
	Digit6TrainingData();
	  
	 // Set up training data
	float labelsDigit6[noOfTrainingSamplesDigit6M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit6M ; i6++){
		labelsDigit6[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit6M; i7 < noOfTrainingSamplesDigit6M; i7++){
			labelsDigit6[i7] = -1.0;
	}

	 
    Mat labelsMatDigit6(1, noOfTrainingSamplesDigit6M, CV_32FC1, labelsDigit6);
	// Set up SVM's parameters
    paramsDigit6.svm_type    = CvSVM::C_SVC;
    paramsDigit6.kernel_type = CvSVM::LINEAR;
    paramsDigit6.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit6.train(trainingDataMatDigit6, labelsMatDigit6, Mat(), Mat(), paramsDigit6);
}

void GestureRecognizerModified::trainSVMDigit7(){
	//////////////////////////////////////////////////// Digit 7 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 7
	Digit7TrainingData();
	  
	 // Set up training data
	float labelsDigit7[noOfTrainingSamplesDigit7M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit7M ; i6++){
		labelsDigit7[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit7M; i7 < noOfTrainingSamplesDigit7M; i7++){
			labelsDigit7[i7] = -1.0;
	}

	 
    Mat labelsMatDigit7(1, noOfTrainingSamplesDigit7M, CV_32FC1, labelsDigit7);
	// Set up SVM's parameters
    paramsDigit7.svm_type    = CvSVM::C_SVC;
    paramsDigit7.kernel_type = CvSVM::LINEAR;
    paramsDigit7.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit7.train(trainingDataMatDigit7, labelsMatDigit7, Mat(), Mat(), paramsDigit7);
}

void GestureRecognizerModified::trainSVMDigit8(){
	//////////////////////////////////////////////////// Digit 8 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 8
	Digit8TrainingData();
	  
	 // Set up training data
	float labelsDigit8[noOfTrainingSamplesDigit8M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit8M ; i6++){
		labelsDigit8[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit8M; i7 < noOfTrainingSamplesDigit8M; i7++){
			labelsDigit8[i7] = -1.0;
	}

	 
    Mat labelsMatDigit8(1, noOfTrainingSamplesDigit8M, CV_32FC1, labelsDigit8);
	// Set up SVM's parameters
    paramsDigit8.svm_type    = CvSVM::C_SVC;
    paramsDigit8.kernel_type = CvSVM::LINEAR;
    paramsDigit8.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit8.train(trainingDataMatDigit8, labelsMatDigit8, Mat(), Mat(), paramsDigit8);
}

void GestureRecognizerModified::trainSVMDigit9(){
	//////////////////////////////////////////////////// Digit 9 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 9
	Digit9TrainingData();
	  
	 // Set up training data
	float labelsDigit9[noOfTrainingSamplesDigit9M];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit9M; i6++){
		labelsDigit9[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit9M; i7 < noOfTrainingSamplesDigit9M; i7++){
			labelsDigit9[i7] = -1.0;
	}

	 
    Mat labelsMatDigit9(1, noOfTrainingSamplesDigit9M, CV_32FC1, labelsDigit9);
	// Set up SVM's parameters
    paramsDigit9.svm_type    = CvSVM::C_SVC;
    paramsDigit9.kernel_type = CvSVM::LINEAR;
    paramsDigit9.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit9.train(trainingDataMatDigit9, labelsMatDigit9, Mat(), Mat(), paramsDigit9);
}

void GestureRecognizerModified::trainSVMCircle(){
	//////////////////////////////////////////////////// Circle Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Circe
	circleTrainingData();
	  
	 // Set up training data
	float labelsCircle[noOfTrainingSamplesCircleM];
	//intialize vec with +ve sample label == 1.0
	for(int i5 = 0; i5 < noOfPVeTrainingSamplesCircleM ; i5++){
		labelsCircle[i5] = 1.0;
	}
	for(int i6 = noOfPVeTrainingSamplesCircleM; i6 < noOfTrainingSamplesCircleM ; i6++){
			labelsCircle[i6] = -1.0;
	}

	 
    Mat labelsMatCircle(1, noOfTrainingSamplesCircleM, CV_32FC1, labelsCircle);
	// Set up SVM's parameters
    paramsCircle.svm_type    = CvSVM::C_SVC;
    paramsCircle.kernel_type = CvSVM::LINEAR;
    paramsCircle.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    SVMCircle.train(trainingDataMatCircle, labelsMatCircle, Mat(), Mat(), paramsCircle);
}
void GestureRecognizerModified::storeVectors(float xW,float yW, float zW){
	
	 
		/*xVector.push_back(xW);
		yVector.push_back(yW);
		zVector.push_back(zW);*/
   
}
bool GestureRecognizerModified::checkForGesture(vector<Point3i>& srcVector){
 
		
		bool status = false;
		status = ifVectorValuesAreInRange(srcVector);
		/*bool yStatus = false;
		yStatus = ifVectorValuesAreInRange(srcVector);
		bool zStatus = false;
		zStatus = ifVectorValuesAreInRange(srcVector);*/
		 
		 
		return status; 
}

bool GestureRecognizerModified::ifVectorValuesAreInRange(vector<Point3i>& srcVector){
	float size = srcVector.size();
	vector<float> result;
	 
	if(abs(srcVector[0].x - srcVector[srcVector.size() - 1].x) < 5 && abs(srcVector[0].y - srcVector[srcVector.size() - 1].y) < 5){
		return true;
	}

	//int elementsOutOfRange = findAdjacentDifference (srcVector);

	// if at least 5 or less elements are satisfying the condition, then hand is not moving ie its stable...
	/*if(elementsOutOfRange > 55 ){
		return true;
	}*/
	 

	return false;

}
 
int GestureRecognizerModified::findAdjacentDifference(vector<Point3i>& srcVector){

	int outOfRange = 0;
	int i = 0;
	//int iEnd = srcVector.size()-1;
	 
	// for(;i < iEnd ; i++){	
	//	float diffX = abs(srcVector[i].x - srcVector[i+1].x);
	//	float diffY = abs(srcVector[i].y - srcVector[i+1].y);
	//	// if diff is bigger than RANGE, that means the hand is moving, ie its not steady...
	//	if(diffX < RANGEM || diffY < RANGEM){
	//		outOfRange++;

	//		//resultVector.push_back(diff);
	//	} else {
	//		outOfRange = 0; //this is very imp.. it says that the points are has to be in order to be same
	//	}
	//	 
	//}
	return outOfRange;
}
int GestureRecognizerModified::findMedianOfVector(vector<int>& srcVector){
	size_t size = srcVector.size();
	int returnValue;
	if(size > 0){
		sort(srcVector.begin(), srcVector.end());
		if (size  % 2 == 0){
			returnValue = (srcVector[size / 2 - 1] + srcVector[size / 2]) / 2;
		}else {
			returnValue = srcVector[size / 2];
		}
	}

	return returnValue;
}

void GestureRecognizerModified::findEdgePointsForTrajectory(vector<Point3i>& srcVector,Point3i& XL,Point3i& XH,Point3i& YL,Point3i& YH,Rect& returnRectAroundGesture
	,int& gestureWidth,int& gestureHeight){
	
	//initialize global variable trajectory for gesture understanding..
	   
	//vector<int> returnReducedVector;
	XL.x = 160;
	XL.y = 120;
	XH.x = 0;
	XH.y = 0;

	YL.x = 160;
	YL.y = 120;
	YH.x = 0;
	YH.y = 0;

	for(int i = 0;i < srcVector.size();i++){
		//Find XL..
		if(srcVector[i].x < XL.x){
			XL.x = srcVector[i].x;
			XL.y = srcVector[i].y;
		}

		//find XH..
		if(srcVector[i].x > XH.x){
			XH.x = srcVector[i].x;
			XH.y = srcVector[i].y;
		}
			
		//find YL..
		if(srcVector[i].y < YL.y){
			YL.x = srcVector[i].x;
			YL.y = srcVector[i].y;
		}

		//find YH..
		if(srcVector[i].y > YH.y){
			YH.x = srcVector[i].x;
			YH.y = srcVector[i].y;
		}
	}

	gestureWidth = abs(XL.x-XH.x);
	gestureHeight = abs(YL.y-YH.y);
	

	returnRectAroundGesture =  Rect(XL.x,YL.y,gestureWidth,gestureHeight);
	  
		 

	//initialize some variables here..
	/*startPointOriginal = srcVector[0];
	stopPointOriginal = srcVector[srcVector.size()-1];*/

	 
	//findAngleVector(srcVector,returnReducedVector);

	//angleReturnedVec = angleVector;
	//removeDuplicateElements(angleVector,returnReducedVector);

	//returnReducedVector = angleVector; 
	
	//closenessCriteria = sqrt (pow ((startPointOriginal.x - stopPointOriginal.x),2.0) + pow ((startPointOriginal.y - stopPointOriginal.y),2.0));	//it checks if the start point and stop point are close enough to say its a close gesture or not?
							//sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20


	 
}
	 
void GestureRecognizerModified::removeDuplicateElements(vector<int>& srcVector,vector<int>& returnVector){
	if(srcVector.size() != 0){
		int currentEle = srcVector[0];
		returnVector.push_back(currentEle);
		for(int i = 1; i < srcVector.size(); i++){
		
			if(currentEle != srcVector[i]){
				returnVector.push_back(srcVector[i]);
				currentEle  = srcVector[i];
			}
		}
	}
}

void GestureRecognizerModified::findAngleVector(vector<Point3i>& srcVector,vector<int>& returnedAngleVector){
	vector<double> tempVector;
	size_t size = (srcVector.size() - 1); //untill second last element
	for(int i = 0; i < size; i++){
		tempVector.push_back(findAngleBetweenTwoPoints(srcVector[i],srcVector[i+1]));
	}

	

	//normalize angle vector with 45
	for(int j = 0 ; j < tempVector.size(); j++){
		double angle =  tempVector[j];
		if(angle == 360){
			angle = 0;
		}
		returnedAngleVector.push_back(angle/22.5);
	}

	 	 
}

void GestureRecognizerModified::findAngleVector(vector<Point>& srcVector,vector<int>& returnedAngleVector){
	vector<double> tempVector;
	size_t size = (srcVector.size() - 1); //untill second last element
	for(int i = 0; i < size; i++){
		tempVector.push_back(findAngleBetweenTwoPoints(srcVector[i],srcVector[i+1]));
	}

	

	//normalize angle vector with 45
	for(int j = 0 ; j < tempVector.size(); j++){
		double angle =  tempVector[j];
		if(angle == 360){
			angle = 0;
		}
		returnedAngleVector.push_back(angle/22.5);
	}

	 	 
}
 
template <typename Fwd>
typename std::map<typename std::iterator_traits<Fwd>::value_type, int>::value_type
GestureRecognizerModified::most_frequent_element(Fwd begin, Fwd end)
{
    std::map<typename std::iterator_traits<Fwd>::value_type, int> count;

    for (Fwd it = begin; it != end; ++it)
        ++count[*it];

    return *std::max_element(count.begin(), count.end(), by_second());
} 

 

void GestureRecognizerModified::circleTrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesCircleM][50] = {	

		{2, 2, 2, 2, 14, 14, 14, 14, 14, 14, 14, 12, 10, 10, 10, 10, 10, 8, 6, 10, 6, 6, 4, 4, 4, 2},
		{6, 2, 2, 2, 2, 0, 0, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 6, 8, 8, 8, 6, 6, 4, 6, 4, 2},
		{2, 2, 2, 2, 14, 14, 14, 14, 14, 12, 12, 12, 10, 10, 10, 10, 8, 6, 10, 6, 6, 4, 4, 4, 4},
		{2, 2, 2, 2, 2, 0, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 8, 8, 10, 6, 6, 4, 6, 4, 2},
		{0, 0, 2, 14, 14, 14, 12, 14, 14, 10, 10, 10, 10, 10, 8, 8, 6, 6, 6, 6, 4, 2},
		{2, 2, 2, 2, 14, 14, 14, 14, 14, 12, 12, 12, 10, 8, 10, 10, 6, 8, 6, 6, 4, 4, 4, 4},
		{2, 2, 0, 2, 14, 14, 14, 12, 14, 14, 10, 10, 10, 10, 10, 8, 8, 6, 8, 6, 6, 4, 2},
		{2, 2, 2, 0, 14, 14, 14, 14, 12, 14, 12, 12, 12, 10, 8, 10, 10, 6, 10, 8, 6, 6, 4, 4, 2, 4},
		{6, 2, 2, 2, 2, 14, 2, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 2},
		{2, 0, 0, 0, 14, 14, 14, 14, 14, 12, 10, 10, 10, 10, 10, 8, 8, 6, 6, 6, 6, 2},		//10th data
		{2, 2, 2, 14, 14, 14, 14, 14, 10, 12, 10, 10, 8, 6, 6, 6, 6, 4, 2, 4, 2},
		{2, 2, 14, 14, 14, 12, 12, 12, 10, 10, 10, 10, 6, 6, 6, 4, 6, 2, 2, 2},
		{2, 14, 14, 14, 14, 12, 12, 10, 10, 10, 10, 6, 8, 6, 6, 6, 2, 2, 2},
		{0, 14, 14, 14, 14, 12, 12, 10, 12, 10, 10, 8, 6, 6, 6, 6, 6, 4, 2, 2, 2},
		{0, 0, 14, 14, 14, 12, 14, 12, 10, 10, 10, 10, 6, 6, 6, 6, 2, 2, 2, 2, 0},
		{2, 14, 14, 14, 14, 12, 14, 12, 10, 10, 10, 8, 8, 6, 6, 4, 6, 4, 2, 4, 2},
		{2, 14, 14, 14, 12, 14, 12, 10, 10, 10, 10, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2},
		{2, 0, 14, 0, 14, 12, 14, 10, 10, 10, 10, 10, 6, 6, 6, 6, 2, 4, 4},
		{2, 0, 14, 14, 14, 12, 12, 10, 12, 10, 10, 6, 8, 6, 6, 4, 4, 2, 2},
		{0, 14, 14, 14, 14, 14, 12, 10, 10, 10, 8, 6, 8, 6, 6, 6, 4, 4, 2, 2, 2},		
		{2, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 6, 6, 6, 4, 4, 6, 4, 2},
		{2, 0, 14, 14, 14, 14, 12, 10, 10, 10, 10, 10, 6, 6, 6, 2, 6, 2, 2, 4},
		{2, 2, 0, 14, 14, 14, 12, 12, 10, 10, 10, 10, 8, 8, 6, 6, 4, 2, 2, 4},
		{2, 2, 2, 0, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 6, 10, 6, 6, 6, 4, 4, 2, 4},
		{2, 2, 2, 14, 14, 14, 14, 12, 12, 12, 10, 10, 10, 6, 6, 6, 6, 6, 6, 2},
		{0, 14, 14, 14, 12, 10, 10, 10, 10, 6, 6, 6, 6, 4, 4, 2, 4, 2},
		{14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 6, 6, 6, 6, 4, 4, 2, 2, 2},
		{14, 0, 14, 14, 12, 12, 10, 10, 10, 8, 6, 6, 6, 2, 4, 2, 2, 2},
		{14, 14, 14, 14, 14, 10, 10, 10, 10, 8, 6, 6, 6, 4, 2, 2, 2, 2},		//29th data
		/*{0, 15, 13, 12, 11, 9, 7, 5, 4, 4, 3}, 
		{1, 15, 13, 12, 11, 9, 7, 5, 4, 4, 3}, 
		{2, 0, 13, 12, 11, 9, 7, 5, 4, 4, 3}, 
		{4, 2, 0, 15, 13, 11, 9, 7, 5, 5, 4}, 
		{4, 2, 1, 15, 13, 11, 9, 7, 6, 5, 4}, 
		{3, 2, 0, 14, 12, 10, 8, 7, 5, 4, 3},
		{2, 2, 14, 14, 14, 12, 10, 10, 8, 8, 6, 6, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 6, 8, 6, 6, 2},
		{2, 2, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4},
		{2, 2, 14, 14, 14, 14, 10, 10, 10, 8, 6, 6, 4, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 10, 8, 6, 6, 4},
		{2, 0, 14, 14, 14, 12, 10, 10, 10, 6, 6, 4},
		{2, 2, 14, 14, 14, 14, 10, 10, 10, 8, 6, 6, 4},
		{2, 14, 14, 14, 12, 10, 10, 8, 8, 6, 6, 2},
		{2, 2, 0, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4, 2},
		{4, 2, 2, 2, 14, 14, 14, 14, 8, 10, 10, 6},
		{14, 14, 14, 10, 10, 10, 6, 6, 2, 2, 2},
		{14, 14, 14, 10, 10, 10, 6, 6, 6, 2, 2, 2},
		{14, 14, 14, 10, 10, 10, 6, 6, 4, 2, 2},
		{14, 14, 14, 12, 10, 10, 6, 6, 4, 2, 2, 2},
		{2, 0, 14, 14, 12, 10, 10, 10, 6, 2, 4},
		{14, 14, 14, 12, 10, 10, 6, 6, 6, 2, 2},
		{14, 14, 14, 12, 10, 10, 8, 6, 6, 2, 4},
		{14, 14, 12, 10, 10, 8, 6, 6, 2, 2},
		{2, 14, 14, 10, 10, 10, 6, 6, 6, 2},
		{2, 14, 14, 10, 10, 6, 6, 4, 2},
		{0, 14, 14, 10, 10, 10, 6, 6, 4, 2, 2},
		{2, 14, 14, 10, 10, 10, 6},
		{14, 12, 10, 10, 8, 6, 4, 2, 2},
		{2, 14, 14, 10, 10, 6, 6, 4, 2},
		{2, 14, 14, 10, 10, 10, 6, 4, 2, 2},
		{2, 14, 14, 10, 10, 10, 6, 4, 4, 2},
		{2, 14, 14, 10, 10, 8, 6, 6, 2, 2},
		{14, 14, 10, 10, 6, 6, 4, 2},
		{2, 0, 0, 14, 14, 12, 10, 10, 10, 6, 6, 4, 4},
		{2, 2, 14, 14, 14, 12, 10, 10, 8, 8, 6, 4, 4},
		{0, 2, 14, 14, 12, 10, 10, 10, 8, 6, 6, 2},
		{2, 0, 14, 14, 14, 14, 10, 10, 6, 10, 6, 6, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 10, 6, 6, 6, 2},
		{2, 2, 14, 14, 14, 10, 10, 10, 8, 6, 6, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 8, 8, 6, 6, 2},
		{2, 2, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4},
		{2, 2, 14, 14, 14, 12, 10, 10, 10, 8, 6, 6, 4},
		{2, 0, 14, 14, 14, 10, 10, 8, 8, 6, 6, 4},
		{2, 0, 14, 14, 14, 12, 10, 10, 6, 8, 6, 4},
		{2, 2, 14, 14, 14, 12, 10, 10, 10, 8, 6, 6, 4, 2},
		{2, 0, 14, 14, 14, 12, 10, 10, 8, 10, 6, 6, 6, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 8, 8, 6, 6, 2},
		{2, 2, 14, 14, 14, 14, 10, 10, 10, 8, 6, 6, 4, 2},
		{2, 2, 14, 14, 14, 10, 10, 10, 8, 6, 6, 2},
		{2, 2, 0, 14, 14, 14, 10, 10, 8, 8, 6, 6, 4},
		{2, 2, 14, 14, 14, 14, 10, 10, 10, 8, 6, 6, 4, 2},
		{2, 2, 14, 14, 14, 14, 10, 10, 10, 10, 6, 6, 6, 4},
		{2, 2, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4, 2},
		{2, 2, 14, 14, 14, 10, 10, 10, 10, 6, 6, 4, 2},
		{2, 2, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4},
		{2, 2, 0, 14, 14, 14, 10, 10, 8, 10, 6, 6, 4, 2},
		{2, 0, 14, 14, 14, 12, 10, 10, 10, 8, 6, 4, 2},
		{2, 2, 14, 14, 14, 12, 10, 10, 10, 6, 6, 6, 2},
		{2, 2, 0, 14, 14, 14, 10, 10, 10, 10, 6, 6, 6},
		{0, 14, 12, 10, 10, 6, 6, 6, 2, 2},
		{14, 14, 12, 10, 10, 10, 6, 6, 6, 2, 2, 2},
		{14, 14, 10, 10, 8, 6, 6, 4, 2, 2},
		{0, 14, 14, 14, 10, 10, 6, 6, 4, 2, 2},
		{14, 14, 10, 10, 10, 6, 6, 6, 2},
		{0, 14, 12, 10, 10, 10, 6, 6, 6, 2},
		{0, 14, 12, 10, 10, 6, 6, 4, 2},
		{2, 14, 14, 10, 10, 10, 6, 6, 2},
		{2, 14, 14, 14, 10, 10, 6, 6, 6, 4},
		{0, 14, 14, 12, 10, 10, 6, 6, 6, 2},
		{0, 14, 14, 10, 10, 8, 6, 6, 6, 2, 2},
		{14, 14, 12, 10, 10, 6, 6, 6, 2, 2},
		{0, 14, 14, 10, 10, 10, 6, 6, 4, 2},
		{14, 14, 14, 10, 10, 10, 8, 6, 6, 2, 2},
		{14, 14, 12, 10, 10, 10, 6, 6, 2, 2},
		{0, 14, 14, 10, 10, 6, 6, 4, 2},
		{0, 14, 14, 10, 10, 10, 6, 6, 4, 2, 2},
		{0, 14, 12, 10, 10, 6, 6, 2},
		{0, 14, 14, 10, 10, 10, 6, 6, 4, 2},
		{2, 14, 14, 10, 10, 10, 6, 6, 2},
		{2, 14, 14, 10, 10, 10, 6, 6},
		{0, 14, 14, 10, 10, 10, 6, 6, 2},
		{0, 14, 12, 10, 10, 6, 6, 2},
		{0, 14, 10, 10, 10, 6, 6, 2},
		{2, 14, 14, 10, 10, 10, 6, 6, 4},
		{0, 14, 14, 10, 10, 8, 6, 6, 4},
		{0, 14, 14, 10, 10, 10, 6, 6, 4, 2},
*/




		//{2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 9, 9, 8, 8, 7, 8, 4, 4, 4, 3, 2},
		//{2, 2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 10, 9, 9, 7, 8, 6, 7, 5, 4, 4, 4, 3, 3, 3},
		//{3, 3, 2, 1, 1, 0, 15, 14, 13, 12, 12, 11, 12, 10, 8, 11, 8, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3},
		//{2, 1, 0, 13, 12, 11, 10, 9, 8, 6, 4, 3, 3 },
		//{1, 0, 0, 13, 12, 12, 12, 11, 12, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		//{3, 3, 1, 0, 0, 14, 13, 13, 12, 12, 11, 11, 8, 9, 7, 6, 6, 4, 4, 4, 3},
		//{3, 2, 1, 1, 0, 0, 15, 15, 14, 13, 12, 13, 12, 11, 11, 9, 10, 9, 9, 8, 7, 8, 8, 7, 6, 4, 4, 4, 4, 4},
		//{2, 1, 2, 1, 0, 0, 0, 15, 14, 14, 12, 13, 12, 12, 11, 11, 11, 10, 9, 9, 8, 7, 8, 8, 7, 6, 7, 4, 4, 4, 4, 3, 4, 3, 3},
		//{2, 2, 1, 0, 0, 15, 14, 14, 12, 12, 12, 11, 11, 10, 10, 10, 9, 6, 8, 7, 7, 6, 5, 4, 4, 4, 3, 3, 3 }, 					 
		//{4, 2, 3, 3, 2, 0, 0, 14, 13, 13, 12, 12, 12, 12, 11, 10, 9, 8, 8, 8, 7, 6, 5, 4}, 
		//{2, 3, 2, 3, 0, 14, 13, 13, 12, 11, 10, 10, 8, 8, 7, 5, 4, 4},
		//{2, 1, 15, 14, 12, 10, 9, 8, 7, 5, 2},
		//{3, 1, 0, 15, 13, 13, 12, 11, 9, 8, 7, 6, 4, 4, 3, 3},
		//{3, 2, 1, 15, 13, 12, 12, 10, 9, 7, 7, 4, 3 },
		//{3, 2, 1, 15, 14, 12, 12, 11, 11, 8, 8, 7, 6, 5, 4, 3},
		//{3, 1, 0, 13, 12, 11, 9, 8, 8, 4, 3 },
		//{2, 0, 13, 12, 10, 9, 6, 4, 3 },//18th data.cicle data end
		//{2, 0, 15, 13, 12, 12, 11, 9, 8, 7, 5, 3, 4 },
		//{3, 2, 0, 15, 13, 12, 12, 11, 10, 8, 7, 6, 4, 4},
		//{3, 2, 1, 0, 14, 13, 12, 11, 12, 10, 10, 8, 6, 6, 5, 4, 4, 2},
		//{2, 1, 15, 15, 13, 12, 12, 11, 11, 9, 7, 6, 5, 4},
		//{3, 2, 0, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 6, 6, 4, 3},
		//{2, 1, 0, 14, 13, 12, 12, 11, 11, 10, 8, 7, 6, 7, 5, 4, 4, 3},
		//{3, 2, 1, 0, 15, 13, 12, 12, 10, 10, 9, 8, 7, 6, 5, 4, 3, 3},
		//{3, 3, 1, 15, 13, 13, 11, 10, 10, 8, 7, 6, 4, 4, 3},
		//{4, 4, 3, 1, 1, 0, 15, 14, 13, 12, 12, 11, 11, 8, 8, 7, 7, 6, 4},// 27th data.
		//{3, 2, 1, 0, 15, 14, 12, 12, 10, 11, 9, 8, 7, 7, 5, 4, 4, 3, 2 },
		//{2, 1, 0, 14, 13, 12, 12, 11, 9, 9, 7, 7, 5, 4, 3, 3},
		//{2, 1, 0, 15, 13, 12, 12, 11, 9, 9, 7, 6, 4, 4, 4, 4, 3},
		//{3, 2, 1, 1, 15, 14, 13, 12, 10, 11, 9, 7, 7, 6, 4, 4, 4, 3},
		//{4, 2, 1, 14, 13, 12, 11, 11, 9, 8, 8, 5, 4, 4, 3},
		//{4, 3, 2, 2, 0, 0, 15, 12, 11, 11, 11, 9, 7, 8, 6, 5, 4, 3},
		//{3, 3, 2, 0, 0, 14, 13, 12, 11, 10, 8, 7, 7, 5, 4, 4},
		//{4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5}, 
		//{2, 1, 0, 0, 13, 12, 12, 11, 10, 9, 8, 7, 7, 5, 4, 4, 4},
		//{2, 3, 2, 1, 15, 15, 13, 12, 11, 11, 9, 9, 7, 7, 6, 4, 4, 3},
		//{3, 2, 2, 1, 15, 15, 14, 12, 12, 12, 11, 11, 11, 9, 8, 6, 6, 5, 4, 3},
		//{2, 2, 0, 0, 14, 13, 13, 12, 11, 9, 8, 8, 6, 5, 4, 3},
		//{2, 1, 15, 15, 13, 12, 12, 11, 9, 9, 8, 7, 6, 4, 4, 3, 3},	
		//{3, 2, 1, 0, 0, 0, 13, 13, 12, 11, 11, 11, 10, 9, 8, 6, 6, 5, 4, 4, 2},
		//{2, 2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 9, 7, 7, 5, 4, 4, 3},
		//{3, 2, 2, 0, 0, 14, 13, 12, 12, 11, 9, 9, 8, 7, 6, 5, 4},
		//{2, 2, 0, 0, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 4},
		//{1, 1, 0, 14, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 3, 2},
		//{3, 2, 3, 1, 1, 0, 0, 14, 13, 13, 13, 12, 11, 11, 10, 9, 8, 8, 8, 7, 7, 6, 5, 4, 4, 3, 3},
		//{1, 2, 1, 0, 14, 13, 13, 12, 11, 11, 8, 9, 8, 7, 6, 5, 4, 4, 4, 4 },
		//{2, 2, 1, 0, 0, 15, 13, 12, 11, 11, 11, 8, 8, 8, 8, 7, 5, 4, 3},
		//{2, 1, 0, 0, 15, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6, 5, 4, 3, 3 },
		//{3, 3, 1, 1, 15, 14, 13, 13, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2},
		//{3, 3, 2, 2, 0, 14, 14, 13, 12, 12, 11, 10, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		//{3, 3, 1, 0, 0, 14, 13, 12, 12, 10, 9, 8, 8, 6, 5, 5, 4, 3 },
		//{2, 2, 1, 0, 15, 14, 12, 12, 12, 10, 8, 8, 8, 6, 6, 4, 4, 4, 3},
		//{2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 6, 5, 4, 4, 3 },
		//{2, 2, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6, 5, 4, 4, 2},
		//{3, 1, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 5, 6, 4, 4, 4},
		//{4, 4, 3, 2, 1, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6 },
		//{4, 3, 2, 1, 0, 0, 15, 14, 13, 12, 12, 11, 10, 9, 7, 8, 7, 6, 5, 3},
		//{2, 3, 2, 1, 1, 0, 0, 14, 14, 13, 12, 12, 12, 11, 11, 10, 9, 9, 7, 8, 7, 7, 6, 5, 4, 3, 3},		//until this works well..
		//{4,4,3,2,1,0,15,13,13,12,11,10,10,9,8,7,6,5},
		//{2,3,0,15,15,13,12,12,11,10,8,7,7,7,4,4,3},
		//{3,2,1,15,13,12,12,11,11,9,6,5,4,4,3},	
		//{2,1,0,14,12,11,11,10,10,8,6,5,5,4,3,3,2},	
		//{2,1,0,15,13,12,12,12,10,11,9,8,7,5,4,3,3},	 
		//{3,2,0,15,13,12,10,9,6,7,4,4},	//65th data.circle data end.
		//{2, 1, 0, 14, 13, 12, 11, 10, 9, 7, 7, 4, 4, 3},
		//{3, 1, 1, 15, 13, 13, 12, 11, 10, 10, 9, 7, 5, 4, 4, 4, 3, 0},
		//{2, 1, 0, 14, 12, 12, 11, 10, 10, 8, 6, 7, 5, 4, 4, 3, 3},
		//{3, 2, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 5, 4, 4, 3, 2},
		//{3, 2, 1, 15, 14, 12, 11, 10, 10, 7, 7, 5, 4, 3, 4},
		//{3, 1, 0, 14, 13, 12, 12, 12, 10, 10, 8, 8, 7, 7, 5, 4, 3, 3, 2},	//71st data  
		//{4, 2, 0, 15, 14, 13, 12, 12, 10, 12, 9, 8, 6, 5, 4, 3, 3},
		//{2, 2, 0, 0, 14, 13, 12, 11, 11, 10, 9, 7, 6, 5, 4, 3, 2},
		//{1, 15, 14, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},
		//{2, 1, 1, 15, 13, 12, 11, 10, 11, 8, 8, 8, 5, 4, 4, 2},
		//{3, 2, 0, 0, 14, 13, 11, 11, 8, 7, 6, 4, 3, 3},
		//{2, 0, 0, 14, 14, 13, 12, 11, 10, 10, 9, 7, 6, 4, 4, 3, 3},
		//{3, 2, 1, 0, 14, 13, 11, 10, 10, 10, 8, 7, 6, 4, 4, 3, 3},
		//{2, 1, 15, 13, 13, 12, 10, 10, 9, 8, 7, 4, 4, 4, 3 },
		//{3, 3, 0, 15, 0, 13, 12, 12, 10, 9, 9, 7, 6, 4, 3},				//80st data  
		//{2, 2, 1, 0, 14, 13, 13, 11, 10, 10, 9, 7, 7, 5, 4, 4, 3},
		//{2, 1, 0, 0, 14, 12, 13, 11, 11, 10, 10, 9, 8, 8, 6, 6, 5, 5, 4, 4, 4, 3, 2},
		//{2, 2, 1, 15, 13, 12, 11, 11, 11, 10, 9, 7, 7, 6, 5, 4, 3, 3 },
		//{2, 2, 1, 0, 15, 13, 12, 12, 11, 11, 9, 8, 8, 6, 4, 4, 3, 3 }, 
		//{2, 0, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 6, 5, 4, 4, 3},
		//{2, 1, 0, 15, 15, 13, 12, 11, 12, 10, 10, 10, 8, 8, 7, 7, 5, 4, 4, 3, 3, 2, 2, 2},
		//{1, 1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},					
		//{1, 1, 15, 14, 13, 12, 10, 10, 9, 7, 7, 7, 5, 4, 3, 3},
		//{4, 3, 0, 3, 15, 13, 13, 12, 11, 9, 7, 7, 5, 4, 4},
		//{3, 2, 1, 0, 0, 15, 13, 13, 12, 11, 10, 11, 7, 8, 6, 5, 4, 4, 3},
		//{3, 2, 0, 0, 13, 12, 12, 10, 10, 9, 6, 5, 4, 3 },
		//{2, 1, 0, 15, 13, 12, 11, 11, 9, 8, 6, 6, 5, 4},
		//{2, 0, 15, 14, 12, 12, 10, 9, 9, 7, 6, 5, 5, 4, 3},
		//{2, 1, 0, 15, 14, 12, 10, 10, 9, 7, 7, 5, 4, 3},
		//{3, 2, 0, 14, 13, 12, 10, 9, 7, 6, 5, 3},
		//{3,2,0,14,13,12,10,9,7,6,5,3},
		//{3, 2, 2, 0, 15, 14, 12, 12, 11, 9, 8, 8, 6, 5, 4},
		//{3, 2, 1, 0, 15, 13, 13, 12, 11, 10, 8, 8, 7, 6, 5, 4, 3},
		//{2, 1, 0, 15, 14, 13, 12, 11, 10, 10, 10, 7, 7, 6, 4, 4, 4, 3, 3},
		//{4, 4, 2, 2, 1, 15, 13, 13, 12, 10, 10, 10, 8, 7, 5, 4, 3, 3},	
		//{3, 4, 3, 2, 2, 1, 0, 15, 15, 15, 13, 12, 12, 12, 11, 10, 8, 9, 9, 8, 8, 6, 4, 4, 4, 3},
		//{2, 1, 0, 14, 13, 12, 11, 11, 10, 8, 7, 6, 5, 4, 3},		
		//{2, 2, 0, 15, 14, 13, 12, 12, 11, 11, 10, 8, 8, 6, 5, 5, 4, 3},
		//{2, 2, 2, 15, 15, 14, 13, 12, 11, 10, 9, 8, 8, 6, 4, 5, 4, 4, 3},
		//{2, 1, 0, 14, 15, 13, 12, 11, 11, 9, 8, 8, 6, 5, 4, 4, 3},
		//{2, 1, 0, 14, 14, 12, 12, 11, 10, 10, 7, 8, 8, 5, 4, 4, 4 }, 
		//{2, 1, 0, 15, 13, 13, 11, 10, 10, 8, 9, 6, 4, 5, 4, 3},
		//{3, 2, 0, 2, 15, 15, 13, 13, 12, 11, 10, 8, 8, 7, 7, 4, 6, 5, 4, 3, 3},
		//{2, 1, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 7, 4, 7, 4, 5, 4, 2},	
		//{2, 2, 15, 14, 14, 12, 11, 10, 9, 8, 6, 6, 5, 4, 3, 2},
		//{2, 1, 1, 15, 15, 13, 12, 11, 10, 9, 8, 7, 5, 5, 4, 4, 3 },
		//{3, 1, 0, 15, 14, 13, 12, 11, 10, 8, 7, 4, 6, 4, 3},
		//{2, 1, 1, 14, 13, 13, 12, 12, 10, 8, 8, 5, 5, 4, 3, 3 }, 
		//{2, 0, 1, 14, 14, 14, 13, 12, 11, 10, 10, 10, 7, 6, 6, 5, 4, 4, 3},
		//{3, 1, 2, 15, 15, 13, 13, 13, 12, 10, 10, 10, 8, 7, 6, 5, 4, 3, 3, 3},
		//{2, 0, 1, 14, 15, 15, 13, 12, 10, 11, 9, 9, 7, 7, 6, 5, 5, 3, 3, 3 },
		//{2, 1, 0, 0, 14, 13, 12, 12, 10, 10, 9, 7, 8, 6, 5, 4, 4, 3, 3 },
		//{2, 1, 0, 15, 14, 13, 12, 11, 11, 10, 10, 9, 10, 6, 7, 5, 7, 5, 4, 3, 2},
		//{2, 2, 0, 15, 14, 13, 12, 11, 11, 10, 8, 8, 6, 5, 4, 3, 3},
		//{2, 0, 15, 13, 13, 12, 11, 10, 8, 8, 6, 6, 5, 4, 4, 4, 2},
		//{2, 2, 1, 15, 13, 12, 11, 10, 10, 9, 7, 6, 4, 5, 4 },
		//{4, 1, 2, 15, 15, 13, 12, 10, 10, 9, 6, 8, 6, 4, 4 },
		//{2, 1, 1, 15, 14, 14, 12, 11, 10, 10, 9, 8, 7, 5, 5, 4, 3 },
		//{3, 1, 2, 14, 0, 14, 12, 12, 11, 10, 10, 8, 6, 5, 5, 4},
		//{2, 2, 15, 15, 13, 13, 12, 10, 10, 9, 7, 8, 6, 4, 4, 4, 4, 3},		
		//{2, 2, 1, 0, 15, 15, 14, 13, 13, 11, 11, 10, 10, 9, 9, 6, 8, 6, 5, 5, 4, 4 },
		//{2, 2, 1, 0, 14, 14, 12, 13, 11, 11, 10, 9, 9, 7, 6, 5, 4, 4, 4, 3},
		//{2, 2, 1, 0, 14, 14, 13, 14, 11, 10, 10, 10, 8, 7, 6, 6, 5, 4, 4, 3},		
		//{3,2,1,0,0,0,0,14,15,13,15,12,13,12,12,11,10,11,8,9,8,8,8,7,7,8,6,5,4,4,4,4,3,3},	
		//{1, 0, 1, 0, 0, 15, 13, 14, 13, 12, 11, 11, 11, 10, 9, 9, 7, 8, 7, 6, 5, 4, 4, 3, 3 },
		//{2, 1, 1, 0, 0, 14, 13, 0, 13, 13, 12, 11, 12, 10, 7, 9, 8, 7, 8, 7, 6, 4, 4, 4, 3, 3 },
		//{1, 0, 0, 0, 15, 15, 13, 13, 13, 12, 12, 11, 10, 9, 9, 7, 8, 8, 7, 6, 5, 4, 4, 4, 3},
		//{ 2, 1, 1, 0, 15, 13, 0, 13, 12, 12, 11, 12, 10, 8, 10, 8, 7, 8, 7, 6, 4, 4, 4, 4, 3, 3},												
		//{3,2,1,0,15,0,0,14,14,12,13,12,12,11,10,10,9,8,8,7,7,6,4,4,4,4,3},
		//{1,1,0,0,15,14,14,12,12,11,11,9,10,8,8,7,7,6,5,4,3},		
		//{2, 2, 0, 0, 15, 14, 14, 12, 12, 11, 11, 10, 9, 10, 7, 7, 7, 6, 4, 5, 4},
		//{2, 2, 0, 0, 15, 14, 14, 13, 12, 11, 10, 10, 7, 8, 8, 8, 5, 5, 4, 4, 4 },
		//{2, 2, 1, 0, 14, 13, 12, 12, 11, 11, 9, 8, 8, 7, 6, 4, 4, 4},
		//{3, 2, 2, 1, 0, 15, 13, 13, 12, 11, 11, 11, 8, 8, 8, 7, 5, 5, 4, 4},
		//{2, 2, 1, 0, 15, 13, 13, 11, 10, 9, 8, 8, 7, 5, 4, 4 },
		//{2, 3, 1, 1, 15, 15, 13, 12, 11, 12, 9, 9, 8, 6, 6, 4, 4 },
		//{3, 2, 1, 0, 15, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 6, 5, 4, 4},
		//{3, 3, 1, 2, 0, 15, 14, 15, 13, 13, 11, 12, 10, 9, 8, 8, 7, 6, 5, 4, 4},
		//{2, 2, 1, 0, 14, 13, 13, 11, 11, 10, 9, 7, 7, 5, 4, 4},
		//{2, 2, 2, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3 },
		//{1, 1, 0, 0, 13, 12, 11, 10, 9, 7, 7, 6, 4, 3 },
		//{2, 3, 1, 0, 0, 15, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4},
		//{2, 1, 0, 0, 13, 13, 12, 10, 10, 10, 8, 6, 6, 5, 4, 3},
		//{2, 1, 1, 0, 15, 14, 13, 12, 10, 9, 10, 7, 7, 6, 4, 4, 3, 4},
		//{2, 3, 1, 0, 0, 15, 14, 12, 11, 10, 10, 8, 7, 7, 6, 4, 4, 3},
		//{2, 2, 2, 14, 15, 14, 14, 13, 11, 11, 10, 10, 8, 8, 7, 6, 4, 5, 3, 3},
		//{1, 1, 15, 14, 13, 12, 11, 9, 9, 5, 5, 3 },
		//{3, 2, 2, 0, 15, 15, 13, 13, 12, 12, 10, 12, 10, 10, 8, 6, 8, 5, 5},
		//{2, 2, 0, 14, 15, 15, 13, 13, 11, 11, 11, 10, 9, 9, 6, 6, 5, 5, 4, 4 },//154th data circledata end.
										 
 
		{0, 15, 0, 14, 0, 3, 4, 6, 8, 9, 12, 15, 0, 14, 15, 0, 10, 9, 8, 7},		//random -ve samples 0th
		{3, 13, 12, 13, 12, 2, 3, 2, 3, 2, 13, 12},		 
		{3, 12, 11, 12, 11, 5, 4, 3, 0, 13, 12},
		{8, 4, 13, 3, 1, 0, 1, 7, 8, 7, 9, 4, 2, 1, 0, 15, 8},  
		{3, 4, 3, 13, 12, 13, 12, 4, 3, 4, 3, 2, 3, 2, 0, 13, 12, 8, 4 },
		{ 2, 1, 0, 1, 2, 12, 11, 13, 12, 8, 6, 8, 7, 8 },
		{2, 3, 2, 3, 12, 11, 1, 0, 15, 0, 3, 4, 5, 12},
		{4, 3, 13, 12, 13, 12, 3, 4, 3, 4, 0, 13, 12, 13, 12},
		{4, 3, 0, 12, 4, 6, 8, 9, 8, 9, 7, 0, 1, 0, 14},
		{7, 8, 9, 8, 13, 12, 2, 3, 4, 3, 4, 15, 13, 12, 11, 12, 5},
		{5, 4, 8, 10, 8, 0, 12, 2, 1, 0, 14, 10, 11, 5, 6, 8, 7, 8, 0},
		{4, 5, 6, 8, 10, 11, 12, 13, 14, 1, 3, 4, 5, 4, 11, 12, 11, 10, 1},
		{4, 2, 14, 11, 3, 1, 8, 9, 8, 9, 11, 10, 11, 0},
		{15, 14, 0, 5, 8, 9, 10, 1, 0, 15, 13, 12, 11, 3, 4, 3, 12, 10},
		{12, 8, 7, 8, 6, 0, 3, 0, 2, 1, 0, 11, 12},
		{4, 5, 7, 8, 10, 12, 3, 14, 4, 11, 0, 3, 2, 1},
		{4, 5, 6, 8, 9, 10, 12, 10, 0, 4, 0, 15, 13, 11},
		{7, 8, 9, 12, 1, 15, 14, 0, 3, 4, 5, 6, 8, 9, 8},
		{7, 8, 0, 1, 0, 15, 0, 11, 12, 11, 12, 11, 6, 9, 3, 7, 8 },
		{2, 1, 0, 14, 13, 12, 11, 10, 9 },
		{2, 1, 0, 15, 14, 13, 12, 11, 10, 2, 4, 5, 8  },
		{1, 0, 15, 13, 12, 4, 3, 4 },
		{2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12},			
		{4, 3, 0, 14, 13, 14, 13, 12, 3, 4, 3, 4, 0, 14, 12, 13, 12, 13, 12, 11},
		{0, 15, 0, 2, 7, 8, 6, 2, 0, 1, 0, 15},
		{15, 14, 0, 2, 3, 2, 3, 4, 5, 7, 8, 9, 8, 9, 10, 1},
		{ 4, 3, 4, 1, 15, 14, 13, 12, 13, 12, 13, 12, 4, 5, 6, 9, 10, 12, 11, 10},
		{15, 0, 7, 8, 7, 8, 7, 9},
		{4, 3, 4, 3, 4, 12, 11, 12 },
		{2, 10, 11 },
		{4, 3, 2, 0, 14, 13, 12, 11, 13, 12, 8, 7, 8, 0},
		{2, 3, 1, 0, 14, 13, 12, 11, 3, 4, 3, 8, 9, 10, 11 },
		{ 2, 1, 0, 14, 13, 12, 11, 4, 3, 2, 1, 0, 15, 14, 13, 11, 12 },
		{2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12},
		{0, 14, 0, 3, 4, 12, 11, 12, 1, 0, 12, 13, 11, 9},
		{4, 10, 9, 10, 12, 13, 12, 14, 13, 0, 1},
		{4, 3, 15, 13, 12, 4, 9, 11, 12, 11, 10, 11, 10, 11, 9},
		{ 4, 3, 4, 3, 4, 7, 8, 9, 11, 12, 14, 0, 4, 3, 2, 0, 13, 12, 11, 10},
		{3, 4, 8, 10, 12, 4, 3, 1, 0, 13, 12},
		{4, 2, 1, 0, 14, 13, 11, 12, 11},
		{4, 5, 6, 8, 9, 10, 11, 12},
		{14, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		{13, 14, 15, 0, 2, 4},
		{3, 4, 12, 11, 12, 11},
		{4,12,11}					//44th -ve random data..
		 
	};	//44th random -ve data

											

	Mat trainingDataMatTemp(noOfTrainingSamplesCircleM, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatCircle);
	 
}

 
void GestureRecognizerModified::copyArrayDataToVec(){
	
	float trainingDataTotal[155][50] = {
		{0, 15, 0, 14, 0, 3, 4, 6, 8, 9, 12, 15, 0, 14, 15, 0, 10, 9, 8, 7},		//random -ve samples 0th
		{3, 13, 12, 13, 12, 2, 3, 2, 3, 2, 13, 12},		 
		{3, 12, 11, 12, 11, 5, 4, 3, 0, 13, 12},
		{8, 4, 13, 3, 1, 0, 1, 7, 8, 7, 9, 4, 2, 1, 0, 15, 8},  
		{3, 4, 3, 13, 12, 13, 12, 4, 3, 4, 3, 2, 3, 2, 0, 13, 12, 8, 4 },
		{ 2, 1, 0, 1, 2, 12, 11, 13, 12, 8, 6, 8, 7, 8 },
		{2, 3, 2, 3, 12, 11, 1, 0, 15, 0, 3, 4, 5, 12},
		{4, 3, 13, 12, 13, 12, 3, 4, 3, 4, 0, 13, 12, 13, 12},
		{4, 3, 0, 12, 4, 6, 8, 9, 8, 9, 7, 0, 1, 0, 14},
		{7, 8, 9, 8, 13, 12, 2, 3, 4, 3, 4, 15, 13, 12, 11, 12, 5},
		{5, 4, 8, 10, 8, 0, 12, 2, 1, 0, 14, 10, 11, 5, 6, 8, 7, 8, 0},
		{4, 5, 6, 8, 10, 11, 12, 13, 14, 1, 3, 4, 5, 4, 11, 12, 11, 10, 1},
		{4, 2, 14, 11, 3, 1, 8, 9, 8, 9, 11, 10, 11, 0},
		{15, 14, 0, 5, 8, 9, 10, 1, 0, 15, 13, 12, 11, 3, 4, 3, 12, 10},
		{12, 8, 7, 8, 6, 0, 3, 0, 2, 1, 0, 11, 12},
		{4, 5, 7, 8, 10, 12, 3, 14, 4, 11, 0, 3, 2, 1},
		{4, 5, 6, 8, 9, 10, 12, 10, 0, 4, 0, 15, 13, 11},
		{7, 8, 9, 12, 1, 15, 14, 0, 3, 4, 5, 6, 8, 9, 8},
		{7, 8, 0, 1, 0, 15, 0, 11, 12, 11, 12, 11, 6, 9, 3, 7, 8 },
		{2, 1, 0, 14, 13, 12, 11, 10, 9 },
		{2, 1, 0, 15, 14, 13, 12, 11, 10, 2, 4, 5, 8  },
		{1, 0, 15, 13, 12, 4, 3, 4 },
		{2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12},			
		{4, 3, 0, 14, 13, 14, 13, 12, 3, 4, 3, 4, 0, 14, 12, 13, 12, 13, 12, 11},
		{0, 15, 0, 2, 7, 8, 6, 2, 0, 1, 0, 15},
		{15, 14, 0, 2, 3, 2, 3, 4, 5, 7, 8, 9, 8, 9, 10, 1},
		{ 4, 3, 4, 1, 15, 14, 13, 12, 13, 12, 13, 12, 4, 5, 6, 9, 10, 12, 11, 10},
		{15, 0, 7, 8, 7, 8, 7, 9},
		{4, 3, 4, 3, 4, 12, 11, 12 },
		{2, 10, 11 },
		{4, 3, 2, 0, 14, 13, 12, 11, 13, 12, 8, 7, 8, 0},
		{2, 3, 1, 0, 14, 13, 12, 11, 3, 4, 3, 8, 9, 10, 11 },
		{ 2, 1, 0, 14, 13, 12, 11, 4, 3, 2, 1, 0, 15, 14, 13, 11, 12 },
		{2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12},
		{0, 14, 0, 3, 4, 12, 11, 12, 1, 0, 12, 13, 11, 9},
		{4, 10, 9, 10, 12, 13, 12, 14, 13, 0, 1},
		{4, 3, 15, 13, 12, 4, 9, 11, 12, 11, 10, 11, 10, 11, 9},
		{ 4, 3, 4, 3, 4, 7, 8, 9, 11, 12, 14, 0, 4, 3, 2, 0, 13, 12, 11, 10},
		{3, 4, 8, 10, 12, 4, 3, 1, 0, 13, 12},
		{4, 2, 1, 0, 14, 13, 11, 12, 11},
		{4, 5, 6, 8, 9, 10, 11, 12},
		{14, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		{13, 14, 15, 0, 2, 4},
		{3, 4, 12, 11, 12, 11},
		{4,12,11},//154th data circledata end.
	};

	Mat trainingDataMatTemp(153, 50, CV_32FC1, trainingDataTotal);

	cv::FileStorage file;
			file.open("vectors.text", cv::FileStorage::WRITE);

			vector<vector<int>> vectors;
			vector<int> testVec,reducedVec;
	for(int j = 0; j< trainingDataMatTemp.rows; j++){
		for(int i = 0; i< trainingDataMatTemp.cols; i++){
			testVec.push_back(trainingDataMatTemp.at<float>(j,i));
		}
		removeDuplicateElements(testVec,reducedVec);
		vectors.push_back(reducedVec);
		ostringstream convert;   // stream used for the conversion
		convert << j; 
		file<<"vec"<<reducedVec ; 		  
		reducedVec.clear();
		testVec.clear();
	}
	file.release();
}
 
void GestureRecognizerModified::LetterSTrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesLetterSM][50] = {	
		{10, 10, 10, 14, 14, 14, 14, 12, 12, 10, 10, 10, 6},
		{10, 10, 14, 14, 14, 12, 14, 12, 10, 10, 10, 6},
		{8, 10, 10, 12, 14, 0, 0, 2, 14, 12, 12, 10, 10, 10, 8, 8, 8, 6},
		{10, 10, 10, 10, 14, 14, 0, 14, 14, 10, 10, 10, 10, 6, 6, 6},
		{8, 10, 10, 10, 14, 14, 2, 14, 12, 10, 10, 10, 10, 10, 6, 6, 6},
		{8, 10, 12, 14, 14, 12, 14, 10, 10, 10, 10, 10, 6},
		{10, 10, 14, 14, 14, 10, 10, 10, 10, 8},
		{10, 10, 14, 14, 14, 14, 10, 12, 10, 10, 10, 8},
		{10, 14, 14, 14, 14, 12, 10, 10, 6},
		{10, 10, 14, 14, 14, 14, 12, 10, 10, 10, 10, 8, 8},		//10th data
		{6, 10, 10, 12, 14, 12, 12, 10, 10, 10},
		{10, 14, 12, 10, 10, 10, 10, 10 },
		{6, 10, 10, 14, 14, 14, 14, 0, 12, 10, 10, 10, 10},
		{10, 10, 14, 12, 14, 14, 10, 10, 10, 10},
		{10, 10, 12, 12, 14, 14, 14, 10, 10, 8},
		{6, 6, 10, 14, 14, 14, 14, 12, 10, 10, 6, 10, 6, 6},	//16th data

		//data for digit 6 : not added yet
		{8, 9, 11, 11, 12, 12, 13, 14, 2, 1, 4, 5, 8, 9, 11},
		{8, 8, 10, 10, 11, 11, 12, 13, 15, 3, 3, 4, 7, 9},
		{7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 2, 2, 4, 4, 7, 9, 10},
		{8, 8, 8, 10, 11, 11, 12, 13, 13, 15, 2, 4, 4, 4, 6, 8, 9, 11},
		{8, 8, 8, 9, 11, 11, 12, 13, 14, 0, 1, 2, 4, 5, 7, 8, 10, 11},
		{8, 9, 9, 10, 12, 12, 13, 14, 0, 1, 3, 4, 5, 8, 9, 11, 11},
		{8, 8, 9, 10, 11, 12, 13, 14, 0, 1, 4, 4, 5, 8, 10, 11, 11},
		{8, 8, 9, 10, 12, 11, 12, 12, 12, 12, 13, 13, 0, 3, 3, 4, 5, 8, 10, 11 },
		{8, 8, 9, 9, 11, 11, 12, 12, 13, 14, 15, 1, 3, 4, 4, 6, 9, 10, 11},
		{8, 8, 9, 11, 12, 12, 12, 13, 14, 1, 2, 4, 4, 6, 9, 10},
		{9, 10, 11, 12, 13, 14, 1, 3, 5, 7, 10, 11},
		{8, 9, 10, 10, 11, 12, 13, 14, 1, 0, 3, 4, 4, 6, 8, 9, 11, 11},
		{7, 9, 9, 9, 11, 11, 12, 12, 12, 14, 0, 1, 2, 4, 5, 7, 9, 10},
		{8, 8, 10, 10, 11, 12, 12, 13, 14, 0, 2, 3, 4, 6, 9, 10},
		{8, 9, 9, 9, 11, 11, 11, 12, 12, 13, 13, 14, 0, 0, 0, 3, 3, 4, 5, 7, 8, 10, 11 },
		{8, 9, 9, 10, 10, 12, 12, 13, 13, 0, 1, 2, 3, 4, 4, 6, 8, 9, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 14, 14, 0, 0, 1, 3, 4, 4, 6, 8, 9, 11, 11, 11},
		{7, 8, 10, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 1, 2, 3, 3, 4, 5, 8, 9, 10, 10, 11, 11},
		{8, 8, 9, 9, 11, 11, 11, 12, 12, 12, 13, 13, 15, 15, 1, 1, 4, 4, 5, 6, 7, 9, 11, 11, 11},
		{8, 8, 9, 9, 10, 10, 10, 12, 11, 12, 13, 13, 13, 15, 0, 1, 2, 3, 4, 5, 6, 9, 10, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		//{8, 8, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 13, 14, 14, 0, 1, 1, 3, 3, 4, 5, 6, 8, 8, 10, 11, 11},
		//{7, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 1, 1, 2, 4, 5, 7, 9, 10, 11, 11, 11},
		//{7, 8, 8, 9, 10, 9, 10, 11, 11, 11, 12, 13, 12, 13, 15, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 11, 11},
		//{8, 8, 8, 10, 10, 10, 10, 11, 10, 12, 12, 12, 13, 13, 15, 15, 0, 1, 1, 2, 3, 4, 5, 7, 8, 8, 9, 11, 11, 11},
		//{8, 8, 10, 10, 11, 11, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 8, 9, 11, 11},
		//{8, 8, 8, 9, 10, 11, 11, 11, 12, 13, 12, 13, 14, 15, 1, 1, 3, 4, 4, 5, 6, 8, 9, 9, 11, 11},
		//{ 8, 9, 9, 9, 10, 11, 12, 12, 12, 13, 14, 15, 0, 1, 2, 4, 5, 7, 8, 9, 10, 12},
		//{8, 8, 8, 10, 10, 11, 12, 12, 12, 13, 14, 15, 0, 2, 3, 4, 5, 8, 9, 10, 11},
		//{8, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 0, 2, 3, 4, 6, 8, 10, 11},
		//{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		//{9, 10, 11, 12, 14, 14, 1, 4, 7, 9},
		//{8, 10, 11, 12, 13, 15, 2, 3, 4, 8, 10 },
		//{8, 9, 9, 11, 11, 12, 12, 13, 15, 1, 2, 4, 5, 7, 8, 10},
		//{8, 9, 10, 11, 11, 12, 14, 0, 2, 3, 4, 6, 8, 10},
		//{8, 9, 10, 11, 11, 12, 12, 15, 0, 2, 3, 4, 8, 9, 11},
		//{8, 9, 10, 11, 12, 12, 13, 14, 2, 3, 4, 7, 9, 10},
		//{8, 9, 10, 11, 11, 12, 12, 13, 0, 1, 2, 4, 6, 8, 10},
		//{8, 9, 10, 10, 12, 11, 12, 13, 15, 1, 3, 3, 4, 5, 8, 9, 10},
		//{8, 9, 10, 10, 11, 12, 13, 14, 0, 2, 3, 6, 8, 9},
		//{8, 9, 9, 10, 11, 12, 12, 14, 0, 2, 3, 4, 6, 9},		//41st data end +ve
		//{8, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 15, 2, 3, 4, 4, 4, 6, 8, 10, 11},
		//{8, 9, 9, 10, 11, 12, 11, 12, 12, 12, 13, 14, 14, 1, 2, 3, 4, 5, 5, 7, 8, 10, 10},
		//{7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 4, 5, 6, 8, 9, 10},
		//{7, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 13, 14, 0, 2, 3, 4, 5, 6, 9},
		//{8, 8, 9, 10, 10, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 5, 7, 9, 10, 12},
		//{8, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 15, 15, 1, 2, 4, 5, 5, 8, 9, 10},
		//{8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 15, 2, 3, 4, 6, 8, 11},
		//{7, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 9, 10},
		//{8, 8, 9, 10, 10, 10, 11, 10, 11, 12, 13, 13, 14, 14, 1, 2, 4, 4, 5, 7, 8, 10},
		//{7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 12, 12, 13, 15, 0, 1, 3, 4, 4, 6, 8, 9, 10},
		//{8, 8, 8, 10, 9, 10, 11, 11, 12, 12, 12, 13, 14, 15, 0, 1, 3, 4, 5, 6, 8, 9},		//52nd data end..
		 
	};
	Mat trainingDataMatTemp(noOfTrainingSamplesLetterSM, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatLetterS);

}

void GestureRecognizerModified::Digit1TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit1M][50] = {	
		{2, 2, 2, 2, 2, 12, 12, 12, 12, 12, 12, 12, 12, 10, 12},
		{2, 0, 2, 2, 14, 12, 12, 12, 12, 12, 12, 10, 12, 12, 12, 12, 10},
		{2, 2, 2, 2, 2, 2, 12, 12, 12, 10, 12, 12, 10, 12, 12, 12, 12, 12, 10},
		{2, 0, 2, 2, 2, 10, 12, 12, 10, 12, 12, 12, 12, 12, 12, 12},
		{2, 2, 0, 4, 0, 12, 12, 12, 12, 12, 12, 10, 12, 14, 10 },
		{2, 0, 2, 2, 2, 14, 12, 12, 10, 12, 12, 12, 12, 14, 10, 12},
		{2, 2, 2, 0, 10, 10, 12, 10, 12, 12, 12, 12, 12, 10, 12, 12},
		{2, 2, 2, 2, 2, 14, 12, 12, 12, 12, 10, 12, 12, 12, 12, 10},
		{2, 2, 2, 2, 2, 2, 14, 12, 12, 12, 12, 12, 12, 12, 10, 12},
		{2, 2, 2, 2, 2, 14, 10, 12, 12, 12, 12, 12, 12, 10, 12, 12, 12},		
		{2, 2, 14, 12, 10, 10, 10, 10},
		{2, 2, 12, 12, 10, 14, 12, 10, 10, 12},
		{2, 2, 14, 12, 10, 12, 10, 12},
		{2, 10, 12, 12, 14, 12, 10},	//14th data
		 
		 //data for digit 4
		{11, 11, 11, 11, 15, 15, 0, 0, 0, 4, 4, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10},
		{12, 11, 11, 11, 14, 0, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 11, 11, 11, 11, 15, 15, 0, 3, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12},
		{11, 11, 11, 11, 11, 12, 15, 0, 15, 0, 15, 3, 4, 4, 4, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 11, 12, 12, 11 },
		{11, 11, 11, 11, 11, 14, 0, 0, 0, 4, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 12, 10},
		{11, 11, 12, 11, 11, 14, 0, 15, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12},
		{11, 11, 12, 11, 11, 14, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 0, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 10, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 15, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 12, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12, 12},
		{11, 11, 11, 11, 12, 11, 14, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 11, 12, 12, 12, 11, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 3, 3, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 12, 11, 11, 0, 0, 0, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 11} ,
		{11, 11, 11, 11, 11, 13, 0, 0, 0, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 0, 0, 4, 4, 3, 4, 12, 11, 11, 11, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 11, 0, 0, 0, 0, 2, 4, 3, 4, 4, 12, 12, 11, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 1, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 12, 12, 11, 11, 11, 0, 0, 1, 3, 4, 4, 4, 4, 12, 12, 12, 11, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 11, 11, 12, 15, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},			//30th data end
		{11, 11, 11, 11, 11, 0, 0, 1, 3, 4, 4, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 12, 0, 15, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 12, 12, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 10, 11, 11, 11, 11, 14, 0, 14, 3, 11, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 11},
		{11, 10, 11, 12, 11, 11, 13, 0, 15, 0, 2, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 10},
		{11, 11, 11, 11, 10, 11, 0, 0, 0, 0, 3, 4, 3, 3, 12, 12, 11, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 15, 0, 3, 4, 4, 4, 11, 12, 12, 11, 12, 11, 12, 11},
		{11, 11, 12, 12, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 0, 0, 0, 0, 4, 4, 3, 4, 12, 12, 12, 11, 12, 12, 12, 12, 11, 11},
		{11, 11, 11, 11, 12, 0, 0, 15, 0, 15, 0, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 12, 11, 12},	
		{12, 11, 11, 11, 11, 11, 15, 0, 15, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 10, 11, 13, 0, 15, 0, 0, 4, 4, 4, 3, 4, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 12, 11, 11, 11, 15, 15, 0, 15, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 11, 12},
		{12, 11, 11, 11, 11, 14, 15, 0, 1, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{12, 11, 11, 11, 11, 13, 15, 0, 0, 0, 4, 3, 4, 4, 12, 12, 12, 12, 12, 11, 11, 12},
		{12, 11, 11, 12, 11, 11, 11, 11, 0, 15, 0, 0, 2, 4, 4, 4, 3, 4, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12},
		{12, 11, 12, 11, 11, 14, 0, 15, 0, 2, 4, 4, 4, 12, 11, 12, 12, 12, 11, 12, 11},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 12, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{12, 11, 11, 11, 15, 15, 0, 2, 4, 4, 12, 11, 12, 12, 11, 12, 11 },		
		{11, 11, 11, 11, 0, 0, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12}, 
		{11, 12, 11, 14, 0, 0, 4, 4, 12, 12, 12, 12, 12},
		{11, 11, 11, 14, 0, 0, 4, 4, 12, 12, 12, 12, 12},
		{11, 11, 11, 15, 0, 2, 4, 12, 12, 12, 12, 12},
		{11, 11, 11, 15, 0, 3, 4, 12, 12, 12, 12, 12},
		{11, 12, 11, 15, 0, 3, 4, 12, 12, 12, 12, 12, 12},
		{11, 10, 11, 11, 11, 15, 0, 0, 0, 2, 4, 4, 12, 11, 12, 12, 11, 12},
		{11, 11, 11, 14, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12},
		{11, 11, 12, 13, 0, 3, 4, 4, 11, 12, 12, 11, 12},
		{11, 11, 11, 12, 15, 15, 0, 2, 4, 4, 12, 12, 11, 12, 12},
		{11, 11, 11, 12, 15, 15, 0, 2, 4, 4, 12, 12, 11, 12, 12},
		{12, 11, 11, 15, 0, 0, 4, 4, 4, 12, 12, 12, 12, 12, 11},
		{11, 12, 15, 0, 2, 4, 4, 3, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 15, 15, 3, 4, 12, 12, 12, 12},
		{11, 11, 12, 1, 15, 3, 4, 4, 12, 12, 12, 12, 12},
		{11, 11, 14, 15, 0, 3, 4, 11, 12, 12, 11, 12, 11},
		{11, 11, 11, 11, 0, 0, 15, 3, 4, 4, 11, 12, 11, 12, 12},
		{11, 10, 15, 0, 3, 4, 4, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 15, 0, 2, 4, 12, 12, 12, 12, 12, 11},		//69th data end



	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit1M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit1);

}

void GestureRecognizerModified::Digit2TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit2M][50] = {	
		{2, 2, 14, 14, 10, 10, 10, 10, 10, 14, 2, 14, 2},
		{2, 2, 14, 14, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 0},
		{2, 2, 14, 12, 10, 10, 10, 10, 10, 2, 14, 2, 0},
		{2, 2, 14, 12, 12, 10, 10, 10, 10, 10, 0, 14, 0},
		{2, 2, 12, 12, 10, 10, 10, 10, 14, 0, 0, 0},
		{2, 2, 14, 12, 10, 10, 10, 10, 10, 0, 14, 2, 2},
		{2, 2, 14, 12, 12, 10, 10, 10, 10, 10, 0, 2, 0},
		{2, 14, 12, 12, 10, 10, 10, 10, 10, 10, 14, 2, 2, 0},
		{2, 0, 14, 14, 12, 10, 10, 10, 10, 10, 12, 2, 2, 0, 0},
		{2, 2, 14, 12, 10, 10, 10, 10, 10, 14, 0, 2},		//10th data
		 
		 

		{11, 11, 11, 11, 15, 15, 0, 0, 0, 4, 4, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10},
		{12, 11, 11, 11, 14, 0, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 11, 11, 11, 11, 15, 15, 0, 3, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12},
		{11, 11, 11, 11, 11, 12, 15, 0, 15, 0, 15, 3, 4, 4, 4, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 11, 12, 12, 11 },
		{11, 11, 11, 11, 11, 14, 0, 0, 0, 4, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 12, 10},
		{11, 11, 12, 11, 11, 14, 0, 15, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12},
		{11, 11, 12, 11, 11, 14, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 0, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 10, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 15, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 12, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12, 12},
		{11, 11, 11, 11, 12, 11, 14, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 11, 12, 12, 12, 11, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 3, 3, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 12, 11, 11, 0, 0, 0, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 11} ,
		{11, 11, 11, 11, 11, 13, 0, 0, 0, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 0, 0, 4, 4, 3, 4, 12, 11, 11, 11, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 11, 0, 0, 0, 0, 2, 4, 3, 4, 4, 12, 12, 11, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 1, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 12, 12, 11, 11, 11, 0, 0, 1, 3, 4, 4, 4, 4, 12, 12, 12, 11, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 11, 11, 12, 15, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},			//30th data end
		{11, 11, 11, 11, 11, 0, 0, 1, 3, 4, 4, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 12, 0, 15, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 12, 12, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 10, 11, 11, 11, 11, 14, 0, 14, 3, 11, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 11},
		{11, 10, 11, 12, 11, 11, 13, 0, 15, 0, 2, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 10},
		{11, 11, 11, 11, 10, 11, 0, 0, 0, 0, 3, 4, 3, 3, 12, 12, 11, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 15, 0, 3, 4, 4, 4, 11, 12, 12, 11, 12, 11, 12, 11},
		{11, 11, 12, 12, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 0, 0, 0, 0, 4, 4, 3, 4, 12, 12, 12, 11, 12, 12, 12, 12, 11, 11},
		{11, 11, 11, 11, 12, 0, 0, 15, 0, 15, 0, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 12, 11, 12},	
		{12, 11, 11, 11, 11, 11, 15, 0, 15, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 10, 11, 13, 0, 15, 0, 0, 4, 4, 4, 3, 4, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 12, 11, 11, 11, 15, 15, 0, 15, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 11, 12},
		{12, 11, 11, 11, 11, 14, 15, 0, 1, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{12, 11, 11, 11, 11, 13, 15, 0, 0, 0, 4, 3, 4, 4, 12, 12, 12, 12, 12, 11, 11, 12},
		{12, 11, 11, 12, 11, 11, 11, 11, 0, 15, 0, 0, 2, 4, 4, 4, 3, 4, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12},
		{12, 11, 12, 11, 11, 14, 0, 15, 0, 2, 4, 4, 4, 12, 11, 12, 12, 12, 11, 12, 11},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 12, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{12, 11, 11, 11, 15, 15, 0, 2, 4, 4, 12, 11, 12, 12, 11, 12, 11 },		
		{11, 11, 11, 11, 0, 0, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12}, //51th data end

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit2M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit2);

}


void GestureRecognizerModified::Digit3TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit3M][50] = {	
		
		{2, 14, 10, 10, 10, 0, 14, 14, 12, 10, 10, 10, 10, 8, 8},
		{14, 14, 14, 10, 10, 14, 14, 12, 10, 10, 8, 6},
		{14, 14, 14, 12, 10, 10, 14, 14, 14, 10, 10, 10, 10, 6},
		{2, 14, 14, 10, 10, 10, 2, 14, 14, 10, 10, 10, 6, 6},
		{2, 14, 12, 12, 10, 10, 14, 14, 10, 10, 8, 6},
		{0, 14, 14, 10, 10, 0, 14, 14, 10, 10, 10, 8, 6},
		{0, 14, 14, 10, 10, 14, 14, 12, 10, 10, 10, 8, 6},
		{0, 14, 10, 10, 14, 14, 12, 12, 10, 10, 6, 8},
		{14, 14, 12, 10, 10, 14, 14, 14, 12, 10, 10, 10, 6, 6},
		{14, 14, 14, 12, 10, 10, 14, 14, 12, 12, 10, 10, 10, 6, 6},		//10th data
		{0, 14, 14, 10, 10, 2, 14, 14, 10, 10, 10, 10, 8},
		{14, 14, 10, 10, 10, 0, 14, 14, 10, 10, 10, 6, 6},
		{0, 0, 14, 10, 10, 10, 10, 2, 14, 14, 14, 10, 10, 10, 8, 6},
		{0, 14, 14, 10, 10, 0, 14, 12, 10, 10, 10, 6, 6},
		{0, 14, 14, 12, 10, 10, 0, 14, 14, 10, 10, 8, 8, 6},
		{14, 14, 12, 10, 10, 14, 14, 10, 10, 10, 10, 8, 6},
		{14, 14, 14, 10, 10, 14, 14, 12, 10, 10, 10, 10, 8, 6},
		{0, 14, 14, 12, 10, 10, 10, 0, 14, 14, 14, 10, 10, 8, 8},
		{0, 14, 12, 10, 10, 10, 0, 0, 14, 12, 10, 10, 10, 6, 8},
		{0, 0, 14, 12, 10, 10, 14, 14, 12, 12, 10, 10, 8, 6},		//20th data
	 

		 //-ve data as Digit4
		{11, 11, 11, 11, 15, 15, 0, 0, 0, 4, 4, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10},
		{12, 11, 11, 11, 14, 0, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 11, 11, 11, 11, 15, 15, 0, 3, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12},
		{11, 11, 11, 11, 11, 12, 15, 0, 15, 0, 15, 3, 4, 4, 4, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 11, 12, 12, 11 },
		{11, 11, 11, 11, 11, 14, 0, 0, 0, 4, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 12, 10},
		{11, 11, 12, 11, 11, 14, 0, 15, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 15, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 11, 11, 15, 0, 0, 0, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12},
		{11, 11, 12, 11, 11, 14, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 11, 15, 0, 0, 0, 0, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 11 },
		{11, 11, 11, 10, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 15, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 12, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 11, 12, 12},
		{11, 11, 11, 11, 12, 11, 14, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 12, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 11, 12, 12, 12, 11, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 12, 11, 11, 11, 14, 0, 0, 3, 3, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 12, 11, 11, 0, 0, 0, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 11} ,
		{11, 11, 11, 11, 11, 13, 0, 0, 0, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 0, 0, 4, 4, 3, 4, 12, 11, 11, 11, 12, 12, 12, 12},
		{11, 11, 11, 12, 11, 11, 0, 0, 0, 0, 2, 4, 3, 4, 4, 12, 12, 11, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 1, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12},
		{11, 11, 12, 12, 11, 11, 11, 0, 0, 1, 3, 4, 4, 4, 4, 12, 12, 12, 11, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 0, 0, 0, 3, 4, 4, 4, 12, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 11, 11, 12, 15, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11},			//30th data end
		{11, 11, 11, 11, 11, 0, 0, 1, 3, 4, 4, 12, 12, 12, 12, 12, 12},
		{12, 11, 11, 11, 12, 0, 15, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11},
		{11, 12, 11, 12, 12, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 12, 11, 11},
		{11, 10, 11, 11, 11, 11, 14, 0, 14, 3, 11, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 11},
		{11, 10, 11, 12, 11, 11, 13, 0, 15, 0, 2, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 10},
		{11, 11, 11, 11, 10, 11, 0, 0, 0, 0, 3, 4, 3, 3, 12, 12, 11, 12, 12, 11, 12, 12, 11},
		{11, 11, 11, 11, 11, 11, 14, 15, 0, 3, 4, 4, 4, 11, 12, 12, 11, 12, 11, 12, 11},
		{11, 11, 12, 12, 12, 11, 15, 0, 0, 0, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{11, 11, 12, 11, 11, 11, 0, 0, 0, 0, 4, 4, 3, 4, 12, 12, 12, 11, 12, 12, 12, 12, 11, 11},
		{11, 11, 11, 11, 12, 0, 0, 15, 0, 15, 0, 4, 4, 4, 4, 12, 11, 12, 12, 11, 12, 12, 12, 11, 12},	
		{12, 11, 11, 11, 11, 11, 15, 0, 15, 0, 4, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12},
		{11, 11, 11, 10, 11, 13, 0, 15, 0, 0, 4, 4, 4, 3, 4, 12, 12, 12, 11, 12, 11, 12, 12},
		{11, 12, 11, 11, 11, 15, 15, 0, 15, 4, 3, 12, 12, 11, 12, 11, 12, 12, 12, 11},
		{11, 11, 11, 11, 11, 14, 15, 0, 0, 3, 4, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 11, 12},
		{12, 11, 11, 11, 11, 14, 15, 0, 1, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11},
		{12, 11, 11, 11, 11, 13, 15, 0, 0, 0, 4, 3, 4, 4, 12, 12, 12, 12, 12, 11, 11, 12},
		{12, 11, 11, 12, 11, 11, 11, 11, 0, 15, 0, 0, 2, 4, 4, 4, 3, 4, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12},
		{12, 11, 12, 11, 11, 14, 0, 15, 0, 2, 4, 4, 4, 12, 11, 12, 12, 12, 11, 12, 11},
		{11, 11, 11, 11, 11, 15, 0, 0, 2, 4, 4, 4, 12, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{12, 11, 11, 11, 15, 15, 0, 2, 4, 4, 12, 11, 12, 12, 11, 12, 11 },		
		{11, 11, 11, 11, 0, 0, 0, 2, 4, 4, 12, 12, 12, 12, 12, 12}, //51th data end

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit3M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit3);

}


void GestureRecognizerModified::Digit4TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit4M][50] = {	

		//data for digit 4
		{10, 10, 10, 10, 14, 2, 0, 2, 4, 4, 6, 12, 12, 12, 12, 12, 10},
		{10, 10, 10, 0, 2, 14, 2, 4, 4, 6, 4, 12, 12, 14, 12, 12, 10},
		{10, 10, 10, 10, 10, 2, 0, 2, 0, 4, 2, 4, 12, 10, 12, 10, 12, 12},
		{10, 10, 10, 10, 10, 14, 0, 0, 0, 2, 4, 4, 12, 10, 12, 12, 12},
		{10, 10, 10, 10, 0, 0, 14, 2, 2, 4, 6, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12},
		{10, 10, 10, 10, 10, 10, 0, 0, 14, 2, 4, 4, 6, 14, 12, 12, 12, 12, 12, 12},
		{10, 10, 12, 10, 12, 0, 2, 14, 0, 4, 4, 4, 6, 12, 12, 12, 10, 12, 14, 10, 10, 10},
		{10, 10, 10, 10, 10, 10, 14, 0, 2, 2, 2, 2, 4, 4, 4, 14, 12, 10, 12, 10, 12, 12, 10},
		{10, 10, 10, 0, 0, 0, 0, 2, 4, 6, 14, 12, 12, 12, 12, 10, 12},
		{10, 10, 10, 10, 10, 14, 0, 2, 0, 2, 4, 4, 4, 10, 14, 12, 12, 10, 12, 12, 12, 10, 14},		//10th data

		{1, 0, 14, 13, 12, 11, 8, 10, 6, 7, 1, 14, 14, 12, 11, 9, 8, 9, 7, 6, 7},			//Digit3 data as -ve data
		{2, 0, 15, 14, 12, 12, 11, 10, 9, 8, 1, 15, 12, 11, 9, 8, 7, 7 },
		{1, 0, 0, 14, 12, 11, 12, 10, 9, 8, 0, 15, 13, 12, 11, 9, 8, 9, 7, 7},
		{0, 15, 14, 13, 12, 11, 10, 9, 8, 8, 1, 15, 14, 12, 12, 10, 9, 8, 7, 7, 8},
		{0, 0, 15, 14, 12, 12, 11, 11, 8, 9, 6, 0, 0, 14, 12, 12, 9, 9, 8, 7, 7},
		{0, 0, 14, 13, 13, 11, 10, 9, 10, 7, 0, 14, 12, 11, 10, 9, 8, 8, 7},
		{0, 0, 14, 13, 13, 12, 11, 10, 8, 8, 7, 0, 15, 13, 0, 12, 12, 10, 10, 8, 8, 8, 6},
		{1, 0, 15, 14, 13, 12, 11, 10, 10, 7, 8, 7, 0, 0, 14, 13, 12, 10, 10, 8, 8, 8, 7},
		{0, 0, 13, 13, 12, 11, 11, 10, 8, 8, 1, 0, 14, 12, 12, 12, 10, 9, 8, 7, 7 },
		{0, 0, 15, 13, 13, 11, 11, 9, 8, 1, 14, 13, 12, 10, 9, 8, 8, 7 },		
		{2, 15, 14, 13, 12, 11, 10, 9, 8, 5, 1, 0, 15, 14, 12, 11, 10, 9, 8, 8 },
		{1, 14, 14, 13, 12, 11, 10, 9, 8, 0, 14, 0, 13, 10, 10, 9, 8, 7, 8},
		{15, 0, 15, 13, 12, 11, 10, 9, 8, 7, 1, 0, 0, 14, 12, 11, 10, 11, 9, 8, 8, 7, 8},
		{1, 1, 0, 15, 13, 12, 11, 11, 10, 9, 8, 1, 15, 12, 11, 10, 9, 9, 7, 8 },
		{0, 0, 0, 15, 14, 12, 12, 11, 10, 8, 8, 8, 5, 15, 0, 0, 13, 12, 11, 11, 10, 11, 7, 7, 7},
		{0, 0, 14, 13, 12, 11, 11, 9, 9, 8, 1, 15, 14, 12, 11, 9, 9, 8, 8, 7, 7 },
		{0, 4, 15, 0, 0, 15, 13, 13, 12, 11, 11, 9, 9, 7, 1, 15, 14, 12, 12, 11, 9, 8, 8, 8, 8, 7},
		{15, 1, 0, 15, 13, 12, 11, 11, 10, 9, 7, 0, 0, 14, 12, 11, 12, 10, 9, 8, 8, 7, 7},
		{1, 1, 1, 15, 15, 13, 12, 12, 11, 11, 10, 11, 8, 7, 7, 1, 14, 14, 12, 11, 11, 9, 9, 8, 8, 7, 6},
		{1, 0, 15, 15, 13, 11, 10, 10, 8, 0, 15, 14, 12, 11, 10, 9, 8, 7, 7},		//20th +ve 3 data end
		{15, 14, 12, 11, 10, 9, 8, 1, 15, 13, 12, 10, 11, 10, 7, 8, 7, 9, 7, 5},
		{0, 14, 13, 12, 10, 10, 8, 15, 14, 12, 10, 9, 7, 8},
		{0, 0, 14, 13, 11, 11, 10, 9, 9, 3, 15, 14, 13, 11, 10, 10, 7, 8, 7, 7 },
		{1, 0, 13, 12, 12, 10, 9, 0, 14, 12, 12, 10, 10, 8, 8, 7, 7 },
		{2, 0, 15, 14, 13, 12, 11, 11, 9, 8, 0, 14, 12, 12, 10, 10, 8, 7, 7, 8, 6},
		{0, 15, 13, 12, 11, 9, 7, 15, 14, 13, 12, 10, 10, 10, 8, 7},
		{0, 0, 14, 12, 11, 10, 8, 8, 0, 15, 15, 12, 12, 10, 10, 9, 8, 7, 5 },
		{0, 0, 14, 12, 10, 9, 9, 15, 15, 13, 11, 10, 10, 8, 6, 7},
		{ 0, 15, 14, 12, 11, 10, 9, 8, 15, 15, 14, 12, 12, 12, 10, 10, 8, 8, 6 },
		{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4},
		{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4 },
		{ 0, 15, 15, 13, 12, 10, 8, 15, 15, 13, 12, 10, 10, 8, 8, 8, 6 },
		{0, 15, 14, 13, 11, 10, 8, 8, 15, 14, 14, 12, 12, 10, 9, 8, 7, 7},
		{0, 0, 15, 13, 12, 11, 10, 9, 7, 0, 14, 14, 13, 12, 10, 8, 9, 8, 7, 7},
		{0, 0, 14, 12, 10, 9, 8, 14, 14, 13, 12, 10, 9, 8, 7, 5},
		{0, 15, 15, 12, 11, 10, 9, 14, 14, 13, 11, 10, 8, 7, 7},
		{0, 15, 14, 12, 10, 9, 9, 15, 14, 12, 12, 12, 9, 9, 7},
		{1, 0, 14, 13, 12, 10, 9, 0, 14, 12, 10, 10, 8, 8, 8, 8, 6, 5, 4 },
		{0, 14, 14, 12, 12, 10, 8, 15, 14, 12, 10, 9, 9, 7, 5, 6, 4},
		{0, 14, 14, 12, 12, 10, 8, 0, 14, 12, 9, 8, 8, 7, 8},
		{0, 15, 13, 11, 9, 15, 15, 14, 12, 11, 10, 10, 8, 7, 7 },		//41th +ve 3 data end
		{15, 13, 11, 9, 9, 8, 0, 15, 14, 14, 13, 10, 9, 8, 8, 7, 4 },
		{0, 15, 13, 12, 11, 9, 9, 8, 0, 15, 13, 12, 11, 9, 9, 8},
		{0, 15, 14, 13, 12, 11, 10, 9, 8, 0, 14, 15, 12, 11, 11, 10, 9, 8, 8},
		{0, 15, 14, 13, 12, 11, 9, 8, 8, 15, 15, 14, 12, 11, 10, 9, 8, 7 },
		{15, 14, 15, 14, 13, 12, 10, 8, 8, 0, 15, 13, 13, 11, 11, 10, 8, 8},
		{15, 14, 13, 13, 11, 9, 8, 7, 15, 0, 15, 14, 13, 11, 11, 10, 8, 7, 8, 5 },
		{0, 0, 15, 14, 13, 12, 11, 10, 8, 8, 7, 1, 15, 14, 13, 12, 12, 11, 10, 8, 8, 8 },
		{0, 15, 13, 12, 10, 10, 8, 15, 13, 12, 11, 11, 10, 9, 7},
		{0, 15, 15, 14, 12, 11, 10, 9, 14, 15, 13, 12, 11, 10, 9, 8, 7, 7 },
		{2, 0, 14, 13, 13, 12, 10, 8, 10, 5, 0, 15, 14, 13, 12, 10, 10, 9, 7, 7 },		//51th +ve 3 data end
		{0, 15, 14, 12, 11, 10, 10, 8, 0, 15, 13, 12, 11, 10, 9, 7 },
		{0, 0, 15, 0, 13, 12, 11, 11, 9, 8, 15, 15, 13, 12, 11, 11, 10, 9, 8, 7, 7, 5 },
		{1, 0, 0, 13, 13, 12, 11, 10, 9, 14, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6},			
		{0, 0, 13, 11, 9, 15, 13, 11, 10, 8, 8},
		{15, 15, 12, 11, 9, 9, 0, 15, 13, 11, 10, 9, 8},
		{15, 13, 12, 10, 9, 15, 14, 12, 11, 10, 9, 8 },	
		{0, 13, 12, 11, 10, 15, 12, 11, 11, 8, 7, 7, 5 },
		{15, 15, 13, 12, 11, 11, 9, 0, 14, 12, 11, 11, 10, 8, 7, 6},
		{1, 15, 0, 13, 12, 11, 10, 13, 13, 12, 11, 10, 8, 6, 5},
		{0, 15, 12, 11, 11, 10, 14, 12, 12, 11, 10, 8, 7, 5},
		{0, 15, 13, 11, 10, 13, 13, 12, 11, 12, 10, 9, 6, 5 },
		{0, 15, 12, 11, 9, 15, 13, 12, 11, 10, 9, 7, 5, 4},
		{1, 15, 13, 11, 10, 8, 15, 14, 12, 12, 10, 8, 7, 6},
		{2, 1, 0, 15, 12, 12, 11, 10, 11, 14, 13, 11, 11, 9, 8, 7, 6 },
		{2, 1, 0, 14, 12, 11, 11, 10, 15, 14, 12, 11, 10, 9, 7, 7, 6, 4},
		{0, 15, 12, 11, 10, 14, 13, 13, 10, 8, 8, 7, 5},
		{0, 15, 13, 12, 10, 9, 14, 13, 12, 10, 10, 9, 6, 6, 4},
		{2, 1, 0, 15, 14, 12, 12, 11, 11, 11, 11, 9, 9, 14, 13, 12, 12, 11, 11, 10, 8, 8, 7 },
		{1, 0, 0, 15, 15, 14, 12, 11, 10, 9, 13, 14, 13, 12, 12, 11, 11, 11, 10, 8, 7, 8},
		{2, 0, 0, 0, 15, 12, 11, 11, 11, 10, 10, 9, 15, 15, 13, 12, 12, 11, 11, 10, 7, 8, 7, 6},
		{0, 3, 2, 15, 15, 13, 12, 11, 11, 10, 9, 9, 15, 15, 14, 13, 12, 11, 11, 10, 8, 7, 7, 8, 4},
		{2, 0, 0, 12, 13, 12, 12, 11, 10, 10, 9, 9, 15, 14, 13, 13, 11, 10, 8, 8, 7, 5},
		{15, 12, 11, 10, 12, 15, 13, 12, 12, 10, 9, 7, 8},		//74th -ve 3 data end

		  

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit4M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit4);

}

void GestureRecognizerModified::Digit5TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit5M][50] = {	

		//data for digit 5 : not added yet

		{8, 8, 8, 8, 10, 12, 10, 12, 2, 14, 2, 14, 14, 14, 10, 10, 10, 8, 10, 6, 6, 6},
		{8, 10, 6, 10, 12, 12, 14, 2, 14, 2, 14, 10, 10, 10, 6, 6, 6},
		{6, 10, 8, 6, 10, 12, 12, 14, 2, 2, 14, 14, 14, 12, 10, 10, 10, 6, 8, 6, 6},
		{8, 8, 6, 8, 14, 10, 12, 14, 0, 14, 12, 14, 10, 10, 10, 6, 8, 6, 6, 6},
		{6, 8, 6, 10, 12, 12, 14, 2, 0, 14, 14, 14, 10, 10, 10, 8, 8, 8, 6},
		{8, 8, 8, 10, 12, 10, 14, 2, 2, 14, 14, 14, 10, 10, 10, 8, 8, 6, 6},
		{8, 6, 8, 10, 12, 12, 12, 0, 2, 0, 14, 14, 12, 10, 10, 10, 10, 6, 6, 6},
		{8, 10, 6, 10, 10, 12, 14, 14, 14, 14, 14, 10, 10, 8, 10, 6, 6},
		{10, 6, 8, 10, 10, 14, 10, 14, 14, 0, 14, 14, 10, 10, 10, 10, 6, 6, 6, 6},
		 
		  //data for digit 6 : not added yet
		{8, 9, 11, 11, 12, 12, 13, 14, 2, 1, 4, 5, 8, 9, 11},
		{8, 8, 10, 10, 11, 11, 12, 13, 15, 3, 3, 4, 7, 9},
		{7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 2, 2, 4, 4, 7, 9, 10},
		{8, 8, 8, 10, 11, 11, 12, 13, 13, 15, 2, 4, 4, 4, 6, 8, 9, 11},
		{8, 8, 8, 9, 11, 11, 12, 13, 14, 0, 1, 2, 4, 5, 7, 8, 10, 11},
		{8, 9, 9, 10, 12, 12, 13, 14, 0, 1, 3, 4, 5, 8, 9, 11, 11},
		{8, 8, 9, 10, 11, 12, 13, 14, 0, 1, 4, 4, 5, 8, 10, 11, 11},
		{8, 8, 9, 10, 12, 11, 12, 12, 12, 12, 13, 13, 0, 3, 3, 4, 5, 8, 10, 11 },
		{8, 8, 9, 9, 11, 11, 12, 12, 13, 14, 15, 1, 3, 4, 4, 6, 9, 10, 11},
		{8, 8, 9, 11, 12, 12, 12, 13, 14, 1, 2, 4, 4, 6, 9, 10},
		{9, 10, 11, 12, 13, 14, 1, 3, 5, 7, 10, 11},
		{8, 9, 10, 10, 11, 12, 13, 14, 1, 0, 3, 4, 4, 6, 8, 9, 11, 11},
		{7, 9, 9, 9, 11, 11, 12, 12, 12, 14, 0, 1, 2, 4, 5, 7, 9, 10},
		{8, 8, 10, 10, 11, 12, 12, 13, 14, 0, 2, 3, 4, 6, 9, 10},
		{8, 9, 9, 9, 11, 11, 11, 12, 12, 13, 13, 14, 0, 0, 0, 3, 3, 4, 5, 7, 8, 10, 11 },
		{8, 9, 9, 10, 10, 12, 12, 13, 13, 0, 1, 2, 3, 4, 4, 6, 8, 9, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 14, 14, 0, 0, 1, 3, 4, 4, 6, 8, 9, 11, 11, 11},
		{7, 8, 10, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 1, 2, 3, 3, 4, 5, 8, 9, 10, 10, 11, 11},
		{8, 8, 9, 9, 11, 11, 11, 12, 12, 12, 13, 13, 15, 15, 1, 1, 4, 4, 5, 6, 7, 9, 11, 11, 11},
		{8, 8, 9, 9, 10, 10, 10, 12, 11, 12, 13, 13, 13, 15, 0, 1, 2, 3, 4, 5, 6, 9, 10, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{8, 8, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 13, 14, 14, 0, 1, 1, 3, 3, 4, 5, 6, 8, 8, 10, 11, 11},
		{7, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 1, 1, 2, 4, 5, 7, 9, 10, 11, 11, 11},
		{7, 8, 8, 9, 10, 9, 10, 11, 11, 11, 12, 13, 12, 13, 15, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 10, 12, 12, 12, 13, 13, 15, 15, 0, 1, 1, 2, 3, 4, 5, 7, 8, 8, 9, 11, 11, 11},
		{8, 8, 10, 10, 11, 11, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 8, 9, 11, 11},
		{8, 8, 8, 9, 10, 11, 11, 11, 12, 13, 12, 13, 14, 15, 1, 1, 3, 4, 4, 5, 6, 8, 9, 9, 11, 11},
		{ 8, 9, 9, 9, 10, 11, 12, 12, 12, 13, 14, 15, 0, 1, 2, 4, 5, 7, 8, 9, 10, 12},
		{8, 8, 8, 10, 10, 11, 12, 12, 12, 13, 14, 15, 0, 2, 3, 4, 5, 8, 9, 10, 11},
		{8, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 0, 2, 3, 4, 6, 8, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{9, 10, 11, 12, 14, 14, 1, 4, 7, 9},
		{8, 10, 11, 12, 13, 15, 2, 3, 4, 8, 10 },
		{8, 9, 9, 11, 11, 12, 12, 13, 15, 1, 2, 4, 5, 7, 8, 10},
		{8, 9, 10, 11, 11, 12, 14, 0, 2, 3, 4, 6, 8, 10},
		{8, 9, 10, 11, 11, 12, 12, 15, 0, 2, 3, 4, 8, 9, 11},
		{8, 9, 10, 11, 12, 12, 13, 14, 2, 3, 4, 7, 9, 10},
		{8, 9, 10, 11, 11, 12, 12, 13, 0, 1, 2, 4, 6, 8, 10},
		{8, 9, 10, 10, 12, 11, 12, 13, 15, 1, 3, 3, 4, 5, 8, 9, 10},
		{8, 9, 10, 10, 11, 12, 13, 14, 0, 2, 3, 6, 8, 9},
		{8, 9, 9, 10, 11, 12, 12, 14, 0, 2, 3, 4, 6, 9},		//41st data end +ve
		{8, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 15, 2, 3, 4, 4, 4, 6, 8, 10, 11},
		{8, 9, 9, 10, 11, 12, 11, 12, 12, 12, 13, 14, 14, 1, 2, 3, 4, 5, 5, 7, 8, 10, 10},
		{7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 4, 5, 6, 8, 9, 10},
		{7, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 13, 14, 0, 2, 3, 4, 5, 6, 9},
		{8, 8, 9, 10, 10, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 5, 7, 9, 10, 12},
		{8, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 15, 15, 1, 2, 4, 5, 5, 8, 9, 10},
		{8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 15, 2, 3, 4, 6, 8, 11},
		{7, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 9, 10},
		{8, 8, 9, 10, 10, 10, 11, 10, 11, 12, 13, 13, 14, 14, 1, 2, 4, 4, 5, 7, 8, 10},
		{7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 12, 12, 13, 15, 0, 1, 3, 4, 4, 6, 8, 9, 10},
		{8, 8, 8, 10, 9, 10, 11, 11, 12, 12, 12, 13, 14, 15, 0, 1, 3, 4, 5, 6, 8, 9},		//52nd data end..

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit5M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit5);

}

void GestureRecognizerModified::Digit6TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit6M][50] = {	

		//data for digit 6 : not added yet
		{10, 10, 10, 10, 12, 14, 14, 14, 14, 14, 2, 2, 2, 2, 6, 6, 8, 8, 10, 10},
		{8, 10, 10, 10, 10, 12, 10, 14, 14, 14, 0, 2, 2, 2, 6, 6, 8, 10, 10},
		{8, 10, 10, 10, 12, 10, 14, 14, 0, 2, 2, 4, 6, 6, 10, 10, 12},
		{6, 8, 10, 10, 10, 10, 12, 10, 14, 14, 14, 14, 0, 2, 2, 2, 2, 6, 10, 10},
		{10, 10, 10, 10, 10, 12, 12, 14, 14, 0, 0, 2, 2, 4, 4, 6, 8, 10},
		{8, 10, 10, 10, 10, 10, 14, 14, 14, 2, 2, 2, 6, 6, 6, 10},
		{10, 10, 10, 10, 10, 12, 14, 14, 14, 2, 2, 2, 4, 6, 6, 10, 12},
		{8, 10, 10, 10, 10, 10, 12, 12, 14, 14, 2, 2, 2, 4, 4, 6, 10, 10},
		{8, 8, 10, 10, 12, 10, 10, 10, 12, 14, 14, 0, 2, 2, 4, 4, 6, 6, 8, 10, 12},
		{6, 10, 10, 10, 10, 10, 10, 14, 14, 14, 2, 2, 2, 4, 6, 4, 6, 10, 12},		//10th data
		{10, 8, 10, 10, 12, 12, 14, 14, 14, 0, 0, 2, 2, 4, 6, 6, 10},
		{10, 10, 10, 10, 12, 12, 12, 14, 14, 2, 2, 2, 4, 6, 6, 10, 10},
		{10, 10, 10, 10, 12, 12, 14, 14, 14, 2, 2, 2, 6, 6, 10},
		{8, 10, 10, 10, 10, 10, 12, 14, 14, 2, 2, 2, 4, 6, 6, 10, 10},
		{10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 2, 2, 2, 6, 6, 10},
		{8, 10, 10, 10, 10, 10, 12, 14, 14, 2, 2, 2, 2, 6, 10, 10},
		{10, 10, 10, 10, 10, 10, 12, 14, 14, 14, 2, 2, 4, 6, 6, 10},
		{6, 8, 10, 10, 10, 10, 14, 14, 14, 14, 2, 2, 2, 6, 6, 10, 10},
		{10, 10, 10, 10, 10, 12, 14, 14, 2, 2, 4, 6, 8, 12},
		{10, 10, 10, 10, 10, 12, 14, 14, 14, 2, 2, 2, 6, 6, 8, 10, 10},		//20th data
		{10, 10, 10, 10, 10, 12, 14, 14, 14, 2, 2, 2, 2, 4, 6, 6, 10},
		{6, 10, 10, 10, 10, 10, 12, 14, 14, 14, 14, 2, 2, 2, 2, 6, 6, 6, 10, 10},
		{6, 10, 10, 10, 10, 10, 12, 10, 14, 14, 14, 2, 2, 2, 4, 6, 10},
		{10, 10, 10, 10, 12, 14, 14, 14, 2, 2, 2, 2, 4, 6, 8, 10, 10},
		{8, 10, 10, 10, 10, 14, 14, 14, 14, 14, 2, 2, 4, 6, 10, 10},
		{6, 10, 10, 10, 10, 10, 14, 12, 14, 14, 14, 2, 2, 4, 6, 6, 10, 12},
		{10, 10, 10, 10, 10, 10, 14, 12, 14, 14, 14, 2, 4, 2, 6, 6, 10},
		{8, 10, 10, 10, 12, 12, 14, 14, 14, 2, 2, 2, 6, 6, 8, 10},			
		{10, 10, 10, 10, 14, 14, 14, 0, 2, 4, 6, 6, 10},
		{6, 10, 10, 10, 12, 14, 14, 14, 2, 2, 4, 6, 6, 10, 14},
		{10, 10, 10, 10, 10, 10, 14, 14, 14, 2, 2, 2, 6, 6, 10},		//31th data
		{8, 10, 10, 12, 12, 10, 14, 14, 0, 2, 4, 6, 6, 6, 10},
		{10, 10, 10, 10, 10, 14, 14, 14, 2, 2, 2, 6, 6, 10},
		{10, 10, 10, 12, 12, 14, 14, 14, 2, 2, 4, 6, 6, 10},
		{10, 10, 10, 12, 10, 14, 14, 14, 2, 2, 4, 4, 6, 6, 8, 10},		//35th data
		 
		{7, 8, 9, 11, 12, 13, 0, 0, 0, 14, 12, 11, 10, 8, 8, 8, 6},			//+ve data for later S
		{7, 8, 9, 10, 12, 12, 15, 15, 0, 15, 13, 12, 12, 10, 9, 8, 7, 7, 6},
		{6, 7, 8, 10, 11, 13, 15, 0, 15, 13, 11, 10, 9, 8, 8, 7},
		{6, 7, 7, 8, 8, 10, 11, 11, 13, 15, 14, 0, 0, 0, 13, 12, 10, 8, 9, 8, 7, 8, 8},
		{7, 8, 9, 9, 11, 12, 13, 14, 0, 15, 15, 13, 11, 10, 9, 8, 8, 6},
		{7, 8, 8, 9, 10, 12, 14, 14, 15, 15, 12, 11, 9, 8, 8, 8, 7},
		{8, 9, 10, 12, 14, 14, 14, 12, 12, 10, 8, 8, 8, 7 },
		{6, 6, 7, 9, 10, 11, 12, 13, 14, 15, 14, 15, 15, 12, 12, 10, 9, 8, 7, 8, 7},
		{8, 9, 10, 11, 13, 14, 0, 13, 15, 14, 11, 10, 8, 8, 8, 6},
		{8, 8, 9, 11, 12, 14, 15, 14, 13, 14, 12, 10, 8, 8, 8, 7},
		{8, 8, 10, 11, 12, 14, 15, 0, 15, 15, 12, 10, 8, 9, 8, 8, 6 },
		{6, 8, 8, 10, 11, 12, 14, 15, 15, 15, 14, 13, 11, 9, 8, 7, 8, 7, 7 },
		{6, 7, 8, 9, 9, 10, 12, 13, 13, 0, 14, 15, 15, 15, 12, 12, 8, 9, 8, 7, 8, 6},
		{7, 7, 9, 9, 11, 11, 12, 13, 15, 15, 14, 14, 13, 12, 10, 9, 7, 8, 7, 7, 7},
		{8, 8, 8, 10, 12, 13, 15, 15, 15, 14, 13, 12, 11, 9, 8, 8, 7, 7},
		{7, 8, 8, 9, 10, 12, 14, 14, 14, 14, 12, 12, 11, 9, 8, 7, 7 },
		{7, 7, 8, 8, 10, 11, 12, 14, 15, 0, 15, 15, 13, 13, 11, 10, 8, 8, 7, 7, 6, 5},
		{8, 8, 8, 10, 11, 11, 13, 15, 0, 15, 14, 12, 11, 9, 8, 8, 7, 7 },
		{6, 8, 8, 8, 10, 11, 12, 13, 15, 15, 15, 15, 13, 12, 11, 9, 8, 7, 7, 7, 6},
		{8, 7, 8, 8, 10, 11, 13, 14, 15, 15, 15, 15, 12, 11, 10, 7, 8, 7, 7, 6},
		{8, 7, 8, 8, 10, 11, 13, 14, 15, 15, 15, 15, 12, 11, 10, 7, 8, 7, 7, 6},
		{8, 10, 12, 15, 15, 14, 12, 10, 8, 8, 7, 7},
		{9, 9, 10, 12, 14, 15, 14, 13, 11, 9, 8, 7, 8, 6},
		{7, 8, 8, 10, 12, 14, 14, 15, 15, 13, 13, 10, 8, 8, 8, 6, 6},
		{7, 9, 9, 10, 12, 15, 15, 15, 15, 14, 13, 11, 9, 8, 7, 8, 6, 6},
		{8, 9, 9, 10, 11, 13, 15, 15, 15, 14, 13, 12, 11, 9, 8, 8, 8, 6, 5, 4},
		{8, 9, 11, 12, 14, 14, 15, 13, 13, 10, 8, 9, 8, 6, 7},
		{8, 8, 9, 9, 10, 12, 13, 15, 14, 14, 13, 12, 11, 9, 8, 8, 7, 7},
		{7, 8, 9, 9, 11, 13, 14, 14, 15, 13, 12, 11, 10, 7, 9, 7, 7, 5},
		{8, 8, 10, 11, 12, 14, 15, 15, 14, 13, 11, 10, 7, 8, 8, 7, 7},
		{8, 8, 9, 10, 12, 14, 15, 0, 15, 13, 12, 11, 8, 8, 9, 8, 6},		
		{8, 7, 9, 10, 11, 14, 15, 0, 15, 13, 12, 12, 10, 10, 8, 8, 7, 5, 4 },		//32st +ve data end


		  

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit6M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit6);

}

void GestureRecognizerModified::Digit7TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit7M][50] = {	

		//data for digit 7 : not added yet
		{0, 0, 0, 0, 11, 11, 11, 11, 10, 11, 10, 11, 12, 11},
		{0, 0, 0, 10, 11, 10, 10, 11, 11, 12, 11},
		{1, 0, 0, 0, 10, 11, 10, 10, 11, 10, 11, 11},
		{1, 0, 0, 10, 11, 11, 11, 11},
		{15, 0, 15, 9, 11, 11, 11, 11},
		{0, 0, 0, 0, 10, 10, 11, 10, 11, 10, 10},
		{0, 0, 0, 0, 10, 11, 10, 11, 10, 11, 11, 10, 11},
		{0, 0, 0, 0, 15, 0, 9, 11, 10, 10, 11, 10, 10},
		{0, 0, 0, 10, 10, 10, 10, 10, 11, 10, 11},
		{0, 0, 0, 12, 11, 11, 10, 11, 10, 11, 10},
		{0, 0, 0, 12, 11, 11, 10, 11, 10, 11, 10},
		{0, 0, 0, 12, 11, 10, 10, 10, 10, 10, 11},
		{0, 0, 1, 0, 13, 10, 11, 10, 10, 10, 10, 11},
		{1, 1, 1, 13, 8, 10, 10, 10, 11, 11, 10, 10},
		{0, 0, 0, 0, 10, 10, 10, 11, 10, 10, 10, 11},
		{0, 0, 0, 11, 11, 10, 10, 10, 11, 11, 11},
		{0, 0, 0, 0, 0, 0, 0, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10},
		{0, 0, 0, 10, 11, 10, 11, 10, 10, 10, 10, 11},
		{0, 0, 0, 13, 11, 11, 10, 11, 10, 11, 11},
		{15, 0, 0, 0, 10, 10, 10, 10, 11, 11, 10},
		{0, 1, 0, 0, 10, 11, 10, 11, 10, 10, 11, 11},
		{0, 0, 15, 0, 10, 10, 11, 10, 10, 11},
		{0, 15, 0, 0, 10, 10, 10, 10, 10, 10, 11, 11},
		{0, 0, 0, 0, 11, 10, 10, 11, 10, 10, 11, 10},
		{0, 0, 0, 11, 10, 11, 10, 10, 10, 10},
		{15, 0, 15, 10, 11, 11, 11, 10, 11},
		{0, 0, 0, 0, 11, 10, 10, 11, 10, 10, 11},
		{0, 0, 0, 0, 0, 11, 11, 11, 10, 10, 10, 10, 10, 11},
		{0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10},
		{0, 0, 0, 11, 10, 10, 11, 10, 11, 11},
		{0, 0, 0, 10, 11, 10, 11, 10, 11},		
		{15, 15, 15, 0, 0, 10, 10, 11, 11, 10, 10, 10, 11, 11, 10},
		{0, 0, 10, 11, 11, 11, 10 },
		{0, 0, 10, 10, 10, 11, 10},
		{0, 0, 10, 11, 10, 11, 10},
		{15, 0, 0, 0, 15, 0, 15, 0, 15, 0, 0, 15, 10, 10, 9, 10, 10, 10, 10, 10, 9, 9, 9, 10, 9},
		{0, 0, 11, 10, 10, 10, 10 },
		{0, 0, 11, 10, 11, 11, 11},
		{0, 12, 10, 10, 10},
		{0, 0, 10, 10, 10, 10, 10},
		{0, 0, 11, 10, 10, 10, 10},
		{0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 12},		
		{0, 0, 12, 10, 11, 10, 11},
		{1, 0, 15, 10, 10, 11, 10, 11},
		{1, 0, 12, 11, 10, 11},
		{1, 0, 12, 10, 10, 10, 11 },
		{0, 0, 12, 10, 10, 11, 10, 11},
		{0, 0, 11, 10, 10, 10, 10},
		{0, 0, 12, 11, 11, 12},
		{0, 15, 12, 10, 10, 11, 10, 10, 10},
		{0, 0, 11, 10, 10, 10, 10},
		{0, 0, 11, 10, 10, 11, 10, 10},
		{15, 15, 11, 10, 10, 11, 10, 10},
		{1, 1, 11, 10, 10, 11, 10, 10},
		{15, 1, 11, 10, 10, 11, 10, 10},
		{1, 0, 12, 11, 11, 12},
		{1, 1, 12, 11, 11, 12},
		{15, 15, 12, 11, 11, 12},
		{15, 0, 12, 10, 10, 10, 11 },
		{15, 15, 12, 10, 10, 10, 11 },
		{0, 0, 12, 10, 10, 10, 11 },
		{1, 1, 12, 10, 10, 10, 11 },
		{1, 0, 12, 10, 9, 10, 11 },
		{1, 0, 12, 10, 10, 10, 9 },
		{1, 0, 12, 10, 10, 10, 11, 12 },	
		{1, 0, 12, 10, 10, 9, 10, 11, 12 },	//66nd data end.


		  //data for digit 6 : not added yet
		{8, 9, 11, 11, 12, 12, 13, 14, 2, 1, 4, 5, 8, 9, 11},
		{8, 8, 10, 10, 11, 11, 12, 13, 15, 3, 3, 4, 7, 9},
		{7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 2, 2, 4, 4, 7, 9, 10},
		{8, 8, 8, 10, 11, 11, 12, 13, 13, 15, 2, 4, 4, 4, 6, 8, 9, 11},
		{8, 8, 8, 9, 11, 11, 12, 13, 14, 0, 1, 2, 4, 5, 7, 8, 10, 11},
		{8, 9, 9, 10, 12, 12, 13, 14, 0, 1, 3, 4, 5, 8, 9, 11, 11},
		{8, 8, 9, 10, 11, 12, 13, 14, 0, 1, 4, 4, 5, 8, 10, 11, 11},
		{8, 8, 9, 10, 12, 11, 12, 12, 12, 12, 13, 13, 0, 3, 3, 4, 5, 8, 10, 11 },
		{8, 8, 9, 9, 11, 11, 12, 12, 13, 14, 15, 1, 3, 4, 4, 6, 9, 10, 11},
		{8, 8, 9, 11, 12, 12, 12, 13, 14, 1, 2, 4, 4, 6, 9, 10},
		{9, 10, 11, 12, 13, 14, 1, 3, 5, 7, 10, 11},
		{8, 9, 10, 10, 11, 12, 13, 14, 1, 0, 3, 4, 4, 6, 8, 9, 11, 11},
		{7, 9, 9, 9, 11, 11, 12, 12, 12, 14, 0, 1, 2, 4, 5, 7, 9, 10},
		{8, 8, 10, 10, 11, 12, 12, 13, 14, 0, 2, 3, 4, 6, 9, 10},
		{8, 9, 9, 9, 11, 11, 11, 12, 12, 13, 13, 14, 0, 0, 0, 3, 3, 4, 5, 7, 8, 10, 11 },
		{8, 9, 9, 10, 10, 12, 12, 13, 13, 0, 1, 2, 3, 4, 4, 6, 8, 9, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 14, 14, 0, 0, 1, 3, 4, 4, 6, 8, 9, 11, 11, 11},
		{7, 8, 10, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 1, 2, 3, 3, 4, 5, 8, 9, 10, 10, 11, 11},
		{8, 8, 9, 9, 11, 11, 11, 12, 12, 12, 13, 13, 15, 15, 1, 1, 4, 4, 5, 6, 7, 9, 11, 11, 11},
		{8, 8, 9, 9, 10, 10, 10, 12, 11, 12, 13, 13, 13, 15, 0, 1, 2, 3, 4, 5, 6, 9, 10, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{8, 8, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 13, 14, 14, 0, 1, 1, 3, 3, 4, 5, 6, 8, 8, 10, 11, 11},
		{7, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 1, 1, 2, 4, 5, 7, 9, 10, 11, 11, 11},
		{7, 8, 8, 9, 10, 9, 10, 11, 11, 11, 12, 13, 12, 13, 15, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 10, 12, 12, 12, 13, 13, 15, 15, 0, 1, 1, 2, 3, 4, 5, 7, 8, 8, 9, 11, 11, 11},
		{8, 8, 10, 10, 11, 11, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 8, 9, 11, 11},
		{8, 8, 8, 9, 10, 11, 11, 11, 12, 13, 12, 13, 14, 15, 1, 1, 3, 4, 4, 5, 6, 8, 9, 9, 11, 11},
		{ 8, 9, 9, 9, 10, 11, 12, 12, 12, 13, 14, 15, 0, 1, 2, 4, 5, 7, 8, 9, 10, 12},
		{8, 8, 8, 10, 10, 11, 12, 12, 12, 13, 14, 15, 0, 2, 3, 4, 5, 8, 9, 10, 11},
		{8, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 0, 2, 3, 4, 6, 8, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{9, 10, 11, 12, 14, 14, 1, 4, 7, 9},
		{8, 10, 11, 12, 13, 15, 2, 3, 4, 8, 10 },
		{8, 9, 9, 11, 11, 12, 12, 13, 15, 1, 2, 4, 5, 7, 8, 10},
		{8, 9, 10, 11, 11, 12, 14, 0, 2, 3, 4, 6, 8, 10},
		{8, 9, 10, 11, 11, 12, 12, 15, 0, 2, 3, 4, 8, 9, 11},
		{8, 9, 10, 11, 12, 12, 13, 14, 2, 3, 4, 7, 9, 10},
		{8, 9, 10, 11, 11, 12, 12, 13, 0, 1, 2, 4, 6, 8, 10},
		{8, 9, 10, 10, 12, 11, 12, 13, 15, 1, 3, 3, 4, 5, 8, 9, 10},
		{8, 9, 10, 10, 11, 12, 13, 14, 0, 2, 3, 6, 8, 9},
		{8, 9, 9, 10, 11, 12, 12, 14, 0, 2, 3, 4, 6, 9},		//41st data end +ve
		{8, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 15, 2, 3, 4, 4, 4, 6, 8, 10, 11},
		{8, 9, 9, 10, 11, 12, 11, 12, 12, 12, 13, 14, 14, 1, 2, 3, 4, 5, 5, 7, 8, 10, 10},
		{7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 4, 5, 6, 8, 9, 10},
		{7, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 13, 14, 0, 2, 3, 4, 5, 6, 9},
		{8, 8, 9, 10, 10, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 5, 7, 9, 10, 12},
		{8, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 15, 15, 1, 2, 4, 5, 5, 8, 9, 10},
		{8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 15, 2, 3, 4, 6, 8, 11},
		{7, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 9, 10},
		{8, 8, 9, 10, 10, 10, 11, 10, 11, 12, 13, 13, 14, 14, 1, 2, 4, 4, 5, 7, 8, 10},
		{7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 12, 12, 13, 15, 0, 1, 3, 4, 4, 6, 8, 9, 10},
		{8, 8, 8, 10, 9, 10, 11, 11, 12, 12, 12, 13, 14, 15, 0, 1, 3, 4, 5, 6, 8, 9},		//52nd data end..

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit7M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit7);

}

void GestureRecognizerModified::Digit8TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit8M][50] = {	

		//data for digit 8 : not added yet
		{8, 10, 10, 14, 14, 14, 14, 14, 12, 10, 10, 10, 6, 6, 4, 2, 2, 2, 2, 2, 4, 6},
		{6, 10, 10, 14, 14, 14, 14, 14, 12, 12, 10, 10, 8, 6, 6, 4, 4, 2, 2, 2, 2, 6, 6},
		{8, 10, 10, 14, 14, 14, 14, 14, 10, 6, 2, 2, 2, 2, 2, 4, 6, 6},
		{8, 10, 14, 14, 14, 14, 10, 10, 10, 6, 6, 4, 2, 2, 2, 2, 4, 6, 6},
		{10, 10, 14, 14, 14, 14, 14, 12, 10, 6, 2, 2, 2, 2, 2, 4, 6, 6},
		{6, 10, 10, 14, 14, 14, 14, 12, 10, 10, 8, 6, 6, 2, 2, 2, 2, 2, 2, 4, 6, 6},
		{10, 10, 14, 14, 14, 14, 14, 14, 10, 6, 2, 4, 2, 2, 2, 2, 4, 6},
		{10, 10, 14, 14, 14, 14, 12, 10, 10, 6, 6, 4, 2, 2, 2, 2, 2, 4, 6, 6},
		{10, 10, 12, 14, 14, 14, 12, 12, 10, 10, 6, 6, 4, 2, 2, 2, 2, 2, 2, 4, 6, 6},	//9th data

		//data for digit 5 : -ve data 
		{9, 8, 8, 8, 13, 11, 12, 11, 11, 1, 1, 15, 14, 13, 12, 12, 10, 9, 8, 8, 7, 6, 3},
		{8, 8, 8, 8, 11, 11, 11, 12, 1, 0, 0, 14, 13, 12, 12, 11, 11, 9, 7, 7, 6, 5},
		{8, 8, 7, 12, 12, 11, 12, 11, 0, 0, 15, 14, 13, 12, 12, 12, 9, 8, 7, 7, 7, 4, 4},
		{8, 8, 8, 8, 12, 11, 12, 11, 11, 0, 0, 0, 14, 13, 12, 12, 12, 11, 9, 8, 7, 8, 4, 5, 4},
		{8, 8, 8, 12, 12, 11, 15, 0, 0, 14, 13, 12, 12, 10, 9, 8, 7, 5},
		{7, 8, 8, 10, 12, 11, 11, 15, 0, 0, 14, 13, 12, 12, 11, 10, 8, 7, 7, 5},
		{8, 8, 8, 8, 8, 8, 11, 12, 11, 12, 11, 11, 0, 0, 15, 13, 14, 12, 12, 12, 10, 9, 8, 8, 6, 7, 4, 4},
		{9, 8, 7, 8, 11, 12, 11, 11, 11, 11, 0, 0, 15, 14, 13, 12, 12, 12, 10, 9, 8, 8, 7, 6, 4},
		{7, 8, 6, 8, 12, 11, 11, 11, 11, 1, 0, 0, 14, 14, 13, 12, 12, 11, 10, 9, 9, 8, 8, 5, 5, 4},
		{7, 7, 8, 7, 9, 10, 12, 11, 11, 12, 11, 0, 1, 15, 14, 13, 13, 12, 12, 12, 10, 8, 8, 8, 7, 5, 4, 4},
		{8, 8, 8, 10, 11, 12, 12, 15, 0, 0, 14, 13, 13, 12, 11, 9, 8, 8, 7, 5},
		{8, 8, 8, 8, 8, 12, 12, 12, 12, 11, 15, 0, 15, 14, 14, 13, 12, 12, 12, 11, 10, 8, 8, 8, 6, 5},
		{8, 7, 8, 7, 8, 11, 12, 12, 12, 11, 12, 15, 0, 14, 14, 12, 13, 12, 12, 12, 9, 8, 8, 8, 7, 8, 4},
		{8, 8, 8, 8, 8, 10, 12, 12, 12, 11, 11, 1, 1, 0, 14, 15, 13, 13, 12, 12, 12, 11, 10, 8, 8, 8, 8, 7, 6, 5, 4},
		{8, 8, 8, 10, 8, 12, 12, 12, 12, 12, 15, 15, 15, 13, 13, 12, 11, 10, 8, 8, 8, 8, 6, 5},
		{7, 8, 7, 8, 12, 12, 11, 11, 11, 1, 0, 1, 14, 13, 12, 12, 12, 10, 10, 9, 8, 7, 7, 4, 4},
		{7, 8, 8, 9, 11, 11, 12, 11, 11, 0, 0, 0, 14, 13, 13, 12, 12, 11, 11, 9, 8, 8, 7, 7, 4},
		{8, 8, 8, 11, 12, 12, 12, 0, 0, 15, 15, 14, 13, 12, 11, 11, 10, 8, 7, 7, 6, 5},
		{8, 8, 8, 8, 12, 12, 11, 12, 15, 0, 15, 14, 13, 13, 11, 10, 8, 8, 8, 7, 5, 4},
		{8, 7, 8, 8, 8, 11, 11, 11, 11, 0, 0, 15, 13, 13, 12, 12, 11, 10, 10, 8, 8, 7, 7},
		{7, 8, 8, 12, 12, 11, 11, 0, 0, 15, 15, 13, 11, 10, 9, 8, 8, 5},
		{8, 8, 8, 7, 10, 12, 12, 11, 11, 12, 1, 0, 0, 15, 15, 12, 12, 12, 11, 10, 9, 9, 8, 8, 7, 5 },
		{8, 8, 8, 8, 10, 12, 12, 11, 11, 1, 0, 0, 15, 14, 13, 12, 12, 11, 11, 10, 9, 7, 8, 6, 6},
		{8, 8, 7, 7, 12, 11, 11, 11, 15, 0, 0, 15, 14, 12, 12, 11, 10, 10, 9, 8, 7, 7, 5},
		{7, 8, 8, 7, 10, 12, 12, 12, 11, 15, 0, 0, 0, 13, 13, 11, 11, 10, 9, 8, 7, 8, 4, 4 },
		{9, 8, 8, 12, 12, 11, 12, 0, 15, 0, 13, 13, 12, 11, 9, 8, 8, 8, 7, 5},
		{8, 8, 7, 11, 12, 11, 1, 0, 15, 14, 12, 12, 11, 10, 10, 8, 8, 5},
		{8, 8, 8, 11, 12, 12, 11, 1, 0, 15, 13, 12, 11, 10, 9, 8, 7 },
		{8, 8, 7, 10, 12, 12, 12, 0, 0, 0, 14, 13, 12, 11, 10, 10, 9, 8, 7, 5},
		{8, 8, 7, 11, 12, 12, 11, 0, 0, 15, 14, 12, 12, 11, 10, 9, 7, 5},
		{8, 8, 7, 11, 12, 12, 11, 0, 0, 15, 14, 12, 12, 11, 10, 9, 7, 5},
		{8, 11, 12, 11, 0, 15, 13, 11, 11, 10, 8, 5, 4},
		{9, 7, 11, 12, 11, 12, 0, 0, 14, 12, 11, 10, 10, 7, 5, 4 },
		{8, 8, 7, 11, 12, 11, 11, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 5, 4, 4},
		{8, 8, 9, 8, 8, 10, 12, 11, 11, 14, 0, 15, 15, 14, 13, 12, 12, 11, 10, 9, 9, 7, 6, 5, 4},
		{7, 10, 7, 8, 10, 12, 12, 12, 0, 0, 15, 13, 12, 12, 12, 10, 10, 8, 8, 6},
		{8, 7, 8, 11, 11, 11, 15, 0, 15, 13, 12, 12, 11, 10, 9, 8, 7, 5 },
		{8, 8, 8, 8, 10, 12, 11, 12, 11, 0, 1, 15, 15, 13, 13, 12, 11, 11, 10, 9, 8, 7, 6, 4},
		{8, 8, 7, 8, 10, 11, 11, 12, 0, 15, 14, 13, 12, 12, 11, 10, 8, 7, 7, 4},
		{8, 7, 8, 10, 11, 11, 11, 15, 0, 0, 14, 13, 12, 12, 11, 11, 9, 8, 7, 5 },
		{8, 8, 7, 11, 11, 11, 15, 0, 14, 13, 12, 11, 11, 10, 9, 8, 6, 5},					
		{8, 8, 8, 12, 11, 15, 14, 1, 13, 12, 11, 10, 9, 7, 6},
		{8, 8, 8, 10, 12, 11, 12, 0, 0, 15, 14, 13, 12, 11, 11, 9, 8, 6, 5},
		{7, 8, 7, 8, 11, 12, 12, 12, 12, 12, 1, 0, 15, 15, 13, 12, 11, 11, 8, 8, 8, 7, 6, 4},
		{8, 7, 8, 10, 8, 8, 10, 12, 11, 11, 12, 12, 14, 0, 0, 15, 14, 13, 12, 12, 11, 9, 8, 8, 7, 8, 8, 4},
		{7, 8, 8, 7, 11, 11, 11, 12, 0, 0, 15, 0, 13, 12, 12, 11, 11, 10, 8, 8, 8, 6, 5},
		{8, 8, 8, 8, 7, 11, 12, 12, 11, 12, 1, 0, 0, 15, 14, 13, 14, 12, 11, 10, 9, 8, 8, 7, 8, 7, 5, 4 },
		{8, 8, 8, 8, 8, 8, 11, 11, 12, 12, 12, 12, 0, 1, 0, 15, 15, 14, 12, 12, 12, 12, 10, 8, 8, 7, 5},
		{8, 8, 8, 8, 11, 12, 12, 12, 0, 0, 15, 14, 12, 12, 12, 10, 9, 6, 7},
		{8, 7, 8, 12, 12, 12, 12, 0, 0, 15, 13, 12, 12, 10, 9, 8, 7, 4},
		{8, 8, 8, 8, 11, 12, 12, 12, 12, 1, 0, 0, 14, 12, 12, 12, 10, 9, 7, 7},
		{7, 8, 8, 7, 12, 11, 12, 0, 0, 0, 13, 12, 11, 9, 9, 7, 7},
		{8, 7, 8, 12, 11, 12, 0, 0, 0, 13, 12, 11, 10, 9, 6, 7, 4},
		{7, 8, 7, 12, 11, 11, 11, 0, 0, 0, 13, 12, 11, 10, 9, 8, 7, 5},
		{8, 7, 8, 10, 12, 11, 12, 0, 0, 15, 13, 12, 11, 10, 9, 8, 6 },
		{8, 8, 7, 8, 11, 11, 12, 12, 11, 1, 0, 15, 14, 13, 12, 12, 10, 10, 9, 8, 6, 5},
		{8, 7, 8, 10, 12, 11, 12, 15, 0, 15, 14, 13, 12, 11, 10, 9, 9, 6, 5, 5},
		{8, 8, 7, 8, 10, 11, 12, 11, 12, 0, 1, 0, 0, 14, 13, 12, 12, 11, 10, 9, 8, 6, 6},
		{7, 8, 8, 10, 11, 11, 15, 0, 1, 13, 0, 13, 12, 12, 12, 11, 11, 10, 8, 6, 6},
		{8, 8, 7, 11, 11, 12, 11, 0, 0, 0, 15, 14, 13, 12, 11, 11, 11, 9, 8, 8, 7, 5 },
		{7, 8, 8, 10, 11, 11, 0, 1, 0, 15, 13, 12, 12, 12, 11, 10, 9, 8, 8, 5, 4},		//61th data end

		 
		 
		  //data for digit 6 : not added yet
		{8, 9, 11, 11, 12, 12, 13, 14, 2, 1, 4, 5, 8, 9, 11},
		{8, 8, 10, 10, 11, 11, 12, 13, 15, 3, 3, 4, 7, 9},
		{7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 2, 2, 4, 4, 7, 9, 10},
		{8, 8, 8, 10, 11, 11, 12, 13, 13, 15, 2, 4, 4, 4, 6, 8, 9, 11},
		{8, 8, 8, 9, 11, 11, 12, 13, 14, 0, 1, 2, 4, 5, 7, 8, 10, 11},
		{8, 9, 9, 10, 12, 12, 13, 14, 0, 1, 3, 4, 5, 8, 9, 11, 11},
		{8, 8, 9, 10, 11, 12, 13, 14, 0, 1, 4, 4, 5, 8, 10, 11, 11},
		{8, 8, 9, 10, 12, 11, 12, 12, 12, 12, 13, 13, 0, 3, 3, 4, 5, 8, 10, 11 },
		{8, 8, 9, 9, 11, 11, 12, 12, 13, 14, 15, 1, 3, 4, 4, 6, 9, 10, 11},
		{8, 8, 9, 11, 12, 12, 12, 13, 14, 1, 2, 4, 4, 6, 9, 10},
		{9, 10, 11, 12, 13, 14, 1, 3, 5, 7, 10, 11},
		{8, 9, 10, 10, 11, 12, 13, 14, 1, 0, 3, 4, 4, 6, 8, 9, 11, 11},
		{7, 9, 9, 9, 11, 11, 12, 12, 12, 14, 0, 1, 2, 4, 5, 7, 9, 10},
		{8, 8, 10, 10, 11, 12, 12, 13, 14, 0, 2, 3, 4, 6, 9, 10},
		{8, 9, 9, 9, 11, 11, 11, 12, 12, 13, 13, 14, 0, 0, 0, 3, 3, 4, 5, 7, 8, 10, 11 },
		{8, 9, 9, 10, 10, 12, 12, 13, 13, 0, 1, 2, 3, 4, 4, 6, 8, 9, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 14, 14, 0, 0, 1, 3, 4, 4, 6, 8, 9, 11, 11, 11},
		{7, 8, 10, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 1, 2, 3, 3, 4, 5, 8, 9, 10, 10, 11, 11},
		{8, 8, 9, 9, 11, 11, 11, 12, 12, 12, 13, 13, 15, 15, 1, 1, 4, 4, 5, 6, 7, 9, 11, 11, 11},
		{8, 8, 9, 9, 10, 10, 10, 12, 11, 12, 13, 13, 13, 15, 0, 1, 2, 3, 4, 5, 6, 9, 10, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{8, 8, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 13, 14, 14, 0, 1, 1, 3, 3, 4, 5, 6, 8, 8, 10, 11, 11},
		{7, 8, 8, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 1, 1, 2, 4, 5, 7, 9, 10, 11, 11, 11},
		{7, 8, 8, 9, 10, 9, 10, 11, 11, 11, 12, 13, 12, 13, 15, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 11, 11},
		{8, 8, 8, 10, 10, 10, 10, 11, 10, 12, 12, 12, 13, 13, 15, 15, 0, 1, 1, 2, 3, 4, 5, 7, 8, 8, 9, 11, 11, 11},
		{8, 8, 10, 10, 11, 11, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 8, 9, 11, 11},
		{8, 8, 8, 9, 10, 11, 11, 11, 12, 13, 12, 13, 14, 15, 1, 1, 3, 4, 4, 5, 6, 8, 9, 9, 11, 11},
		{ 8, 9, 9, 9, 10, 11, 12, 12, 12, 13, 14, 15, 0, 1, 2, 4, 5, 7, 8, 9, 10, 12},
		{8, 8, 8, 10, 10, 11, 12, 12, 12, 13, 14, 15, 0, 2, 3, 4, 5, 8, 9, 10, 11},
		{8, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 0, 2, 3, 4, 6, 8, 10, 11},
		{8, 7, 8, 10, 9, 10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 15, 0, 0, 1, 3, 3, 4, 5, 7, 8, 9, 11, 11, 11},
		{9, 10, 11, 12, 14, 14, 1, 4, 7, 9},
		{8, 10, 11, 12, 13, 15, 2, 3, 4, 8, 10 },
		{8, 9, 9, 11, 11, 12, 12, 13, 15, 1, 2, 4, 5, 7, 8, 10},
		{8, 9, 10, 11, 11, 12, 14, 0, 2, 3, 4, 6, 8, 10},
		{8, 9, 10, 11, 11, 12, 12, 15, 0, 2, 3, 4, 8, 9, 11},
		{8, 9, 10, 11, 12, 12, 13, 14, 2, 3, 4, 7, 9, 10},
		{8, 9, 10, 11, 11, 12, 12, 13, 0, 1, 2, 4, 6, 8, 10},
		{8, 9, 10, 10, 12, 11, 12, 13, 15, 1, 3, 3, 4, 5, 8, 9, 10},
		{8, 9, 10, 10, 11, 12, 13, 14, 0, 2, 3, 6, 8, 9},
		{8, 9, 9, 10, 11, 12, 12, 14, 0, 2, 3, 4, 6, 9},		//41st data end +ve
		{8, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 15, 2, 3, 4, 4, 4, 6, 8, 10, 11},
		{8, 9, 9, 10, 11, 12, 11, 12, 12, 12, 13, 14, 14, 1, 2, 3, 4, 5, 5, 7, 8, 10, 10},
		{7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 4, 5, 6, 8, 9, 10},
		{7, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 13, 14, 0, 2, 3, 4, 5, 6, 9},
		{8, 8, 9, 10, 10, 11, 12, 12, 12, 13, 14, 15, 1, 3, 4, 5, 7, 9, 10, 12},
		{8, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 15, 15, 1, 2, 4, 5, 5, 8, 9, 10},
		{8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 15, 2, 3, 4, 6, 8, 11},
		{7, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 2, 2, 4, 5, 6, 9, 10},
		{8, 8, 9, 10, 10, 10, 11, 10, 11, 12, 13, 13, 14, 14, 1, 2, 4, 4, 5, 7, 8, 10},
		{7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 12, 12, 13, 15, 0, 1, 3, 4, 4, 6, 8, 9, 10},
		{8, 8, 8, 10, 9, 10, 11, 11, 12, 12, 12, 13, 14, 15, 0, 1, 3, 4, 5, 6, 8, 9},		//52nd data end..

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit8M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit8);

}

void GestureRecognizerModified::Digit9TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit9M][50] = {	

		//data for digit 9 : not added yet
		{6, 8, 10, 10, 10, 14, 14, 2, 2, 2, 4, 6, 12, 14, 12, 12, 12, 10, 12, 12, 12},
		{6, 10, 10, 10, 12, 14, 14, 2, 2, 2, 4, 10, 12, 10, 10, 10, 12, 12, 10},
		{6, 10, 10, 10, 12, 12, 14, 14, 2, 2, 4, 4, 4, 4, 12, 10, 12, 10, 12, 10, 12, 10},
		{8, 8, 10, 10, 10, 14, 14, 14, 2, 2, 4, 4, 4, 12, 12, 10, 10, 10, 12, 10},
		{6, 10, 10, 10, 12, 14, 14, 14, 2, 2, 4, 2, 4, 4, 10, 12, 10, 12, 12, 10, 12},
		{10, 10, 10, 10, 10, 12, 14, 2, 2, 2, 4, 4, 12, 10, 12, 12, 12, 12, 12, 12, 12},
		{8, 8, 10, 10, 10, 14, 14, 14, 2, 2, 4, 4, 6, 12, 12, 10, 12, 12, 10, 12, 12},
		{6, 10, 10, 10, 12, 12, 14, 14, 2, 2, 2, 2, 10, 10, 12, 10, 12, 10, 10, 12, 10},
		{8, 6, 10, 10, 12, 12, 14, 0, 2, 2, 2, 4, 10, 12, 12, 12, 10, 12, 12},
		{8, 10, 12, 10, 14, 14, 0, 2, 2, 4, 4, 14, 10, 12, 12, 10, 10, 12},		//10th data
		{6, 10, 10, 14, 14, 14, 2, 2, 4, 14, 12, 12, 12, 10, 10, 12},
		{8, 10, 10, 14, 14, 2, 2, 4, 4, 4, 14, 10, 12, 12, 12, 12, 10, 10},
		{10, 10, 10, 14, 14, 14, 2, 2, 6, 4, 12, 10, 12, 10, 12, 12, 10},		//13th data

		{1, 0, 14, 13, 12, 11, 8, 10, 6, 7, 1, 14, 14, 12, 11, 9, 8, 9, 7, 6, 7},			//Digit3 data as -ve data
		{2, 0, 15, 14, 12, 12, 11, 10, 9, 8, 1, 15, 12, 11, 9, 8, 7, 7 },
		{1, 0, 0, 14, 12, 11, 12, 10, 9, 8, 0, 15, 13, 12, 11, 9, 8, 9, 7, 7},
		{0, 15, 14, 13, 12, 11, 10, 9, 8, 8, 1, 15, 14, 12, 12, 10, 9, 8, 7, 7, 8},
		{0, 0, 15, 14, 12, 12, 11, 11, 8, 9, 6, 0, 0, 14, 12, 12, 9, 9, 8, 7, 7},
		{0, 0, 14, 13, 13, 11, 10, 9, 10, 7, 0, 14, 12, 11, 10, 9, 8, 8, 7},
		{0, 0, 14, 13, 13, 12, 11, 10, 8, 8, 7, 0, 15, 13, 0, 12, 12, 10, 10, 8, 8, 8, 6},
		{1, 0, 15, 14, 13, 12, 11, 10, 10, 7, 8, 7, 0, 0, 14, 13, 12, 10, 10, 8, 8, 8, 7},
		{0, 0, 13, 13, 12, 11, 11, 10, 8, 8, 1, 0, 14, 12, 12, 12, 10, 9, 8, 7, 7 },
		{0, 0, 15, 13, 13, 11, 11, 9, 8, 1, 14, 13, 12, 10, 9, 8, 8, 7 },		
		{2, 15, 14, 13, 12, 11, 10, 9, 8, 5, 1, 0, 15, 14, 12, 11, 10, 9, 8, 8 },
		{1, 14, 14, 13, 12, 11, 10, 9, 8, 0, 14, 0, 13, 10, 10, 9, 8, 7, 8},
		{15, 0, 15, 13, 12, 11, 10, 9, 8, 7, 1, 0, 0, 14, 12, 11, 10, 11, 9, 8, 8, 7, 8},
		{1, 1, 0, 15, 13, 12, 11, 11, 10, 9, 8, 1, 15, 12, 11, 10, 9, 9, 7, 8 },
		{0, 0, 0, 15, 14, 12, 12, 11, 10, 8, 8, 8, 5, 15, 0, 0, 13, 12, 11, 11, 10, 11, 7, 7, 7},
		{0, 0, 14, 13, 12, 11, 11, 9, 9, 8, 1, 15, 14, 12, 11, 9, 9, 8, 8, 7, 7 },
		{0, 4, 15, 0, 0, 15, 13, 13, 12, 11, 11, 9, 9, 7, 1, 15, 14, 12, 12, 11, 9, 8, 8, 8, 8, 7},
		{15, 1, 0, 15, 13, 12, 11, 11, 10, 9, 7, 0, 0, 14, 12, 11, 12, 10, 9, 8, 8, 7, 7},
		{1, 1, 1, 15, 15, 13, 12, 12, 11, 11, 10, 11, 8, 7, 7, 1, 14, 14, 12, 11, 11, 9, 9, 8, 8, 7, 6},
		{1, 0, 15, 15, 13, 11, 10, 10, 8, 0, 15, 14, 12, 11, 10, 9, 8, 7, 7},		//20th +ve 3 data end
		{15, 14, 12, 11, 10, 9, 8, 1, 15, 13, 12, 10, 11, 10, 7, 8, 7, 9, 7, 5},
		{0, 14, 13, 12, 10, 10, 8, 15, 14, 12, 10, 9, 7, 8},
		{0, 0, 14, 13, 11, 11, 10, 9, 9, 3, 15, 14, 13, 11, 10, 10, 7, 8, 7, 7 },
		{1, 0, 13, 12, 12, 10, 9, 0, 14, 12, 12, 10, 10, 8, 8, 7, 7 },
		{2, 0, 15, 14, 13, 12, 11, 11, 9, 8, 0, 14, 12, 12, 10, 10, 8, 7, 7, 8, 6},
		{0, 15, 13, 12, 11, 9, 7, 15, 14, 13, 12, 10, 10, 10, 8, 7},
		{0, 0, 14, 12, 11, 10, 8, 8, 0, 15, 15, 12, 12, 10, 10, 9, 8, 7, 5 },
		{0, 0, 14, 12, 10, 9, 9, 15, 15, 13, 11, 10, 10, 8, 6, 7},
		{ 0, 15, 14, 12, 11, 10, 9, 8, 15, 15, 14, 12, 12, 12, 10, 10, 8, 8, 6 },
		{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4},
		{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4 },
		{ 0, 15, 15, 13, 12, 10, 8, 15, 15, 13, 12, 10, 10, 8, 8, 8, 6 },
		{0, 15, 14, 13, 11, 10, 8, 8, 15, 14, 14, 12, 12, 10, 9, 8, 7, 7},
		{0, 0, 15, 13, 12, 11, 10, 9, 7, 0, 14, 14, 13, 12, 10, 8, 9, 8, 7, 7},
		{0, 0, 14, 12, 10, 9, 8, 14, 14, 13, 12, 10, 9, 8, 7, 5},
		{0, 15, 15, 12, 11, 10, 9, 14, 14, 13, 11, 10, 8, 7, 7},
		{0, 15, 14, 12, 10, 9, 9, 15, 14, 12, 12, 12, 9, 9, 7},
		{1, 0, 14, 13, 12, 10, 9, 0, 14, 12, 10, 10, 8, 8, 8, 8, 6, 5, 4 },
		{0, 14, 14, 12, 12, 10, 8, 15, 14, 12, 10, 9, 9, 7, 5, 6, 4},
		{0, 14, 14, 12, 12, 10, 8, 0, 14, 12, 9, 8, 8, 7, 8},
		{0, 15, 13, 11, 9, 15, 15, 14, 12, 11, 10, 10, 8, 7, 7 },		//41th +ve 3 data end
		{15, 13, 11, 9, 9, 8, 0, 15, 14, 14, 13, 10, 9, 8, 8, 7, 4 },
		{0, 15, 13, 12, 11, 9, 9, 8, 0, 15, 13, 12, 11, 9, 9, 8},
		{0, 15, 14, 13, 12, 11, 10, 9, 8, 0, 14, 15, 12, 11, 11, 10, 9, 8, 8},
		{0, 15, 14, 13, 12, 11, 9, 8, 8, 15, 15, 14, 12, 11, 10, 9, 8, 7 },
		{15, 14, 15, 14, 13, 12, 10, 8, 8, 0, 15, 13, 13, 11, 11, 10, 8, 8},
		{15, 14, 13, 13, 11, 9, 8, 7, 15, 0, 15, 14, 13, 11, 11, 10, 8, 7, 8, 5 },
		{0, 0, 15, 14, 13, 12, 11, 10, 8, 8, 7, 1, 15, 14, 13, 12, 12, 11, 10, 8, 8, 8 },
		{0, 15, 13, 12, 10, 10, 8, 15, 13, 12, 11, 11, 10, 9, 7},
		{0, 15, 15, 14, 12, 11, 10, 9, 14, 15, 13, 12, 11, 10, 9, 8, 7, 7 },
		{2, 0, 14, 13, 13, 12, 10, 8, 10, 5, 0, 15, 14, 13, 12, 10, 10, 9, 7, 7 },		//51th +ve 3 data end
		{0, 15, 14, 12, 11, 10, 10, 8, 0, 15, 13, 12, 11, 10, 9, 7 },
		{0, 0, 15, 0, 13, 12, 11, 11, 9, 8, 15, 15, 13, 12, 11, 11, 10, 9, 8, 7, 7, 5 },
		{1, 0, 0, 13, 13, 12, 11, 10, 9, 14, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6},			
		{0, 0, 13, 11, 9, 15, 13, 11, 10, 8, 8},
		{15, 15, 12, 11, 9, 9, 0, 15, 13, 11, 10, 9, 8},
		{15, 13, 12, 10, 9, 15, 14, 12, 11, 10, 9, 8 },	
		{0, 13, 12, 11, 10, 15, 12, 11, 11, 8, 7, 7, 5 },
		{15, 15, 13, 12, 11, 11, 9, 0, 14, 12, 11, 11, 10, 8, 7, 6},
		{1, 15, 0, 13, 12, 11, 10, 13, 13, 12, 11, 10, 8, 6, 5},
		{0, 15, 12, 11, 11, 10, 14, 12, 12, 11, 10, 8, 7, 5},
		{0, 15, 13, 11, 10, 13, 13, 12, 11, 12, 10, 9, 6, 5 },
		{0, 15, 12, 11, 9, 15, 13, 12, 11, 10, 9, 7, 5, 4},
		{1, 15, 13, 11, 10, 8, 15, 14, 12, 12, 10, 8, 7, 6},
		{2, 1, 0, 15, 12, 12, 11, 10, 11, 14, 13, 11, 11, 9, 8, 7, 6 },
		{2, 1, 0, 14, 12, 11, 11, 10, 15, 14, 12, 11, 10, 9, 7, 7, 6, 4},
		{0, 15, 12, 11, 10, 14, 13, 13, 10, 8, 8, 7, 5},
		{0, 15, 13, 12, 10, 9, 14, 13, 12, 10, 10, 9, 6, 6, 4},
		{2, 1, 0, 15, 14, 12, 12, 11, 11, 11, 11, 9, 9, 14, 13, 12, 12, 11, 11, 10, 8, 8, 7 },
		{1, 0, 0, 15, 15, 14, 12, 11, 10, 9, 13, 14, 13, 12, 12, 11, 11, 11, 10, 8, 7, 8},
		{2, 0, 0, 0, 15, 12, 11, 11, 11, 10, 10, 9, 15, 15, 13, 12, 12, 11, 11, 10, 7, 8, 7, 6},
		{0, 3, 2, 15, 15, 13, 12, 11, 11, 10, 9, 9, 15, 15, 14, 13, 12, 11, 11, 10, 8, 7, 7, 8, 4},
		{2, 0, 0, 12, 13, 12, 12, 11, 10, 10, 9, 9, 15, 14, 13, 13, 11, 10, 8, 8, 7, 5},
		{15, 12, 11, 10, 12, 15, 13, 12, 12, 10, 9, 7, 8},		//74th -ve 3 data end

	};
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit9M, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit9);

}
void GestureRecognizerModified::writeCircleAngleStreamInFile(vector<int>& returnReducedVector,int& indexOfGestures,String gestureText){
	
	if(indexOfGesturesGlobal != indexOfGestures){
		
		//ostringstream convert;   // stream used for the conversion
		//convert << indexOfGestures;
		gestureTextVec.push_back(gestureText);
		 Mat tempMat;
		 tempMat.create(1,returnReducedVector.size(),CV_8U);

		 for(int i = 0; i< returnReducedVector.size(); i++){
			tempMat.at<uchar>(0,i) = returnReducedVector[i];
		 
		 }
		 angleMatVec.push_back(tempMat);
		// file<<"mat"+convert.str()<<tempMat ; 
		 indexOfGesturesGlobal = indexOfGestures;
	}  

	if(angleMatVec.size() == 10){
		writeCircleAngleStreamInFile(angleMatVec,gestureTextVec);
		indexOfGestures = 0;
	}
	 

	 
}

void GestureRecognizerModified::writeCircleAngleStreamInFile(vector<Mat>& angleMatVecSrc,vector<String>& gestureTextVecSrc){
	cv::FileStorage file;
	file.open("some_name.text", cv::FileStorage::WRITE);

	if(gestureTextVecSrc.size() == angleMatVecSrc.size())
	for(int i = 0; i< angleMatVecSrc.size(); i++){
		ostringstream convert;   // stream used for the conversion
		convert << i; 
		file<<"mat"+convert.str()<<angleMatVecSrc[i] ; 
		file<<"gestureText"<<gestureTextVecSrc[i]; 
	}
	angleMatVecSrc.clear();
	gestureTextVecSrc.clear();
	file.release();
}

static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
 

//http://www.cplusplus.com/reference/algorithm/adjacent_find/
//bool myfunction (int i, int j) {
//  return (i==j);
//}

//problem with this method is that it finds pattern only with same digit, ie  if there is pattern of 12 for 2 times 
//then it returns the 12 but if it has 12,11,12,11 then it does not understand it as pattern
//void GestureRecognizerModified::findRepetedPatternsInASrcVec(vector<int>& srcVec,int& returnFirst,int& returnSecond){
//	std::vector<int>::iterator it;
//
//  // using default comparison:
//  it = std::adjacent_find (srcVec.begin(), srcVec.end());
//
//  if (it!=srcVec.end()){
//	  returnFirst = *it;
//    std::cout << "the first pair of repeated elements are: " << *it << '\n';
//  }
//  //using predicate comparison:
//  it = adjacent_find (++it, srcVec.end(), myfunction);
//	
//  if (it!=srcVec.end()){
//	  returnSecond = *it;
//	  std::cout << "the second pair of repeated elements are: " << *it << '\n';
//  }
//}
 
//this method will search left pattern like : 7,8,9,9,8,7 
//param : srcVec: Angle vec chain
//param : return{attern: return str vec with direction
//param : limit: required threshold that needs to be matched in order to define direction..
void GestureRecognizerModified::findRepetedPatternsInASrcVecLeft(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterLeft = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 7-8-9 range?
		if(srcVec[i] == 6 || srcVec[i] == 7 || srcVec[i] == 8 || srcVec[i] == 9 || srcVec[i] == 10){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 6 || srcVec[i+1] == 7 || srcVec[i+1] == 8 || srcVec[i+1] == 9 || srcVec[i+1] == 10){
				counterLeft++;
				if(counterLeft >=limit){	
					returnPattern.push_back("Left");		
				}
			} else {	//if next element is not in the range, then reset counter..
				counterLeft = 0;
			}
		}
	}
}

//this method will search left pattern like : 3,4,5,5,4,3
//param : srcVec: Angle vec chain
//param : return{attern: return str vec with direction
//param : limit: required threshold that needs to be matched in order to define direction..
void GestureRecognizerModified::findRepetedPatternsInASrcVecUp(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterUp = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 3-4-5 range?
		if(srcVec[i] == 3 || srcVec[i] == 4 || srcVec[i] == 5){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 3 || srcVec[i+1] == 4 || srcVec[i+1] == 5){
				counterUp++;
				if(counterUp >=limit){	
					returnPattern.push_back("Up");		
				}
			} else {	//if next element is not in the range, then reset counter..
				counterUp = 0;
			}
		}
	}
}

//this method will search left pattern like : 13,12,11,11,12,13
//param : srcVec: Angle vec chain
//param : return{attern: return str vec with direction
//param : limit: required threshold that needs to be matched in order to define direction..
void GestureRecognizerModified::findRepetedPatternsInASrcVecDown(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterDown = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 11-12-13 range?
		if(srcVec[i] == 10 || srcVec[i] == 11 || srcVec[i] == 12 || srcVec[i] == 13 || srcVec[i] == 14){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 10 || srcVec[i+1] == 11 || srcVec[i+1] == 12 || srcVec[i+1] == 13 ||srcVec[i+1] == 14){
				counterDown++;
				if(counterDown >=limit){	
					returnPattern.push_back("Down");		
				}
			} else {	//if next element is not in the range, then reset counter..
				counterDown = 0;
			}
		}
	}
}


String GestureRecognizerModified::understandTheGestureWithShapeDescriptor(vector<Point3i>& srcVector){
	//convert 3D vector to 2D vector for shape recognision..
	vector<Point> srcVector2D;

	for(int cont = 0; cont < srcVector.size(); cont++){
		srcVector2D.push_back(Point(srcVector[cont].x,srcVector[cont].y));
	}

	String returnString;

	//serve closeness criteria
	if( closenessCriteria < 10){
		// The array for storing the approximation curve
		std::vector<cv::Point> approx;
		//cv::approxPolyDP( cv::Mat(srcVector2D),  approx,  cv::arcLength(cv::Mat(srcVector2D), true) * 0.02,  true);
		approxPolyDP(srcVector2D, approx, 10, true);

		if (approx.size() == 3){
			returnString += "Triangle";    // Triangles
		} else if (approx.size() >= 4 && approx.size() <= 6) {
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the degree (in cosines) of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc+1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

			// Sort ascending the corner degree values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest degree
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
			{
				// Detect rectangle or square
				cv::Rect r = cv::boundingRect(srcVector2D);
				double ratio = std::abs(1 - (double)r.width / r.height);

				if(ratio <= 0.02){
					returnString += "Square";    // Square
				} else {
					returnString += "Rectangle";    // Rectangle
				}
			}
			else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
				returnString += "PENTA";    // PENTA
			else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
				returnString += "HEXA";    // HEXA
		
		}  else {
			// Detect and label circles
			double area = cv::contourArea(srcVector2D);
			cv::Rect r = cv::boundingRect(srcVector2D);
			double radius = r.width / 2;

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
			{
				returnString += "Circle";    // Circle
			}
		}
	}	//end of closeness criteria

	return returnString;
}//EOM

bool GestureRecognizerModified::findCircle(vector<Point3i>& srcVectorOriginalSize, String& returnString,Mat& srcMat,
	int index){
	
	 
	//step1: find ratios for current gesture
	int stdWidth = 20;
	int stdHeight = 25;

	double widthRatio = (double)widthOriginal/(double)stdWidth;
	double heightRatio = (double)heightOriginal/(double)stdHeight;
	double finalRatio;

	if(widthRatio < heightRatio){
		finalRatio = widthRatio;
	} else {
		finalRatio = heightRatio;
	}

	if(finalRatio != 0){
		//step2: convert original size vec to standard size vec.
		vector<Point3i> scaledDownVec;
		for(int i = 0; i < srcVectorOriginalSize.size(); i++){
			scaledDownVec.push_back(Point3i(srcVectorOriginalSize[i].x/finalRatio,srcVectorOriginalSize[i].y/finalRatio,0));
		}
	 
		//step3: first save only element which is 5 pixels away from each other.
		vector<Point3i> scaledDownVecScattered;
		Point3i point = scaledDownVec[0];
		for(int cn = 0; cn < scaledDownVec.size() ; cn++){
			if(abs(point.x - scaledDownVec[cn].x) >= 5 || abs(point.y - scaledDownVec[cn].y) >= 5){
				scaledDownVecScattered.push_back(scaledDownVec[cn]);
				point = scaledDownVec[cn];
								 
			}
		}
		 
		//step4: then feed that vector to find angle method and get vector with all angles and codes
		vector<int> angleVecScattred;
		findAngleVector(scaledDownVecScattered,angleVecScattred);

		//step5: then convert that angle vet to testSvmMat for further calculations.
		//Common Mat for all SVM classifiers
		Mat testMatForSVM;
		testMatForSVM = Mat::zeros(1,50,CV_32FC1);

		int sizeOfsrcReducedVec = angleVecScattred.size();
		if(sizeOfsrcReducedVec > 50){		//it blocks the no of column greater than 50, because in training data, we train our chain for max 50 elements.
			sizeOfsrcReducedVec = 50;
		}
		 
		//assign angleVecScattred to returnedAngleVecReduced 
		//returnedAngleVecReduced = angleVecScattred;
		for(int reducedVecCnt = 0; reducedVecCnt < sizeOfsrcReducedVec ;reducedVecCnt++){
			testMatForSVM.at<float>(0,reducedVecCnt) = angleVecScattred[reducedVecCnt];
		}

		bool flagForCircle = false;
		float responseForCircle = SVMCircle.predict(testMatForSVM);
		if (responseForCircle == 1){
			//Gesture 5(circle) & 6(Ellipse/Zero) by Manual Methods
			if(heightOriginal <= widthOriginal*1.4){
				 
				returnString += "circle";
			} else {
				returnString += "ellipse or Digit 0";		//ellipse or 0
			}
			
			flagForCircle = true;		//initialize falg == true for string output
		}

	 
		//step6: visualize the scaled Down gesture if you want
		Point3i XL,XH,YL,YH;
		Rect boundingRect;
		int SDGestureWidth;
		int SDGestureHeight;
		
		findEdgePointsForTrajectory(scaledDownVec,XL,XH,YL,YH,boundingRect,SDGestureWidth,SDGestureHeight);
		drawTrajectoryBasedOnVecPoints(scaledDownVec,120,160,"circleSD",boundingRect);
		//newWayForGestureDetectionWithAdriansSuggestion(scaledDownVec,120,160,"imageName");
		  
		if(scaledDownVecScattered.size() != 0 && angleVecScattred.size() != 0){
			for (int i = 0; i < scaledDownVecScattered.size()-1; i++) {
				Point point1,point2;
				point1.x = scaledDownVecScattered[i].x*4*finalRatio;
				point1.y = scaledDownVecScattered[i].y*4*finalRatio;
				point2.x = scaledDownVecScattered[i+1].x*4*finalRatio;
				point2.y = scaledDownVecScattered[i+1].y*4*finalRatio;
							
				line(srcMat,point1, point2, Scalar(0,255,0), 3);
				//circle(colorMatPyrDown,point1,1,Scalar(255,255,255),2);
						 
			}
		}
		 
		writeCircleAngleStreamInFile(angleVecScattred,index,returnString);
		 
		return flagForCircle;
	}

	return false;
}

void GestureRecognizerModified::scaledDownSrcVec(vector<Point3i>& srcVectorOriginalSize,vector<Point3i>& scaledDownVecReturn,
	vector<int>& angleVecSDReturn,Mat& returnSDMat,Mat& srcMatForVisualization){
	//step1: find ratios for current gesture
	int stdWidth = 15;
	int stdHeight = 30;

	double widthRatio = (double)widthOriginal/(double)stdWidth;
	double heightRatio = (double)heightOriginal/(double)stdHeight;
	double finalRatio = 1;

	//check out if the ratio is smaller than 1, because if its do, then we are upscaling and it will crash the code
	 
	if(widthRatio < heightRatio){
		finalRatio = widthRatio;
	} else{
		finalRatio = heightRatio;
	}
	if(heightRatio < 1 || widthRatio < 1){
		finalRatio = 1;
	}
	 
	if(finalRatio != 0){
		//Step1: Assign this ratio to final ratio 
		rationForGestureReduced = finalRatio;

		//step2: convert original size vec to standard size vec.
		vector<Point3i> scaledDownVec;
		for(int i = 0; i < srcVectorOriginalSize.size(); i++){
			scaledDownVec.push_back(Point3i(srcVectorOriginalSize[i].x/finalRatio,srcVectorOriginalSize[i].y/finalRatio,0));
		}
	 
		//step3: first save only element which is 5 pixels away from each other.
		//vector<Point3i> scaledDownVecScattered;
		Point3i point = scaledDownVec[0];
		for(int cn = 0; cn < scaledDownVec.size() ; cn++){
			if(abs(point.x - scaledDownVec[cn].x) >= 3 || abs(point.y - scaledDownVec[cn].y) >= 3){
				scaledDownVecReturn.push_back(scaledDownVec[cn]);
				point = scaledDownVec[cn];
								 
			}
		}
			if(scaledDownVecReturn.size() > 0){
		 
			 //cout<<"inside step4.1"<<endl;
			//step4: then feed that vector to find angle method and get vector with all angles and codes
			//vector<int> angleVecScattred;
			findAngleVector(scaledDownVecReturn,angleVecSDReturn);
			 
			//cout<<"inside step4.2"<<endl;
			//step5: then convert that angle vet to testSvmMat for further calculations.
			//Common Mat for all SVM classifiers
			//Mat testMatForSVM;
			returnSDMat = Mat::zeros(1,50,CV_32FC1);

			int sizeOfsrcReducedVec = angleVecSDReturn.size();
			if(sizeOfsrcReducedVec > 50){		//it blocks the no of column greater than 50, because in training data, we train our chain for max 50 elements.
				sizeOfsrcReducedVec = 50;
			}
		 
			//assign angleVecScattred to returnedAngleVecReduced 
			//returnedAngleVecReduced = angleVecScattred;
			for(int reducedVecCnt = 0; reducedVecCnt < sizeOfsrcReducedVec ;reducedVecCnt++){
				returnSDMat.at<float>(0,reducedVecCnt) = angleVecSDReturn[reducedVecCnt];
			}

			//Draw trajectory Points
			//step6: visualize the scaled Down gesture if you want
			//Point3i XL,XH,YL,YH;
			Rect boundingRect;
			int SDGestureWidth;
			int SDGestureHeight;
			findEdgePointsForTrajectory(scaledDownVecReturn,extreamLeftScaledDownVec,extreamRightScaledDownVec,extreamTopScaledDownVec,extreamBottomScaledDownVec
				,boundingRect,SDGestureWidth,SDGestureHeight);

			startPointScaledDownVec = scaledDownVecReturn[0];
			stopPointScaledDownVec = scaledDownVecReturn[scaledDownVecReturn.size() - 1];
			//cout<<"inside step4.3"<<endl;
			//cout<<"scaledDownVecReturn"<<scaledDownVecReturn<<endl;
			//cout<<"boundingRect"<<boundingRect<<endl;
			drawTrajectoryBasedOnVecPoints(scaledDownVecReturn,120,160,"Digit",boundingRect);
		  
			angleVecForDisplayPurpose.clear();			//clear before using it.
			pointsVecForDisplayPurpose.clear();			//clear before using it.
			angleVecForDisplayPurpose = angleVecSDReturn;
			points3iVecForDisplayPurpose = scaledDownVecReturn;
			  
			//drawTrajectoryOnMat(scaledDownVecReturn,angleVecSDReturn,srcMatForVisualization,finalRatio);


		//cout<<"inside step4.5"<<endl;	
		}
	}
}



void GestureRecognizerModified::drawTrajectoryOnMat(Mat& drawMat){
	if(points3iVecForDisplayPurpose.size() != 0 && angleVecForDisplayPurpose.size() != 0){
		for (int i = 0; i < points3iVecForDisplayPurpose.size()-1; i++) {
			Point point1,point2;
			point1.x = points3iVecForDisplayPurpose[i].x*4*rationForGestureReduced;
			point1.y = points3iVecForDisplayPurpose[i].y*4*rationForGestureReduced;
			point2.x = points3iVecForDisplayPurpose[i+1].x*4*rationForGestureReduced;
			point2.y = points3iVecForDisplayPurpose[i+1].y*4*rationForGestureReduced;
							
			line(drawMat,point1, point2, Scalar(0,255,0), 3);
			//circle(colorMatPyrDown,point1,1,Scalar(255,255,255),2);
							 
			stringstream text;
			text<< angleVecForDisplayPurpose[i];
			putText(drawMat,text.str(),point1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),2,1);
		}
	}

	if(pointsVecForDisplayPurpose.size() != 0 && angleVecForDisplayPurpose.size() != 0){
		for (int i = 0; i < pointsVecForDisplayPurpose.size()-1; i++) {
			Point point1,point2;
			point1.x = pointsVecForDisplayPurpose[i].x*4*rationForGestureReduced;
			point1.y = pointsVecForDisplayPurpose[i].y*4*rationForGestureReduced;
			point2.x = pointsVecForDisplayPurpose[i+1].x*4*rationForGestureReduced;
			point2.y = pointsVecForDisplayPurpose[i+1].y*4*rationForGestureReduced;
							
			line(drawMat,point1, point2, Scalar(0,255,0), 3);
			//circle(colorMatPyrDown,point1,1,Scalar(255,255,255),2);
							 
			stringstream text;
			text<< angleVecForDisplayPurpose[i];
			putText(drawMat,text.str(),point1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),2,1);
		}
	}
}
void GestureRecognizerModified::drawTrajectoryOnMat(vector<Point3i>& srcVecPoints,vector<int>& angleVec,Mat& drawMat,double ratio){
	if(srcVecPoints.size() != 0 && angleVec.size() != 0){
				for (int i = 0; i < srcVecPoints.size()-1; i++) {
					Point point1,point2;
					point1.x = srcVecPoints[i].x*4*ratio;
					point1.y = srcVecPoints[i].y*4*ratio;
					point2.x = srcVecPoints[i+1].x*4*ratio;
					point2.y = srcVecPoints[i+1].y*4*ratio;
							
					line(drawMat,point1, point2, Scalar(0,255,0), 3);
					//circle(colorMatPyrDown,point1,1,Scalar(255,255,255),2);
							 
					stringstream text;
					text<< angleVec[i];
					putText(drawMat,text.str(),point1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),2,1);
				}
			}
}
bool GestureRecognizerModified::findCircle(vector<int>& angleVecScattred, String& returnString,Mat& srcMat, int index){
	//cout<<"Inside Circle"<<endl; 
	bool flagForCircle = false;
	float responseForCircle = SVMCircle.predict(srcMat);
	if (responseForCircle == 1){
		//Gesture : Circle 	 
		//returnString += "Circle, ";
			 
		flagForCircle = true;		//initialize falg == true for string output
	}
				 
	

	return flagForCircle;
		 
}

bool GestureRecognizerModified::findDigit1(vector<int>& angleVecScattred, String& returnString,Mat& srcMat, int index){
	//cout<<"Inside Digit1"<<endl; 
	bool flagForDigit1 = false;
	float responseForDigit1 = SVMDigit1.predict(srcMat);
	if (responseForDigit1 == 1){
		//Gesture : Digit1 	 
		returnString += "Digit1, ";
			 
		flagForDigit1 = true;		//initialize falg == true for string output
	}
				 
	

	return flagForDigit1;
		 
}

bool GestureRecognizerModified::findDigit2(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit2 = false;
	float responseForDigit2 = SVMDigit2.predict(srcMat);
	if (responseForDigit2 == 1){
		//Gesture : Digit2 	 
		returnString += "Digit2, ";
			 
		flagForDigit2 = true;		//initialize falg == true for string output
	} 
	return flagForDigit2;
}

bool GestureRecognizerModified::findDigit3(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit3 = false;
	float responseForDigit3 = SVMDigit3.predict(srcMat);
	if (responseForDigit3 == 1){
		//Gesture : Digit3 	 
		returnString += "Digit3, ";
			 
		flagForDigit3 = true;		//initialize falg == true for string output
	}
	 
	return flagForDigit3;
}

bool GestureRecognizerModified::findDigit4(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit4 = false;
	float responseForDigit4 = SVMDigit4.predict(srcMat);
	if (responseForDigit4 == 1){
		//Gesture : Digit4 	 
		returnString += "Digit4, ";
			 
		flagForDigit4 = true;		//initialize falg == true for string output
	}
			 
	return flagForDigit4;
}

bool GestureRecognizerModified::findDigit5(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){

	bool flagForDigit5 = false;
	float responseForDigit5 = SVMDigit5.predict(srcMat);
	if (responseForDigit5 == 1){
		//Gesture : Digit5 	 
		returnString += "Digit5, ";
			 
		flagForDigit5 = true;		//initialize falg == true for string output
	}
				  
	return flagForDigit5;
}

bool GestureRecognizerModified::findDigit6(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit6 = false;
	float responseForDigit6 = SVMDigit6.predict(srcMat);
	if (responseForDigit6 == 1){
		//Gesture : Digit6 	 
		returnString += "Digit6, ";
			 
		flagForDigit6 = true;		//initialize falg == true for string output
	}
	 
	return flagForDigit6;
}

bool GestureRecognizerModified::findDigit7(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	//cout<<"Inside Digit7"<<endl;
	bool flagForDigit7 = false;
	float responseForDigit7 = SVMDigit7.predict(srcMat);
	if (responseForDigit7 == 1){
		//Gesture : Digit7 	 
		returnString += "Digit7, ";
			 
		flagForDigit7 = true;		//initialize falg == true for string output
	} 
	return flagForDigit7;
}

bool GestureRecognizerModified::findDigit8(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit8 = false;
	float responseForDigit8 = SVMDigit8.predict(srcMat);
	if (responseForDigit8 == 1){
		//Gesture : Digit8 	 
		returnString += "Digit8, ";
			 
		flagForDigit8 = true;		//initialize falg == true for string output
	} 
	return flagForDigit8;
}

bool GestureRecognizerModified::findDigit9(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForDigit9 = false;
	float responseForDigit9 = SVMDigit9.predict(srcMat);
	if (responseForDigit9 == 1){
		//Gesture : Digit9 	 
		returnString += "Digit9, ";
			 
		flagForDigit9 = true;		//initialize falg == true for string output
	} 
	return flagForDigit9;
}

bool GestureRecognizerModified::findLetterS(vector<int>& angleVecScattred, String& returnString,Mat& srcMat,int index){
	bool flagForLetterS = false;
	float responseForLetterS = SVMLetterS.predict(srcMat);
	if (responseForLetterS == 1){
		//Gesture : LetterS 	 
		returnString += "LetterS, ";
			 
		flagForLetterS = true;		//initialize falg == true for string output
	}
	 
	return flagForLetterS;
}
//Para: srcVectorScattered is a 3D points with scattered trajectory. 
//		ie they are not that contineous they are atleast 5 pixels seperated from each other in x or y direction.
//Para: srcReducedVec is a vector of codes from scattered points, helps to train and test SVM
//Para: srcVectorOriginal is a 3D points with scattered trajectory helps to find shapes from shape descriptors.
bool GestureRecognizerModified::understandTheGesture(vector<Point3i>& srcVectorOriginal,vector<Point3i>& srcVector5x5Scattered,Mat& srcMatForShapes,
	Rect& returnedGestureBoundingRect,int indexOfGestures,String& returnGestureText){
	
	//step0: clear some of the global vectors
		points3iVecForDisplayPurpose.clear();
		pointsVecForDisplayPurpose.clear();
		angleVecForDisplayPurpose.clear();

	//step1: find scattered Vec from original src Vec of Points..
	 
	findEdgePointsForTrajectory(srcVectorOriginal,extreamLeftOriginal,extreamRightOriginal,extreamTopOriginal,extreamBottomOriginal,boundingRectAroundGestureOriginal,
		widthOriginal,heightOriginal);
	 
	drawTrajectoryBasedOnVecPoints(srcVectorOriginal,120,160,"original",boundingRectAroundGestureOriginal);
	 
	vector<Point> srcVector2D;
	Point3i point;
	for(int cn = 0; cn < srcVector5x5Scattered.size() ; cn++){
			srcVector2D.push_back(Point(srcVector5x5Scattered[cn].x,srcVector5x5Scattered[cn].y));
	}
  
	//step2: find the angle vector for original src vector of Points and draw it..
	vector<int> angleVec5x5Scattred;
	findAngleVector(srcVector2D,angleVec5x5Scattred);
	 
	//step3: use these two vectors for shape descriptors..
	  //cout<<"step2"<<endl;
	ostringstream convertWidth,convertHeight;   // stream used for the conversion
	convertWidth << widthOriginal;
	convertHeight << heightOriginal;
	 
	String returnString = "W : " + convertWidth.str() + "H : " + convertHeight.str() + ", ";	
	 
	bool flag = false;	//flag if any of the following gesture is detected.?
	
	bool flagForTriangle = false;	//flag if any of the following gesture is detected.?
	bool flagForSquare = false;		//flag if any of the following gesture is detected.?
	bool flagForCircle = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit1 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit2 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit3 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit4 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit5 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit6 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit7 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit8 = false;		//flag if any of the following gesture is detected.?
	bool flagForDigit9 = false;		//flag if any of the following gesture is detected.?
	bool flagForLetterS = false;	//flag if any of the following gesture is detected.?
	 
	  
	//initialize all the important points for original trajectory and 5x5 scattered trajectory
	startPointOriginal.x = srcVectorOriginal[0].x;
	startPointOriginal.y = srcVectorOriginal[0].y;
	stopPointOriginal.x =  srcVectorOriginal[srcVectorOriginal.size()-1].x;
	stopPointOriginal.y =  srcVectorOriginal[srcVectorOriginal.size()-1].y;
	closenessCriteria = sqrt (pow ((startPointOriginal.x - stopPointOriginal.x),2.0) + pow ((startPointOriginal.y - stopPointOriginal.y),2.0));	//it checks if the start point and stop point are close enough to say its a close gesture or not?
							//sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20
	  
	if( closenessCriteria < 10){
		
		returnString += "CNC: " ;
		flag = true;
		 
		// The array for storing the approximation curve
		 
		//approxPolyDP(srcVector2D5x5Scattered, approx, 10, true);
		/*double digit = arcLength(Mat(srcVector2D5x5Scattered), true)*0.02;
		cout<<"digit : "<<digit<<endl;*/
		approx.clear();		//clear this global vector before using it.
		if(srcVector2D.size() > 0){
			approxPolyDP(srcVector2D, approx, 6.5, true);

			for (int i = 0; i < approx.size(); i++) {
				Point point1;
				point1.x = approx[i].x*4;
				point1.y = approx[i].y*4;
				putText(srcMatForShapes,"angle",point1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,0,0),2,1);
				circle(srcMatForShapes,point1,3,Scalar(255,255,0),3,8);
			}
		 
			if (approx.size() == 3){
				returnString += "Triangle, ";    // Triangles
				flagForTriangle = true;
		 
			} else if (approx.size() >= 4 && approx.size() <= 6 ) {
				// Number of vertices of polygonal curve
			 
				int vtc = approx.size();
			 
				// Get the degree (in cosines) of all corners
				double maxCosine = 0;
			 
				for( int j = 2; j < 5; j++ ) {
				 
					// find the maximum cosine of the angle between joint edges
					double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
					maxCosine = MAX(maxCosine, cosine);
				 
				} 
				// Use the degrees obtained above and the number of vertices
				// to determine the shape of the contour
			 
				if ((((vtc == 4/* || vtc == 5*/)  && maxCosine < 0.3) && flagForTriangle != true )
					/*&& detectSquareWithSimpleTechniqueModified(angleVec5x5Scattred)*/) {
					// Detect rectangle or square
					if(heightOriginal != 0){
						float ratio = widthOriginal / heightOriginal; 
						if(ratio >= 0.8 &&  ratio <= 1.2){
							returnString += "Square, ";    // Square
							flagForSquare = true;
						} else {
							returnString += "Rectangle, ";    // Rectangle
							flagForSquare = true;
						}
					}
					//setLabel(dst, ratio <= 0.02 ? "SQU" : "RECT", contours[i]);
				}
	  
			} //else if (approx.size() >= 4 && approx.size() <= 6)
	 
			//if any of the flag is set then display trajectory
			if(flagForTriangle || flagForSquare){
				//drawTrajectoryOnMat(srcVector2D,angleVec5x5Scattred,srcMatForShapes,1.0); 
				angleVecForDisplayPurpose.clear();			//clear before using it.
				pointsVecForDisplayPurpose.clear();			//clear before using it.
				angleVecForDisplayPurpose = angleVec5x5Scattred;
				pointsVecForDisplayPurpose = srcVector2D;
				rationForGestureReduced = 1;
			} 
		}//if srcVector2D.size() > 0
	}	//if(sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20)
	 
	if(flagForTriangle == false && flagForSquare == false && flagForCircle == false){
		
		vector<Point3i> scaledDownVec;
		vector<int> angleVecReduced;
		Mat testMatSVM;
		  
		scaledDownSrcVec(srcVectorOriginal,scaledDownVec,angleVecReduced,testMatSVM,srcMatForShapes);
		 
		if( closenessCriteria < 10 && angleVecReduced.size() > 0){		//angleVecReduced.size() must be > 0, otherwise it's crashes
			 
			//Gesture  (8-Eight): condition is very imp..  
			//condition 0: closenessCriteria < 10	
			//condition 2: chain must start with 5,6,7,8,9,10,11 and stop with {4,5,6,7,8}
			if((angleVecReduced[0] == 5 || angleVecReduced[0] == 6 || angleVecReduced[0] == 7 || angleVecReduced[0] == 8 || angleVecReduced[0] == 9 ||
				angleVecReduced[0] == 10 || angleVecReduced[0] == 11) &&  
				(angleVecReduced[angleVecReduced.size() - 1] == 4 || angleVecReduced[angleVecReduced.size() - 1] == 5 || angleVecReduced[angleVecReduced.size() - 1] == 6 ||
				angleVecReduced[angleVecReduced.size() - 1] == 7 || angleVecReduced[angleVecReduced.size() - 1] == 8 )){	
					flagForDigit8 = findDigit8(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}
		 
			 //findCircle
			//if(flagForTriangle != true && flagForDigit8 != true && flagForSquare != true ){
			//	// Detect and label circles
			//	double area = cv::contourArea(srcVector2D);
			//	//cv::Rect r = cv::boundingRect(contours[i]);
			//	double radius = widthOriginal / 2;

			//	//if (std::abs(1 - ((double)widthOriginal / heightOriginal)) <= 0.2 &&
			//	//	std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
			//	//{
			//		//flagForCircle = findCircle(srcVectorOriginal,returnString,srcMatForShapes,indexOfGestures);
			//	//}  
			//}
			//Gesture: Circle : Condition : should not be 8
			if(!flagForDigit8){
				flagForCircle = findCircle(angleVecReduced,returnString,testMatSVM,indexOfGestures);
				if(flagForCircle){
					if(heightOriginal <= widthOriginal*1.4){
				 
						returnString += "circle";
					} else {
						returnString += "ellipse or Digit 0";		//ellipse or 0
					}
				}
			}
		} else if(angleVecReduced.size() > 0){		//angleVecReduced.size() must be > 0, otherwise it's crashes
				
			//Digit1: condition, first element has to be {0,1,2,3,15,14} and last element {10,11,12,13,14}  and abs(extreamLeft.x - extreamBottom.x) > 10
			if((angleVecReduced[0] == 0 || angleVecReduced[0] == 1 || angleVecReduced[0] == 2 || angleVecReduced[0] == 3 || angleVecReduced[0] == 14 || angleVecReduced[0] == 15 ) 
				&& (angleVecReduced[angleVecReduced.size() - 1] == 10 || angleVecReduced[angleVecReduced.size() - 1] == 11 || angleVecReduced[angleVecReduced.size() - 1] == 12 || 
				angleVecReduced[angleVecReduced.size() - 1] == 13 || angleVecReduced[angleVecReduced.size() - 1] == 14) 
				&& abs(extreamLeftOriginal.x - extreamBottomOriginal.x) >= 10){
					flagForDigit1 = findDigit1(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}
			//cout<<"step4.2"<<endl;
			//Digit7: condition, first element has to be {0,1,2,15,14} and last elemt {10,11,12,13} 
			if( (angleVecReduced[0] == 2 || angleVecReduced[0] == 1 || angleVecReduced[0] == 0 || angleVecReduced[0] == 15 || angleVecReduced[0] == 14) && 
					(angleVecReduced[angleVecReduced.size() - 1] == 10 || angleVecReduced[angleVecReduced.size() - 1] == 11 || angleVecReduced[angleVecReduced.size() - 1] == 12 || angleVecReduced[angleVecReduced.size() - 1] == 13) 
						&& abs(extreamLeftOriginal.x - extreamBottomOriginal.x) < 10){
							flagForDigit7 = findDigit7(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}

			//Digit2: condition, first element has to be {0,1,2,15,14} and last must be any of {0,1,2,15,14} /*and the last point has to be extream left point*/
			if((angleVecReduced[0] == 2 || angleVecReduced[0] == 1 || angleVecReduced[0] == 0 || angleVecReduced[0] == 15 || angleVecReduced[0] == 14) &&
				(angleVecReduced[angleVecReduced.size() - 1] == 0 || angleVecReduced[angleVecReduced.size() - 1] == 1 || 
					angleVecReduced[angleVecReduced.size() - 1] == 2 || angleVecReduced[angleVecReduced.size() - 1] == 15)){
						flagForDigit2 = findDigit2(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}

			//Digit3: condition, first element has to be 0,1,2,3,13,14,15 and last must be any of {4,5,6,7,8,9,10,11}
			if((angleVecReduced[0] == 0 || angleVecReduced[0] == 1 || angleVecReduced[0] == 2 || angleVecReduced[0] == 3 || angleVecReduced[0] == 13
				|| angleVecReduced[0] == 14 || angleVecReduced[0] == 15) && (angleVecReduced[angleVecReduced.size() - 1] == 5 || 
				angleVecReduced[angleVecReduced.size() - 1] == 6 || angleVecReduced[angleVecReduced.size() - 1] == 7 || angleVecReduced[angleVecReduced.size() - 1] == 8
				|| angleVecReduced[angleVecReduced.size() - 1] == 9 )){
					flagForDigit3 = findDigit3(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}

			////Digit4: condition, first element has to be {9,10,11,12} && startPoint.x < stopPoint.p
			if((angleVecReduced[0] == 9 || angleVecReduced[0] == 10 || angleVecReduced[0] == 11 || angleVecReduced[0] == 12 ) 
				&& startPointOriginal.x <= stopPointOriginal.x){
					flagForDigit4 = findDigit4(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}
			
	 
			//Gesture 5 Vs S: Tradition way.. 
			if(startPointOriginal.y < stopPointOriginal.y && startPointOriginal.x > stopPointOriginal.x ){
			//Gesture  (5-Five): condition is very imp.. It is similar to Letter S
				if((angleVecReduced[0] == 6 || angleVecReduced[0] == 7 || angleVecReduced[0] == 8 || angleVecReduced[0] == 9 || angleVecReduced[0] == 10) &&		//chain must start with 6,7,8,9,10
					(angleVecReduced[angleVecReduced.size()-1] == 4 || angleVecReduced[angleVecReduced.size()-1] == 5 || 
						angleVecReduced[angleVecReduced.size()-1] == 6
							|| angleVecReduced[angleVecReduced.size()-1] == 7 ||angleVecReduced[angleVecReduced.size()-1] == 8)) {//chain must stop with 4,5,6,7 or 8
								 
								vector<String> returnStrVecLeft,returnStrVecDown;
								findRepetedPatternsInASrcVecLeft(angleVecReduced,returnStrVecLeft,1);		//additional condition for Digit5
								findRepetedPatternsInASrcVecDown(angleVecReduced,returnStrVecDown,1);		//additional condition for Digit5
									if(returnStrVecLeft.size() > 0 && returnStrVecDown.size() > 0 && !flagForDigit3){
										flagForDigit5 = findDigit5(angleVecReduced,returnString,testMatSVM,indexOfGestures);
										 
									} 
				}
				//Gesture  (S-Letter) 
				//Rule1: start point has to be on top-left of stop point.
				if(stopPointScaledDownVec.x == extreamLeftScaledDownVec.x && stopPointScaledDownVec.y == extreamLeftScaledDownVec.y &&	//end point must be extream left point.
					(angleVecReduced[0] == 6 || angleVecReduced[0] == 7 || angleVecReduced[0] == 8 || 
						angleVecReduced[0] == 9 || angleVecReduced[0] == 10 && !flagForDigit5)){	//chain must start with 6,7,8,9 or 10
							//float response = SVMLetterS.predict(testMatForSVMGesture5AndS);
							flagForLetterS = findLetterS(angleVecReduced,returnString,testMatSVM,indexOfGestures);
							 
				} 
				 
			}
			//Digit6: condition, first element has to be 6,7,8,9,10,11 and last must be any of {8,9,10,11,12,13} && startPointOriginal.x > stopPointOriginal.x
			if((angleVecReduced[0] == 6 || angleVecReduced[0] == 7 || angleVecReduced[0] == 8 || angleVecReduced[0] == 9 || angleVecReduced[0] == 10
				|| angleVecReduced[0] == 11) && (angleVecReduced[angleVecReduced.size() - 1] == 8 || angleVecReduced[angleVecReduced.size() - 1] == 9 || 
					angleVecReduced[angleVecReduced.size() - 1] == 10 || angleVecReduced[angleVecReduced.size() - 1] == 11 || angleVecReduced[angleVecReduced.size() - 1] == 12
						|| angleVecReduced[angleVecReduced.size() - 1] == 13) && startPointScaledDownVec.x > stopPointScaledDownVec.x && !flagForLetterS &&
							stopPointScaledDownVec.y < extreamBottomScaledDownVec.y/*this is helpful for Digit6 vs 9*/){
		
					flagForDigit6 = findDigit6(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}

		

			//Gesture  (9-Nine): condition is very imp..  
			//condition 1: chain must start with 5,6,7,8,9,10,11 and stop with {10,11,12,13} && flagForDigit4 == false && stopPoint.x == extreamBottom.x && stopPoint.y == extreamBottom.y 
			if((angleVecReduced[0] == 5 || angleVecReduced[0] == 6 || angleVecReduced[0] == 7 || angleVecReduced[0] == 8 || angleVecReduced[0] == 9 ||
				angleVecReduced[0] == 10 || angleVecReduced[0] == 11) && (angleVecReduced[angleVecReduced.size() - 1] == 10 || angleVecReduced[angleVecReduced.size() - 1] == 11 || angleVecReduced[angleVecReduced.size() - 1] == 12 ||
						angleVecReduced[angleVecReduced.size() - 1] == 13 || angleVecReduced[angleVecReduced.size() - 1] == 14) && !flagForDigit4
						&& abs(startPointOriginal.x - extreamBottomOriginal.x) < 10){	
									flagForDigit9 = findDigit9(angleVecReduced,returnString,testMatSVM,indexOfGestures);
			}

			 
		}	//else there is not closecriteria
		//Write the chain code to text file.
		writeCircleAngleStreamInFile(angleVecReduced,indexOfGestures,returnString);
	}
	 
	 
	returnGestureText = returnString;
	bool returnFlag = (flagForTriangle || flagForSquare || flagForCircle 
		|| flagForDigit1 || flagForDigit2 || flagForDigit3 || flagForDigit4 || flagForDigit5 
		|| flagForDigit6 || flagForDigit7 || flagForDigit8 || flagForDigit9 || flagForLetterS);
	
	return returnFlag;
  
}
 
//this manual detection works very well.  
bool GestureRecognizerModified::detectSquareWithSimpleTechniqueModified(vector<int>& srcVector){

	//remove all duplicate elements from the chain
	vector<int> reducedVec,quarterVec;
	
	int nonSquareCount = 0;
	
	int firstQuarterCount = 0;
	int secondQuarterCount = 0;
	int thirdQuarterCount = 0;
	int fourthQuarterCount = 0;

	for(int i = 0; i < srcVector.size(); i++){
		//create new quarter vec..
		if(srcVector[i] == 5 || srcVector[i] == 4 || srcVector[i] == 3){
			quarterVec.push_back(1);		//first quarter..
			firstQuarterCount++;	//first quarter..
		} else if(srcVector[i] == 7 || srcVector[i] == 8 || srcVector[i] == 9){
			quarterVec.push_back(2);		//Second quarter..
			secondQuarterCount++;		//Second quarter..
		}  else if(srcVector[i] == 11 || srcVector[i] == 12 || srcVector[i] == 13){
			quarterVec.push_back(3);		//third quarter..
			thirdQuarterCount++;	//third quarter..
		}  else if(srcVector[i] == 15 || srcVector[i] == 0 || srcVector[i] == 1){
			quarterVec.push_back(4);		//fourth quarter..
			fourthQuarterCount++;		//fourth quarter..
		}	else {
			nonSquareCount++;
		}	
	}

	//this condition is for checking about square walls which are not more than 1 elements, that means if square walls are as 1 element then its not a square.
	if(firstQuarterCount <= 1 || secondQuarterCount <= 1 || thirdQuarterCount <= 1 || fourthQuarterCount <= 1){
		return false;
	}

	removeDuplicateElements(quarterVec,reducedVec);
	 
	if(nonSquareCount > 1 ){		//this is condition for nonSquareElement in the chhain code.. 
									//Because in the real world, there has to be some elements which are not part of the square walls.
		return false;
	}
	 
	//remove any possibility of having duplicate elements in quarterVec
	//vector<int> reducedQuarterVec;
	//removeDuplicateElements(quarterVec,reducedQuarterVec);
	//find pattern for Square.
	if(reducedVec.size() == 4){
		if(reducedVec[0] == 1 && reducedVec[1] == 4 && reducedVec[2] == 3 && reducedVec[3] == 2){	 // square combination : 1-4-3-2
			return true;
		} else if(reducedVec[0] == 2 && reducedVec[1] == 1 && reducedVec[2] == 4 && reducedVec[3] == 3){	 // square combination : 2-1-4-3
			return true;
		}  else if(reducedVec[0] == 3 && reducedVec[1] == 2 && reducedVec[2] == 1 && reducedVec[3] == 4){	 // square combination : 3-2-1-4
			return true;
		}  else if(reducedVec[0] == 4 && reducedVec[1] == 3 && reducedVec[2] == 2 && reducedVec[3] == 1){	 // square combination : 4-3-2-1
			return true;
		}
	}
}

//TODO: find batter way to calculate an angle...
double GestureRecognizerModified::findAngleBetweenTwoPoints(Point3i& A,Point3i& B){
// This function calculates the angle of the line from A to B with respect to the positive X-axis in degrees
 
	double returnValue;
	//condition 1: Quarter 1..
	 if(B.x > A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			returnValue = acos (param) * 180.0 / PI;
		}
		returnValue = 45;	//if denominator is 0 then assum first quarter angle 45'
	} 
	 

	//condition 2: Quarter 2..
	if(B.x < A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 180 - theta;
		}
		returnValue = 135;	//if denominator is 0 then assum second quarter angle 135'
	}
	 
	//condition 3: Quarter 3..
	if(B.x < A.x && B.y > A.y){

		//cos(theta) = 180 + abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 180 + theta;
		}
		returnValue = 225;	//if denominator is 0 then assum third quarter angle 225'
	}

	//condition 4: Quarter 4..
	if(B.x > A.x && B.y > A.y){

		//cos(theta) = 360 - abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 360 - theta;
		}
		returnValue = 315;	//if denominator is 0 then assum fourth quarter angle 315'
	}

	//condition 5: 0 position..
	if(B.y == A.y){
		if(B.x > A.x)
		returnValue = 0;	//if both y are same and B.x > A.x then it should be 0'
	}
	//condition 6: 180 position..
	if(B.y == A.y){
		if(B.x < A.x)
		returnValue = 180;	//if both y are same and B.x < A.x then it should be 180'
	}
	//condition 7: 90 position..
	if(B.x == A.x){
		if(B.y < A.y)
		returnValue = 90;	//if both x are same and B.y < A.y then it should be 90'
	}
	//condition 8: 270 position..
	if(B.x == A.x){
		if(B.y > A.y)
		returnValue = 270;	//if both x are same and B.y > A.y then it should be 270'
	}

	return returnValue;
}

double GestureRecognizerModified::findAngleBetweenTwoPoints(Point& A,Point& B){
// This function calculates the angle of the line from A to B with respect to the positive X-axis in degrees
 
	double returnValue;
	//condition 1: Quarter 1..
	 if(B.x > A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			returnValue = acos (param) * 180.0 / PI;
		}
		returnValue = 45;	//if denominator is 0 then assum first quarter angle 45'
	} 
	 

	//condition 2: Quarter 2..
	if(B.x < A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 180 - theta;
		}
		returnValue = 135;	//if denominator is 0 then assum second quarter angle 135'
	}
	 
	//condition 3: Quarter 3..
	if(B.x < A.x && B.y > A.y){

		//cos(theta) = 180 + abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 180 + theta;
		}
		returnValue = 225;	//if denominator is 0 then assum third quarter angle 225'
	}

	//condition 4: Quarter 4..
	if(B.x > A.x && B.y > A.y){

		//cos(theta) = 360 - abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			returnValue = 360 - theta;
		}
		returnValue = 315;	//if denominator is 0 then assum fourth quarter angle 315'
	}

	//condition 5: 0 position..
	if(B.y == A.y){
		if(B.x > A.x)
		returnValue = 0;	//if both y are same and B.x > A.x then it should be 0'
	}
	//condition 6: 180 position..
	if(B.y == A.y){
		if(B.x < A.x)
		returnValue = 180;	//if both y are same and B.x < A.x then it should be 180'
	}
	//condition 7: 90 position..
	if(B.x == A.x){
		if(B.y < A.y)
		returnValue = 90;	//if both x are same and B.y < A.y then it should be 90'
	}
	//condition 8: 270 position..
	if(B.x == A.x){
		if(B.y > A.y)
		returnValue = 270;	//if both x are same and B.y > A.y then it should be 270'
	}

	return returnValue;
}


//Param:	srcVec is input vec with all codes
int GestureRecognizerModified::findNoOfStreightSidesInVector(vector<int>& srcVec){

	vector<int> valueVec;
	int valueIndex = 0;
	int value = srcVec[0];
	int returnValue = 0;
	for(int i = 0; i < srcVec.size()-1; i++){
		//for each element and next, if they are close enought
		if((abs(value - srcVec[i+1]) <= 2) || (abs(value - srcVec[i+1]) == 15) || (abs(value - srcVec[i+1]) == 14)){
			valueIndex++;
			
			
			//if(valueIndex >= 2){	
			//	returnValue++;
			//	valueIndex = 0;		//very imp.. reintialize this once checked.
			//}
		} else {		//if the element and the next element is not close then initialize value to next element so it can consider it for next time.
			valueVec.push_back(valueIndex);	
			value = srcVec[i+1];	
			valueIndex = 0;
		}


		
	}

	//add the last one manually..
	valueVec.push_back(valueIndex);	
	  
	for(int i = 0; i < valueVec.size(); i++){
		if(valueVec[i] >= 2){
			returnValue++;	
		}
	}

	return returnValue;
}
 
void GestureRecognizerModified::drawTrajectoryBasedOnVecPoints(vector<Point3i>& originalTrajectory,int rows,int cols,String ImageName,Rect boundingRect){

	//step1: create cv::Mat
	Mat tempMat,trajectoryMat/* = Mat(originalTrajectory)*/;
	 
	tempMat = Mat::zeros(rows*2,cols*2,CV_32FC1);
	//cvtColor(tempMat,tempMat,CV_GRAY2BGR);
	for(int i = 0; i < originalTrajectory.size(); i++){
		circle(tempMat,Point(originalTrajectory[i].x*2,originalTrajectory[i].y*2),1,Scalar(255,255,255),3,8);
		//tempMat.at<float>(originalTrajectory[i].y*4,originalTrajectory[i].x*4) = 255; 
		
	}
	//cout<<"boundingRect"<<boundingRect.width<<","<<boundingRect.height<<boundingRect.tl()<<endl;
	//cout<<"orgTra"<<originalTrajectory.size()<<endl;
	if(boundingRect.height > 0 && boundingRect.width > 0 && boundingRect.tl().x >= 0 &&
		boundingRect.tl().y >= 0 && abs(boundingRect.width - boundingRect.tl().x) < cols
		&& abs(boundingRect.height - boundingRect.tl().y) < rows){
		//Rect gestureBoundingRect = Rect(extreamLeft.x,extreamTop.y,height,width);
		//trajectoryMat = Mat::zeros(boundingRect.height,boundingRect.width,tempMat.type());
		
			Rect boundingRectBig = Rect(boundingRect.tl().x*2,boundingRect.tl().y*2,boundingRect.width*2,boundingRect.height*2);
			tempMat(boundingRectBig ).copyTo(trajectoryMat);
			imshow(ImageName+"trajectoryMat",trajectoryMat);
		
			imshow(ImageName+"tempMat",tempMat);
		//cout<<"orgTra::inside"<<endl;
	}
	 
	
}