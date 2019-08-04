#include "GestureRecognizer.hpp"

GestureRecognizer::GestureRecognizer(){
	//counterForLeft = 0;
	gestureStartStopFlag = false;
	indexOfGesturesGlobal = 0;
	  
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

GestureRecognizer::~GestureRecognizer(){
	 
}

void GestureRecognizer::trainSVMLetterS(){
	//////////////////////////////////////////////////// Letter S Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Letter S
	LetterSTrainingData();
	  
	 // Set up training data
	float labelsLetterS[noOfTrainingSamplesLetterS];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesLetterS ; i6++){
		labelsLetterS[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesLetterS; i7 < noOfTrainingSamplesLetterS ; i7++){
			labelsLetterS[i7] = -1.0;
	}

	 
    Mat labelsMatLetterS(1, noOfTrainingSamplesLetterS, CV_32FC1, labelsLetterS);
	// Set up SVM's parameters
    paramsLetterS.svm_type    = CvSVM::C_SVC;
    paramsLetterS.kernel_type = CvSVM::LINEAR;
    paramsLetterS.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMLetterS.train(trainingDataMatLetterS, labelsMatLetterS, Mat(), Mat(), paramsLetterS);
}
void GestureRecognizer::trainSVMDigit1(){
	//////////////////////////////////////////////////// Digit 1 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 1
	Digit1TrainingData();
	  
	 // Set up training data
	float labelsDigit1[noOfTrainingSamplesDigit1];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit1 ; i6++){
		labelsDigit1[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit1; i7 < noOfTrainingSamplesDigit1; i7++){
			labelsDigit1[i7] = -1.0;
	}

	 
    Mat labelsMatDigit1(1, noOfTrainingSamplesDigit1, CV_32FC1, labelsDigit1);
	// Set up SVM's parameters
    paramsDigit1.svm_type    = CvSVM::C_SVC;
    paramsDigit1.kernel_type = CvSVM::LINEAR;
    paramsDigit1.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit1.train(trainingDataMatDigit1, labelsMatDigit1, Mat(), Mat(), paramsDigit1);
}

void GestureRecognizer::trainSVMDigit2(){
//////////////////////////////////////////////////// Digit 2 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 2
	Digit2TrainingData();
	  
	 // Set up training data
	float labelsDigit2[noOfTrainingSamplesDigit2];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit2 ; i6++){
		labelsDigit2[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit2; i7 < noOfTrainingSamplesDigit2; i7++){
			labelsDigit2[i7] = -1.0;
	}

	 
    Mat labelsMatDigit2(1, noOfTrainingSamplesDigit2, CV_32FC1, labelsDigit2);
	// Set up SVM's parameters
    paramsDigit2.svm_type    = CvSVM::C_SVC;
    paramsDigit2.kernel_type = CvSVM::LINEAR;
    paramsDigit2.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit2.train(trainingDataMatDigit2, labelsMatDigit2, Mat(), Mat(), paramsDigit2);
}

void GestureRecognizer::trainSVMDigit3(){
	//////////////////////////////////////////////////// Digit 3 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 3
	Digit3TrainingData();
	  
	 // Set up training data
	float labelsDigit3[noOfTrainingSamplesDigit3];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit3 ; i6++){
		labelsDigit3[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit3; i7 < noOfTrainingSamplesDigit3; i7++){
			labelsDigit3[i7] = -1.0;
	}

	 
    Mat labelsMatDigit3(1, noOfTrainingSamplesDigit3, CV_32FC1, labelsDigit3);
	// Set up SVM's parameters
    paramsDigit3.svm_type    = CvSVM::C_SVC;
    paramsDigit3.kernel_type = CvSVM::LINEAR;
    paramsDigit3.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit3.train(trainingDataMatDigit3, labelsMatDigit3, Mat(), Mat(), paramsDigit3);
}

void GestureRecognizer::trainSVMDigit4(){
	//////////////////////////////////////////////////// Digit 4 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 4
	Digit4TrainingData();
	  
	 // Set up training data
	float labelsDigit4[noOfTrainingSamplesDigit4];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit4 ; i6++){
		labelsDigit4[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit4; i7 < noOfTrainingSamplesDigit4; i7++){
			labelsDigit4[i7] = -1.0;
	}

	 
    Mat labelsMatDigit4(1, noOfTrainingSamplesDigit4, CV_32FC1, labelsDigit4);
	// Set up SVM's parameters
    paramsDigit4.svm_type    = CvSVM::C_SVC;
    paramsDigit4.kernel_type = CvSVM::LINEAR;
    paramsDigit4.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit4.train(trainingDataMatDigit4, labelsMatDigit4, Mat(), Mat(), paramsDigit4);
}

void GestureRecognizer::trainSVMDigit5(){
	//////////////////////////////////////////////////// Digit 5 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 5
	Digit5TrainingData();
	  
	 // Set up training data
	float labelsDigit5[noOfTrainingSamplesDigit5];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit5 ; i6++){
		labelsDigit5[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit5; i7 < noOfTrainingSamplesDigit5; i7++){
			labelsDigit5[i7] = -1.0;
	}

	 
    Mat labelsMatDigit5(1, noOfTrainingSamplesDigit5, CV_32FC1, labelsDigit5);
	// Set up SVM's parameters
    paramsDigit5.svm_type    = CvSVM::C_SVC;
    paramsDigit5.kernel_type = CvSVM::LINEAR;
    paramsDigit5.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit5.train(trainingDataMatDigit5, labelsMatDigit5, Mat(), Mat(), paramsDigit5);
}

void GestureRecognizer::trainSVMDigit6(){
	//////////////////////////////////////////////////// Digit 6 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 6
	Digit6TrainingData();
	  
	 // Set up training data
	float labelsDigit6[noOfTrainingSamplesDigit6];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit6 ; i6++){
		labelsDigit6[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit6; i7 < noOfTrainingSamplesDigit6; i7++){
			labelsDigit6[i7] = -1.0;
	}

	 
    Mat labelsMatDigit6(1, noOfTrainingSamplesDigit6, CV_32FC1, labelsDigit6);
	// Set up SVM's parameters
    paramsDigit6.svm_type    = CvSVM::C_SVC;
    paramsDigit6.kernel_type = CvSVM::LINEAR;
    paramsDigit6.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit6.train(trainingDataMatDigit6, labelsMatDigit6, Mat(), Mat(), paramsDigit6);
}

void GestureRecognizer::trainSVMDigit7(){
	//////////////////////////////////////////////////// Digit 7 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 7
	Digit7TrainingData();
	  
	 // Set up training data
	float labelsDigit7[noOfTrainingSamplesDigit7];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit7 ; i6++){
		labelsDigit7[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit7; i7 < noOfTrainingSamplesDigit7; i7++){
			labelsDigit7[i7] = -1.0;
	}

	 
    Mat labelsMatDigit7(1, noOfTrainingSamplesDigit7, CV_32FC1, labelsDigit7);
	// Set up SVM's parameters
    paramsDigit7.svm_type    = CvSVM::C_SVC;
    paramsDigit7.kernel_type = CvSVM::LINEAR;
    paramsDigit7.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit7.train(trainingDataMatDigit7, labelsMatDigit7, Mat(), Mat(), paramsDigit7);
}

void GestureRecognizer::trainSVMDigit8(){
	//////////////////////////////////////////////////// Digit 8 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 8
	Digit8TrainingData();
	  
	 // Set up training data
	float labelsDigit8[noOfTrainingSamplesDigit8];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit8 ; i6++){
		labelsDigit8[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit8; i7 < noOfTrainingSamplesDigit8; i7++){
			labelsDigit8[i7] = -1.0;
	}

	 
    Mat labelsMatDigit8(1, noOfTrainingSamplesDigit8, CV_32FC1, labelsDigit8);
	// Set up SVM's parameters
    paramsDigit8.svm_type    = CvSVM::C_SVC;
    paramsDigit8.kernel_type = CvSVM::LINEAR;
    paramsDigit8.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit8.train(trainingDataMatDigit8, labelsMatDigit8, Mat(), Mat(), paramsDigit8);
}

void GestureRecognizer::trainSVMDigit9(){
	//////////////////////////////////////////////////// Digit 9 Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Digit 9
	Digit9TrainingData();
	  
	 // Set up training data
	float labelsDigit9[noOfTrainingSamplesDigit9];
	//intialize vec with +ve sample label == 1.0
	for(int i6 = 0; i6 < noOfPVeTrainingSamplesDigit9; i6++){
		labelsDigit9[i6] = 1.0;
	}
	for(int i7 = noOfPVeTrainingSamplesDigit9; i7 < noOfTrainingSamplesDigit9; i7++){
			labelsDigit9[i7] = -1.0;
	}

	 
    Mat labelsMatDigit9(1, noOfTrainingSamplesDigit9, CV_32FC1, labelsDigit9);
	// Set up SVM's parameters
    paramsDigit9.svm_type    = CvSVM::C_SVC;
    paramsDigit9.kernel_type = CvSVM::LINEAR;
    paramsDigit9.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
	SVMDigit9.train(trainingDataMatDigit9, labelsMatDigit9, Mat(), Mat(), paramsDigit9);
}

void GestureRecognizer::trainSVMCircle(){
	//////////////////////////////////////////////////// Circle Taining Phase //////////////////////////////////////////////

	//SVM training phase-> Circe
	circleTrainingData();
	  
	 // Set up training data
	float labelsCircle[noOfTrainingSamplesCircle];
	//intialize vec with +ve sample label == 1.0
	for(int i5 = 0; i5 < noOfPVeTrainingSamplesCircle ; i5++){
		labelsCircle[i5] = 1.0;
	}
	for(int i6 = noOfPVeTrainingSamplesCircle; i6 < noOfTrainingSamplesCircle ; i6++){
			labelsCircle[i6] = -1.0;
	}

	 
    Mat labelsMatCircle(1, noOfTrainingSamplesCircle, CV_32FC1, labelsCircle);
	// Set up SVM's parameters
    paramsCircle.svm_type    = CvSVM::C_SVC;
    paramsCircle.kernel_type = CvSVM::LINEAR;
    paramsCircle.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    SVMCircle.train(trainingDataMatCircle, labelsMatCircle, Mat(), Mat(), paramsCircle);
}
void GestureRecognizer::storeVectors(float xW,float yW, float zW){
	
	 
		/*xVector.push_back(xW);
		yVector.push_back(yW);
		zVector.push_back(zW);*/
   
}
String GestureRecognizer::checkForGesture(vector<Point3i>& srcVector){
 
		
		bool xStatus = false;
		xStatus = ifVectorValuesAreInRange(srcVector);
		bool yStatus = false;
		yStatus = ifVectorValuesAreInRange(srcVector);
		bool zStatus = false;
		zStatus = ifVectorValuesAreInRange(srcVector);
		 
		 
		if(xStatus && yStatus ){
			/*if(gestureStartStopFlag == false){
				gestureStartStopFlag = true;
				return "Start";
			} else {
				gestureStartStopFlag = false;
				return "Stop";
			}*/
			return "Halt";
		}	 	
	  return "Moving";
}

bool GestureRecognizer::ifVectorValuesAreInRange(vector<Point3i>& srcVector){
	float size = srcVector.size();
	vector<float> result;
	// 
	//if (srcVector.begin()!=srcVector.end()) {
 //   typename iterator_traits<InputIterator>::value_type val,prev;
 //   *result = prev = *srcVector.begin();
 //   while (++first!=last) {
 //     val = *first;
 //     *++result = val - prev;  // or: *++result = binary_op(val,prev)
 //     prev = val;
 //   }
 //   ++result;
 // }
 // return result;

	int elementsOutOfRange = findAdjacentDifference (srcVector);

	// if at least 5 or less elements are satisfying the condition, then hand is not moving ie its stable...
	if(elementsOutOfRange < 2 ){
		return true;
	}
	//float sum = std::accumulate(result.begin(),result.end(),0);
	//float avg =  sum / result.size();

	/*if(avg < range ){
		return true;
	}*/

	return false;

}
 
int GestureRecognizer::findAdjacentDifference(vector<Point3i>& srcVector){

	int outOfRange = 0;
	int i = 0;
	int iEnd = srcVector.size()-1;
	for(;i < iEnd ; i++){	
		float diff = abs(abs(srcVector[i].x) - abs(srcVector[i+1].x));
		// if diff is bigger than RANGE, that means the hand is moving, ie its not steady...
		if(diff > RANGE){
			outOfRange++;

			//resultVector.push_back(diff);
		}
	}
	return outOfRange;
}
int GestureRecognizer::findMedianOfVector(vector<int>& srcVector){
	size_t size = srcVector.size();
	if(size > 0){
		sort(srcVector.begin(), srcVector.end());
		if (size  % 2 == 0){
			return (srcVector[size / 2 - 1] + srcVector[size / 2]) / 2;
		}else {
			return srcVector[size / 2];
		}
	}
}

void GestureRecognizer::findEdgePointsOnTrajectory(vector<Point3i>& srcVector,Point& XL,Point& XH,Point& YL,Point& YH,Rect& returnRectAroundGesture,
	vector<int>& angleReturnedVec,vector<int>& returnReducedVector){
	
	//initialize global variable trajectory for gesture understanding..
	 

	XL.x = 180;
	XL.y = 180;
	XH.x = 0;
	XH.y = 0;

	YL.x = 120;
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

	width = abs(XL.x-XH.x);
	height = abs(YL.y-YH.y);
	returnRectAroundGesture =  Rect(XL.x,YL.y,width,height);
	//initialize all extream points based on XL,XH,YL,YH
	extreamLeft.x = XL.x;	extreamLeft.y = XL.y;
	extreamRight.x = XH.x;	extreamRight.y = XH.y;
	extreamTop.x = YL.x;	extreamTop.y = YL.y;
	extreamBottom.x = YH.x;	extreamBottom.y = YH.y;

	//initialize some variables here..
	startPoint = srcVector[0];
	stopPoint = srcVector[srcVector.size()-1];


	findAngleVector(srcVector);

	angleReturnedVec = angleVector;
	//removeDuplicateElements(angleVector,returnReducedVector);

	returnReducedVector = angleVector; 
	
	closenessCriteria = sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0));	//it checks if the start point and stop point are close enough to say its a close gesture or not?
							//sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20


	 
}
	 
void GestureRecognizer::removeDuplicateElements(vector<int>& srcVector,vector<int>& returnVector){
	int currentEle = srcVector[0];
	returnVector.push_back(currentEle);
	for(int i = 1; i < srcVector.size(); i++){
		
		if(currentEle != srcVector[i]){
			returnVector.push_back(srcVector[i]);
			currentEle  = srcVector[i];
		}
	}
}

void GestureRecognizer::findAngleVector(vector<Point3i>& srcVector){
	vector<double> tempVector;
	int size = (srcVector.size() - 1); //untill second last element
	for(int i = 0; i < size; i++){
		tempVector.push_back(findAngleBetweenTwoPoints(srcVector[i],srcVector[i+1]));
	}

	//claer angleVector before using it..
	if(angleVector.size() != 0 )
	angleVector.clear();

	//normalize angle vector with 45
	for(int j = 0 ; j < tempVector.size(); j++){
		int angle =  tempVector[j];
		if(angle == 360){
			angle = 0;
		}
		angleVector.push_back(angle/22.5);
	}


}
 
template <typename Fwd>
typename std::map<typename std::iterator_traits<Fwd>::value_type, int>::value_type
GestureRecognizer::most_frequent_element(Fwd begin, Fwd end)
{
    std::map<typename std::iterator_traits<Fwd>::value_type, int> count;

    for (Fwd it = begin; it != end; ++it)
        ++count[*it];

    return *std::max_element(count.begin(), count.end(), by_second());
}

//geture 1: Move Up Rule..
//Rule 1: if width is half the size of height 
bool GestureRecognizer::detectUp(vector<Point3i>& srcVector,String& returnString){
	if(startPoint.y > stopPoint.y){		//if start point is below stop point
			 
				//std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());

				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 2 (90') occurs multiple times that means its upward direction..
				if( x.first == 3 || x.first == 4 || x.first == 5) {
					returnString =  "upward";
					return true;			
				}
		 
		}
	 
	return false;
}

//used for Digit4..
bool GestureRecognizer::detectUp(vector<Point3i>& srcVector){
				//std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());

				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 2 (90') occurs multiple times that means its upward direction..
				if( x.first == 3 || x.first == 4 || x.first == 5 || x.second == 3) {
					return true;			
				}
	 
	return false;
}

//geture 2: Move Down Rule..
//Rule 1: if width is half the size of height 
bool GestureRecognizer::detectDown(vector<Point3i>& srcVector,String& returnString){
	if(startPoint.y < stopPoint.y){		//if start point is above stop point
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 6 (270') occurs multiple times that means its upward direction..
				if( x.first == 11 || x.first == 12 || x.first == 13) {
					returnString = "downward";			
					return true;
				}
		}
	 
	return false;
}

//Used for Digit1 finding..
bool GestureRecognizer::detectDown(vector<Point3i>& srcVector){
	 
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 6 (270') occurs multiple times that means its upward direction..
				if( x.first == 11 || x.first == 12 || x.first == 13 || x.second == 3) {
					return true;
				}
	 
	return false;
}

//geture 3: Move Left Rule..
//Rule 1: if height is half the size of width
bool GestureRecognizer::detectLeft(vector<Point3i>& srcVector,String& returnString){
	if(startPoint.x > stopPoint.x){		//if start point is right of stop point
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 4 (180') occurs multiple times that means its upward direction..
				if( x.first == 7 || x.first == 8 || x.first == 9) {
					returnString = "left";			
					return true;
				}
		}
	 
	return false;
}

//used for Digit 5
bool GestureRecognizer::detectLeft(vector<Point3i>& srcVector){
	 
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 4 (180') occurs multiple times that means its upward direction..
				if( x.first == 7 || x.first == 8 || x.first == 9 || x.second == 3) {		
					return true;
				} 
	return false;
}

//geture 4: Move Right Rule..
//Rule 1: if height is half the size of width
bool GestureRecognizer::detectRight(vector<Point3i>& srcVector,String& returnString){
	if(startPoint.x < stopPoint.x){		//if start point is left of stop point
			 
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());

				//if 0 (0') occurs multiple times that means its upward direction..
				if( x.first == 1 || x.first == 0 || x.first == 15) {
					returnString = "right";	
					return true;
				}
			 
		}
 
	return false;
}

//used for Digit 2..
bool GestureRecognizer::detectRight(vector<Point3i>& srcVector){
	 
				std::pair<int, int> x = most_frequent_element(angleVector.begin(), angleVector.end());
				//if 0 (0') occurs multiple times that means its upward direction..
				if( x.first == 1 || x.first == 0 || x.first == 15 || x.second == 3) {
					return true;
				}
		 
	return false;
}

//geture 5: Detect Circle..
bool GestureRecognizer::detectCircle(vector<int>& returnReducedVector,String& returnString){
	//gesture 5: Circle
	 
	//Rule1: start point and stop points should be close enough if not same.!
	if( sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 25){

		vector<Mat> rulesForCircle;
		
		Mat firstElementRule,secondElementRule,thirdElementRule,fourthElementRule,
			fifthElementRule,sixthElementRule,sevenththElementRule,eigthElementRule,ninthElementRule,
			tenthElementRule,eleventhElementRule;
		
		firstElementRule.create(4,1,CV_8U);secondElementRule.create(4,1,CV_8U);thirdElementRule.create(4,1,CV_8U);
		fourthElementRule.create(4,1,CV_8U);fifthElementRule.create(4,1,CV_8U);sixthElementRule.create(5,1,CV_8U);
		sevenththElementRule.create(7,1,CV_8U);eigthElementRule.create(7,1,CV_8U);ninthElementRule.create(7,1,CV_8U);
		tenthElementRule.create(7,1,CV_8U);eleventhElementRule.create(7,1,CV_8U);
		
		firstElementRule.at<uchar>(0,0) = 0;	firstElementRule.at<uchar>(1,0) = 1;	firstElementRule.at<uchar>(2,0) = 2;	firstElementRule.at<uchar>(3,0) = 3; 
		secondElementRule.at<uchar>(0,0) = 0;	secondElementRule.at<uchar>(1,0) = 1;	secondElementRule.at<uchar>(2,0) = 2;	secondElementRule.at<uchar>(3,0) = 15;	  
		thirdElementRule.at<uchar>(0,0) = 15;	thirdElementRule.at<uchar>(1,0) = 14;	thirdElementRule.at<uchar>(2,0) = 13;	thirdElementRule.at<uchar>(3,0) = 0; 
		fourthElementRule.at<uchar>(0,0) = 11;	fourthElementRule.at<uchar>(1,0) = 12;	fourthElementRule.at<uchar>(2,0) = 13;	fourthElementRule.at<uchar>(3,0) = 14;  
		fifthElementRule.at<uchar>(0,0) = 9;	fifthElementRule.at<uchar>(1,0) = 10;	fifthElementRule.at<uchar>(2,0) = 11;	fifthElementRule.at<uchar>(3,0) = 12; 
		sixthElementRule.at<uchar>(0,0) = 7;	sixthElementRule.at<uchar>(1,0) = 8;	sixthElementRule.at<uchar>(2,0) = 9;	sixthElementRule.at<uchar>(3,0) = 10;		
		sixthElementRule.at<uchar>(4,0) = 11; 
		
		sevenththElementRule.at<uchar>(0,0) = 4;sevenththElementRule.at<uchar>(1,0) = 5;sevenththElementRule.at<uchar>(2,0) = 6;sevenththElementRule.at<uchar>(3,0) = 7;	
		sevenththElementRule.at<uchar>(4,0) = 8;	sevenththElementRule.at<uchar>(5,0) = 9;	sevenththElementRule.at<uchar>(6,0) = 10;   
		
		eigthElementRule.at<uchar>(0,0) = 3;	eigthElementRule.at<uchar>(1,0) = 4;	eigthElementRule.at<uchar>(2,0) = 5;	eigthElementRule.at<uchar>(3,0) = 6;
		eigthElementRule.at<uchar>(4,0) = 7;	eigthElementRule.at<uchar>(5,0) = 8;	eigthElementRule.at<uchar>(6,0) = 9;

		ninthElementRule.at<uchar>(0,0) = 1;	ninthElementRule.at<uchar>(1,0) = 2;	ninthElementRule.at<uchar>(2,0) = 3;	ninthElementRule.at<uchar>(3,0) = 4;
		ninthElementRule.at<uchar>(4,0) = 5;	ninthElementRule.at<uchar>(5,0) = 6;	ninthElementRule.at<uchar>(6,0) = 7; 

		tenthElementRule.at<uchar>(0,0) = 1;	tenthElementRule.at<uchar>(1,0) = 2;	tenthElementRule.at<uchar>(2,0) = 3;	tenthElementRule.at<uchar>(3,0) = 4;
		tenthElementRule.at<uchar>(4,0) = 5;	tenthElementRule.at<uchar>(5,0) = 6;	tenthElementRule.at<uchar>(6,0) = 7;

		eleventhElementRule.at<uchar>(0,0) = 1;	eleventhElementRule.at<uchar>(1,0) = 2;	eleventhElementRule.at<uchar>(2,0) = 3;	eleventhElementRule.at<uchar>(3,0) = 4;
		eleventhElementRule.at<uchar>(4,0) = 5;	eleventhElementRule.at<uchar>(5,0) = 6;	eleventhElementRule.at<uchar>(6,0) = 7;
		 
		rulesForCircle.push_back(firstElementRule);
		rulesForCircle.push_back(secondElementRule);
		rulesForCircle.push_back(thirdElementRule);
		rulesForCircle.push_back(fourthElementRule);
		rulesForCircle.push_back(fifthElementRule);
		rulesForCircle.push_back(sixthElementRule);
		rulesForCircle.push_back(sevenththElementRule);
		rulesForCircle.push_back(eigthElementRule);
		rulesForCircle.push_back(ninthElementRule);
		rulesForCircle.push_back(tenthElementRule);
		rulesForCircle.push_back(eleventhElementRule);


		int counter = 0;
		int sizeOfReducedVec =  returnReducedVector.size();
		int sizeOfRuleVec =  rulesForCircle.size();

		for(int indx = 0; indx <  sizeOfReducedVec; indx++){
			if(indx < sizeOfRuleVec){
				for(int indxRule = 0; indxRule < rulesForCircle[indx].rows ; indxRule++){
					if(returnReducedVector[indx] == rulesForCircle[indx].at<uchar>(indxRule,0)){
						counter++;
							//return "circle";
					}
				}
			}
		} 
		
		if(counter >= sizeOfReducedVec*0.9){
			returnString = "circle";
			return true;
		}
	} //Rule1 condition end..
	return false;

}


//this method is with non repetition sequence..
//void GestureRecognizer::circleTrainingData(){
//	   
//	copyArrayDataToVec();
//	float trainingDataTotal[noOfTrainingSamplesCircle][50] ={
//		{ 2, 1, 0, 15, 13, 12, 11, 9, 8, 7, 8, 4, 3, 2, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 10, 9, 7, 8, 6, 7, 5, 4, 3, 0 },
//		{ 3, 2, 1, 0, 15, 14, 13, 12, 11, 12, 10, 8, 11, 8, 9, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 13, 12, 11, 10, 9, 8, 6, 4, 3, 0 },
//		{ 1, 0, 13, 12, 11, 12, 8, 6, 4, 3, 0 },
//		{ 3, 1, 0, 14, 13, 12, 11, 8, 9, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 0, 15, 14, 13, 12, 13, 12, 11, 9, 10, 9, 8, 7, 8, 7, 6, 4, 0 },
//		{ 2, 1, 2, 1, 0, 15, 14, 12, 13, 12, 11, 10, 9, 8, 7, 8, 7, 6, 7, 4, 3, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 12, 11, 10, 9, 6, 8, 7, 6, 5, 4, 3, 0 },
//		{ 4, 2, 3, 2, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0 },
//		{ 2, 3, 2, 3, 0, 14, 13, 12, 11, 10, 8, 7, 5, 4, 0 },
//		{ 2, 1, 15, 14, 12, 10, 9, 8, 7, 5, 2, 0 },
//		{ 3, 1, 0, 15, 13, 12, 11, 9, 8, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 15, 13, 12, 10, 9, 7, 4, 3, 0 },
//		{ 3, 2, 1, 15, 14, 12, 11, 8, 7, 6, 5, 4, 3, 0 },
//		{ 3, 1, 0, 13, 12, 11, 9, 8, 4, 3, 0 },
//		{ 2, 0, 13, 12, 10, 9, 6, 4, 3, 0 },
//		{ 2, 0, 15, 13, 12, 11, 9, 8, 7, 5, 3, 4, 0 },
//		{ 3, 2, 0, 15, 13, 12, 11, 10, 8, 7, 6, 4, 0 },
//		{ 3, 2, 1, 0, 14, 13, 12, 11, 12, 10, 8, 6, 5, 4, 2, 0 },
//		{ 2, 1, 15, 13, 12, 11, 9, 7, 6, 5, 4, 0 },
//		{ 3, 2, 0, 14, 13, 12, 11, 10, 8, 7, 6, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 10, 8, 7, 6, 7, 5, 4, 3, 0 },
//		{ 3, 2, 1, 0, 15, 13, 12, 10, 9, 8, 7, 6, 5, 4, 3, 0 },
//		{ 3, 1, 15, 13, 11, 10, 8, 7, 6, 4, 3, 0 },
//		{ 4, 3, 1, 0, 15, 14, 13, 12, 11, 8, 7, 6, 4, 0 },
//		{ 3, 2, 1, 0, 15, 14, 12, 10, 11, 9, 8, 7, 5, 4, 3, 2, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 9, 7, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 9, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 15, 14, 13, 12, 10, 11, 9, 7, 6, 4, 3, 0 },
//		{ 4, 2, 1, 14, 13, 12, 11, 9, 8, 5, 4, 3, 0 },
//		{ 4, 3, 2, 0, 15, 12, 11, 9, 7, 8, 6, 5, 4, 3, 0 },
//		{ 3, 2, 0, 14, 13, 12, 11, 10, 8, 7, 5, 4, 0 },
//		{ 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 0 },
//		{ 2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 5, 4, 0 },
//		{ 2, 3, 2, 1, 15, 13, 12, 11, 9, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 15, 14, 12, 11, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 0, 14, 13, 12, 11, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 15, 13, 12, 11, 9, 8, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 6, 5, 4, 2, 0 },
//		{ 2, 1, 0, 15, 13, 12, 10, 9, 7, 5, 4, 3, 0 },
//		{ 3, 2, 0, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4, 0 },
//		{ 2, 0, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 0 },
//		{ 1, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0 },
//		{ 3, 2, 3, 1, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0 },
//		{ 1, 2, 1, 0, 14, 13, 12, 11, 8, 9, 8, 7, 6, 5, 4, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 8, 7, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 10, 8, 7, 8, 6, 5, 4, 3, 0 },
//		{ 3, 1, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0 },
//		{ 3, 2, 0, 14, 13, 12, 11, 10, 8, 6, 4, 3, 0 },
//		{ 3, 1, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 12, 10, 8, 6, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 10, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 0, 15, 14, 13, 12, 10, 9, 8, 7, 6, 5, 4, 2, 0 },
//		{ 3, 1, 0, 15, 13, 12, 10, 9, 8, 5, 6, 4, 0 },
//		{ 4, 3, 2, 1, 0, 15, 14, 13, 12, 10, 9, 8, 7, 6, 0 },
//		{ 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 7, 8, 7, 6, 5, 3, 0 },
//		{ 2, 3, 2, 1, 0, 14, 13, 12, 11, 10, 9, 7, 8, 7, 6, 5, 4, 3, 0 },
//		{ 4, 3, 2, 1, 0, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 0 },
//		{ 2, 3, 0, 15, 13, 12, 11, 10, 8, 7, 4, 3, 0 },
//		{ 3, 2, 1, 15, 13, 12, 11, 9, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 12, 11, 10, 8, 6, 5, 4, 3, 2, 0 },
//		{ 2, 1, 0, 15, 13, 12, 10, 11, 9, 8, 7, 5, 4, 3, 0 },
//		{ 3, 2, 0, 15, 13, 12, 10, 9, 6, 7, 4, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 10, 9, 7, 4, 3, 0 },
//		{ 3, 1, 15, 13, 12, 11, 10, 9, 7, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 12, 11, 10, 8, 6, 7, 5, 4, 3, 0 },
//		{ 3, 2, 0, 14, 13, 12, 11, 10, 8, 7, 5, 4, 3, 2, 0 },
//		{ 3, 2, 1, 15, 14, 12, 11, 10, 7, 5, 4, 3, 4, 0 },
//		{ 3, 1, 0, 14, 13, 12, 10, 8, 7, 5, 4, 3, 2, 0 },
//		{ 4, 2, 0, 15, 14, 13, 12, 10, 12, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 0, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 0 },
//		{ 1, 15, 14, 12, 11, 10, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 15, 13, 12, 11, 10, 11, 8, 5, 4, 2, 0 },
//		{ 3, 2, 0, 14, 13, 11, 8, 7, 6, 4, 3, 0 },
//		{ 2, 0, 14, 13, 12, 11, 10, 9, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 0, 14, 13, 11, 10, 8, 7, 6, 4, 3, 0 },
//		{ 2, 1, 15, 13, 12, 10, 9, 8, 7, 4, 3, 0 },
//		{ 3, 0, 15, 0, 13, 12, 10, 9, 7, 6, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 11, 10, 9, 7, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 12, 13, 11, 10, 9, 8, 6, 5, 4, 3, 2, 0 },
//		{ 2, 1, 15, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 9, 8, 6, 4, 3, 0 },
//		{ 2, 0, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 12, 10, 8, 7, 5, 4, 3, 2, 0 },
//		{ 1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 0 },
//		{ 1, 15, 14, 13, 12, 10, 9, 7, 5, 4, 3, 0 },
//		{ 4, 3, 0, 3, 15, 13, 12, 11, 9, 7, 5, 4, 0 },
//		{ 3, 2, 1, 0, 15, 13, 12, 11, 10, 11, 7, 8, 6, 5, 4, 3, 0 },
//		{ 3, 2, 0, 13, 12, 10, 9, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 12, 11, 9, 8, 6, 5, 4, 0 },
//		{ 2, 0, 15, 14, 12, 10, 9, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 12, 10, 9, 7, 5, 4, 3, 0 },
//		{ 3, 2, 0, 14, 13, 12, 10, 9, 7, 6, 5, 3, 0 },
//		{ 3, 2, 0, 14, 13, 12, 10, 9, 7, 6, 5, 3, 0 },
//		{ 3, 2, 0, 15, 14, 12, 11, 9, 8, 6, 5, 4, 0 },
//		{ 3, 2, 1, 0, 15, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 13, 12, 11, 10, 7, 6, 4, 3, 0 },
//		{ 4, 2, 1, 15, 13, 12, 10, 8, 7, 5, 4, 3, 0 },
//		{ 3, 4, 3, 2, 1, 0, 15, 13, 12, 11, 10, 8, 9, 8, 6, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 0, 15, 14, 13, 12, 11, 10, 8, 6, 5, 4, 3, 0 },
//		{ 2, 15, 14, 13, 12, 11, 10, 9, 8, 6, 4, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 15, 13, 12, 11, 9, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 12, 11, 10, 7, 8, 5, 4, 0 },
//		{ 2, 1, 0, 15, 13, 11, 10, 8, 9, 6, 4, 5, 4, 3, 0 },
//		{ 3, 2, 0, 2, 15, 13, 12, 11, 10, 8, 7, 4, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 10, 8, 7, 4, 7, 4, 5, 4, 2, 0 },
//		{ 2, 15, 14, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 0 },
//		{ 2, 1, 15, 13, 12, 11, 10, 9, 8, 7, 5, 4, 3, 0 },
//		{ 3, 1, 0, 15, 14, 13, 12, 11, 10, 8, 7, 4, 6, 4, 3, 0 },
//		{ 2, 1, 14, 13, 12, 10, 8, 5, 4, 3, 0 },
//		{ 2, 0, 1, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3, 0 },
//		{ 3, 1, 2, 15, 13, 12, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 0, 1, 14, 15, 13, 12, 10, 11, 9, 7, 6, 5, 3, 0 },
//		{ 2, 1, 0, 14, 13, 12, 10, 9, 7, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 10, 6, 7, 5, 7, 5, 4, 3, 2, 0 },
//		{ 2, 0, 15, 14, 13, 12, 11, 10, 8, 6, 5, 4, 3, 0 },
//		{ 2, 0, 15, 13, 12, 11, 10, 8, 6, 5, 4, 2, 0 },
//		{ 2, 1, 15, 13, 12, 11, 10, 9, 7, 6, 4, 5, 4, 0 },
//		{ 4, 1, 2, 15, 13, 12, 10, 9, 6, 8, 6, 4, 0 },
//		{ 2, 1, 15, 14, 12, 11, 10, 9, 8, 7, 5, 4, 3, 0 },
//		{ 3, 1, 2, 14, 0, 14, 12, 11, 10, 8, 6, 5, 4, 0 },
//		{ 2, 15, 13, 12, 10, 9, 7, 8, 6, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 13, 11, 10, 9, 6, 8, 6, 5, 4, 0 },
//		{ 2, 1, 0, 14, 12, 13, 11, 10, 9, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 14, 11, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 3, 2, 1, 0, 14, 15, 13, 15, 12, 13, 12, 11, 10, 11, 8, 9, 8, 7, 8, 6, 5, 4, 3, 0 },
//		{ 1, 0, 1, 0, 15, 13, 14, 13, 12, 11, 10, 9, 7, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 14, 13, 0, 13, 12, 11, 12, 10, 7, 9, 8, 7, 8, 7, 6, 4, 3, 0 },
//		{ 1, 0, 15, 13, 12, 11, 10, 9, 7, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 13, 0, 13, 12, 11, 12, 10, 8, 10, 8, 7, 8, 7, 6, 4, 3, 0 },
//		{ 3, 2, 1, 0, 15, 0, 14, 12, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 0 },
//		{ 1, 0, 15, 14, 12, 11, 9, 10, 8, 7, 6, 5, 4, 3, 0 },
//		{ 2, 0, 15, 14, 12, 11, 10, 9, 10, 7, 6, 4, 5, 4, 0 },
//		{ 2, 0, 15, 14, 13, 12, 11, 10, 7, 8, 5, 4, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 9, 8, 7, 6, 4, 0 },
//		{ 3, 2, 1, 0, 15, 13, 12, 11, 8, 7, 5, 4, 0 },
//		{ 2, 1, 0, 15, 13, 11, 10, 9, 8, 7, 5, 4, 0 },
//		{ 2, 3, 1, 15, 13, 12, 11, 12, 9, 8, 6, 4, 0 },
//		{ 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0 },
//		{ 3, 1, 2, 0, 15, 14, 15, 13, 11, 12, 10, 9, 8, 7, 6, 5, 4, 0 },
//		{ 2, 1, 0, 14, 13, 11, 10, 9, 7, 5, 4, 0 },
//		{ 2, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0 },
//		{ 1, 0, 13, 12, 11, 10, 9, 7, 6, 4, 3, 0 },
//		{ 2, 3, 1, 0, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0 },
//		{ 2, 1, 0, 13, 12, 10, 8, 6, 5, 4, 3, 0 },
//		{ 2, 1, 0, 15, 14, 13, 12, 10, 9, 10, 7, 6, 4, 3, 4, 0 },
//		{ 2, 3, 1, 0, 15, 14, 12, 11, 10, 8, 7, 6, 4, 3, 0 },
//		{ 2, 14, 15, 14, 13, 11, 10, 8, 7, 6, 4, 5, 3, 0 },
//		{ 1, 15, 14, 13, 12, 11, 9, 5, 3, 0 },
//		{ 3, 2, 0, 15, 13, 12, 10, 12, 10, 8, 6, 8, 5, 0 },
//		{ 2, 0, 14, 15, 13, 11, 10, 9, 6, 5, 4, 0 }, //153th +ve data circle
//
//		 
//		{ 0, 15, 0, 14, 0, 3, 4, 6, 8, 9, 12, 15, 0, 14, 15, 0, 10, 9, 8, 7, 0 },	//random -ve samples 0th
//		{ 3, 13, 12, 13, 12, 2, 3, 2, 3, 2, 13, 12, 0 },
//		{ 3, 12, 11, 12, 11, 5, 4, 3, 0, 13, 12, 0 },
//		{ 8, 4, 13, 3, 1, 0, 1, 7, 8, 7, 9, 4, 2, 1, 0, 15, 8, 0 },
//		{ 3, 4, 3, 13, 12, 13, 12, 4, 3, 4, 3, 2, 3, 2, 0, 13, 12, 8, 4, 0 },
//		{ 2, 1, 0, 1, 2, 12, 11, 13, 12, 8, 6, 8, 7, 8, 0 },
//		{ 2, 3, 2, 3, 12, 11, 1, 0, 15, 0, 3, 4, 5, 12, 0 },
//		{ 4, 3, 13, 12, 13, 12, 3, 4, 3, 4, 0, 13, 12, 13, 12, 0 },
//		{ 4, 3, 0, 12, 4, 6, 8, 9, 8, 9, 7, 0, 1, 0, 14, 0 },
//		{ 7, 8, 9, 8, 13, 12, 2, 3, 4, 3, 4, 15, 13, 12, 11, 12, 5, 0 },
//		{ 5, 4, 8, 10, 8, 0, 12, 2, 1, 0, 14, 10, 11, 5, 6, 8, 7, 8, 0 },
//		{ 4, 5, 6, 8, 10, 11, 12, 13, 14, 1, 3, 4, 5, 4, 11, 12, 11, 10, 1, 0 },
//		{ 4, 2, 14, 11, 3, 1, 8, 9, 8, 9, 11, 10, 11, 0 },
//		{ 15, 14, 0, 5, 8, 9, 10, 1, 0, 15, 13, 12, 11, 3, 4, 3, 12, 10, 0 },
//		{ 12, 8, 7, 8, 6, 0, 3, 0, 2, 1, 0, 11, 12, 0 },
//		{ 4, 5, 7, 8, 10, 12, 3, 14, 4, 11, 0, 3, 2, 1, 0 },
//		{ 4, 5, 6, 8, 9, 10, 12, 10, 0, 4, 0, 15, 13, 11, 0 },
//		{ 7, 8, 9, 12, 1, 15, 14, 0, 3, 4, 5, 6, 8, 9, 8, 0 },
//		{ 7, 8, 0, 1, 0, 15, 0, 11, 12, 11, 12, 11, 6, 9, 3, 7, 8, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 10, 9, 0 },
//		{ 2, 1, 0, 15, 14, 13, 12, 11, 10, 2, 4, 5, 8, 0 },
//		{ 1, 0, 15, 13, 12, 4, 3, 4, 0 },
//		{ 2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12, 0 },
//		{ 4, 3, 0, 14, 13, 14, 13, 12, 3, 4, 3, 4, 0, 14, 12, 13, 12, 13, 12, 11, 0 },
//		{ 0, 15, 0, 2, 7, 8, 6, 2, 0, 1, 0, 15, 0 },
//		{ 15, 14, 0, 2, 3, 2, 3, 4, 5, 7, 8, 9, 8, 9, 10, 1, 0 },
//		{ 4, 3, 4, 1, 15, 14, 13, 12, 13, 12, 13, 12, 4, 5, 6, 9, 10, 12, 11, 10, 0 },
//		{ 15, 0, 7, 8, 7, 8, 7, 9, 0 },
//		{ 4, 3, 4, 3, 4, 12, 11, 12, 0 },
//		{ 2, 10, 11, 0 },
//		{ 4, 3, 2, 0, 14, 13, 12, 11, 13, 12, 8, 7, 8, 0 },
//		{ 2, 3, 1, 0, 14, 13, 12, 11, 3, 4, 3, 8, 9, 10, 11, 0 },
//		{ 2, 1, 0, 14, 13, 12, 11, 4, 3, 2, 1, 0, 15, 14, 13, 11, 12, 0 },
//		{ 2, 1, 0, 14, 12, 11, 0, 1, 0, 14, 12, 0 },
//		{ 0, 14, 0, 3, 4, 12, 11, 12, 1, 0, 12, 13, 11, 9, 0 },
//		{ 4, 10, 9, 10, 12, 13, 12, 14, 13, 0, 1, 0 },
//		{ 4, 3, 15, 13, 12, 4, 9, 11, 12, 11, 10, 11, 10, 11, 9, 0 },
//		{ 4, 3, 4, 3, 4, 7, 8, 9, 11, 12, 14, 0, 4, 3, 2, 0, 13, 12, 11, 10, 0 },
//		{ 3, 4, 8, 10, 12, 4, 3, 1, 0, 13, 12, 0 },
//		{ 4, 2, 1, 0, 14, 13, 11, 12, 11, 0 },
//		{ 4, 5, 6, 8, 9, 10, 11, 12, 0 },
//		{ 14, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 },
//		{ 13, 14, 15, 0, 2, 4, 0 },
//		{ 3, 4, 12, 11, 12, 11, 0 },
//		{ 4, 12, 11, 0 },		//45th -ve random data
//
//
//	};
//
//	Mat trainingDataMatTemp(noOfTrainingSamplesCircle, 50, CV_32FC1, trainingDataTotal);
//	trainingDataMatTemp.copyTo(trainingDataMatCircle);
//}


void GestureRecognizer::circleTrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesCircle][50] = {	{2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 9, 9, 8, 8, 7, 8, 4, 4, 4, 3, 2},
		{2, 2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 10, 9, 9, 7, 8, 6, 7, 5, 4, 4, 4, 3, 3, 3},
		{3, 3, 2, 1, 1, 0, 15, 14, 13, 12, 12, 11, 12, 10, 8, 11, 8, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3},
		{2, 1, 0, 13, 12, 11, 10, 9, 8, 6, 4, 3, 3 },
		{1, 0, 0, 13, 12, 12, 12, 11, 12, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		{3, 3, 1, 0, 0, 14, 13, 13, 12, 12, 11, 11, 8, 9, 7, 6, 6, 4, 4, 4, 3},
		{3, 2, 1, 1, 0, 0, 15, 15, 14, 13, 12, 13, 12, 11, 11, 9, 10, 9, 9, 8, 7, 8, 8, 7, 6, 4, 4, 4, 4, 4},
		{2, 1, 2, 1, 0, 0, 0, 15, 14, 14, 12, 13, 12, 12, 11, 11, 11, 10, 9, 9, 8, 7, 8, 8, 7, 6, 7, 4, 4, 4, 4, 3, 4, 3, 3},
		{2, 2, 1, 0, 0, 15, 14, 14, 12, 12, 12, 11, 11, 10, 10, 10, 9, 6, 8, 7, 7, 6, 5, 4, 4, 4, 3, 3, 3 }, 					 
		{4, 2, 3, 3, 2, 0, 0, 14, 13, 13, 12, 12, 12, 12, 11, 10, 9, 8, 8, 8, 7, 6, 5, 4}, 
		{2, 3, 2, 3, 0, 14, 13, 13, 12, 11, 10, 10, 8, 8, 7, 5, 4, 4},
		{2, 1, 15, 14, 12, 10, 9, 8, 7, 5, 2},
		{3, 1, 0, 15, 13, 13, 12, 11, 9, 8, 7, 6, 4, 4, 3, 3},
		{3, 2, 1, 15, 13, 12, 12, 10, 9, 7, 7, 4, 3 },
		{3, 2, 1, 15, 14, 12, 12, 11, 11, 8, 8, 7, 6, 5, 4, 3},
		{3, 1, 0, 13, 12, 11, 9, 8, 8, 4, 3 },
		{2, 0, 13, 12, 10, 9, 6, 4, 3 },//18th data.cicle data end
		{2, 0, 15, 13, 12, 12, 11, 9, 8, 7, 5, 3, 4 },
		{3, 2, 0, 15, 13, 12, 12, 11, 10, 8, 7, 6, 4, 4},
		{3, 2, 1, 0, 14, 13, 12, 11, 12, 10, 10, 8, 6, 6, 5, 4, 4, 2},
		{2, 1, 15, 15, 13, 12, 12, 11, 11, 9, 7, 6, 5, 4},
		{3, 2, 0, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 6, 6, 4, 3},
		{2, 1, 0, 14, 13, 12, 12, 11, 11, 10, 8, 7, 6, 7, 5, 4, 4, 3},
		{3, 2, 1, 0, 15, 13, 12, 12, 10, 10, 9, 8, 7, 6, 5, 4, 3, 3},
		{3, 3, 1, 15, 13, 13, 11, 10, 10, 8, 7, 6, 4, 4, 3},
		{4, 4, 3, 1, 1, 0, 15, 14, 13, 12, 12, 11, 11, 8, 8, 7, 7, 6, 4},// 27th data.
		{3, 2, 1, 0, 15, 14, 12, 12, 10, 11, 9, 8, 7, 7, 5, 4, 4, 3, 2 },
		{2, 1, 0, 14, 13, 12, 12, 11, 9, 9, 7, 7, 5, 4, 3, 3},
		{2, 1, 0, 15, 13, 12, 12, 11, 9, 9, 7, 6, 4, 4, 4, 4, 3},
		{3, 2, 1, 1, 15, 14, 13, 12, 10, 11, 9, 7, 7, 6, 4, 4, 4, 3},
		{4, 2, 1, 14, 13, 12, 11, 11, 9, 8, 8, 5, 4, 4, 3},
		{4, 3, 2, 2, 0, 0, 15, 12, 11, 11, 11, 9, 7, 8, 6, 5, 4, 3},
		{3, 3, 2, 0, 0, 14, 13, 12, 11, 10, 8, 7, 7, 5, 4, 4},
		{4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5}, 
		{2, 1, 0, 0, 13, 12, 12, 11, 10, 9, 8, 7, 7, 5, 4, 4, 4},
		{2, 3, 2, 1, 15, 15, 13, 12, 11, 11, 9, 9, 7, 7, 6, 4, 4, 3},
		{3, 2, 2, 1, 15, 15, 14, 12, 12, 12, 11, 11, 11, 9, 8, 6, 6, 5, 4, 3},
		{2, 2, 0, 0, 14, 13, 13, 12, 11, 9, 8, 8, 6, 5, 4, 3},
		{2, 1, 15, 15, 13, 12, 12, 11, 9, 9, 8, 7, 6, 4, 4, 3, 3},	
		{3, 2, 1, 0, 0, 0, 13, 13, 12, 11, 11, 11, 10, 9, 8, 6, 6, 5, 4, 4, 2},
		{2, 2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 9, 7, 7, 5, 4, 4, 3},
		{3, 2, 2, 0, 0, 14, 13, 12, 12, 11, 9, 9, 8, 7, 6, 5, 4},
		{2, 2, 0, 0, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 4},
		{1, 1, 0, 14, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 3, 2},
		{3, 2, 3, 1, 1, 0, 0, 14, 13, 13, 13, 12, 11, 11, 10, 9, 8, 8, 8, 7, 7, 6, 5, 4, 4, 3, 3},
		{1, 2, 1, 0, 14, 13, 13, 12, 11, 11, 8, 9, 8, 7, 6, 5, 4, 4, 4, 4 },
		{2, 2, 1, 0, 0, 15, 13, 12, 11, 11, 11, 8, 8, 8, 8, 7, 5, 4, 3},
		{2, 1, 0, 0, 15, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6, 5, 4, 3, 3 },
		{3, 3, 1, 1, 15, 14, 13, 13, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2},
		{3, 3, 2, 2, 0, 14, 14, 13, 12, 12, 11, 10, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		{3, 3, 1, 0, 0, 14, 13, 12, 12, 10, 9, 8, 8, 6, 5, 5, 4, 3 },
		{2, 2, 1, 0, 15, 14, 12, 12, 12, 10, 8, 8, 8, 6, 6, 4, 4, 4, 3},
		{2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 6, 5, 4, 4, 3 },
		{2, 2, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6, 5, 4, 4, 2},
		{3, 1, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 5, 6, 4, 4, 4},
		{4, 4, 3, 2, 1, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6 },
		{4, 3, 2, 1, 0, 0, 15, 14, 13, 12, 12, 11, 10, 9, 7, 8, 7, 6, 5, 3},
		{2, 3, 2, 1, 1, 0, 0, 14, 14, 13, 12, 12, 12, 11, 11, 10, 9, 9, 7, 8, 7, 7, 6, 5, 4, 3, 3},		//until this works well..
		{4,4,3,2,1,0,15,13,13,12,11,10,10,9,8,7,6,5},
		{2,3,0,15,15,13,12,12,11,10,8,7,7,7,4,4,3},
		{3,2,1,15,13,12,12,11,11,9,6,5,4,4,3},	
		{2,1,0,14,12,11,11,10,10,8,6,5,5,4,3,3,2},	
		{2,1,0,15,13,12,12,12,10,11,9,8,7,5,4,3,3},	 
		{3,2,0,15,13,12,10,9,6,7,4,4},	//65th data.circle data end.
		{2, 1, 0, 14, 13, 12, 11, 10, 9, 7, 7, 4, 4, 3},
		{3, 1, 1, 15, 13, 13, 12, 11, 10, 10, 9, 7, 5, 4, 4, 4, 3, 0},
		{2, 1, 0, 14, 12, 12, 11, 10, 10, 8, 6, 7, 5, 4, 4, 3, 3},
		{3, 2, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 5, 4, 4, 3, 2},
		{3, 2, 1, 15, 14, 12, 11, 10, 10, 7, 7, 5, 4, 3, 4},
		{3, 1, 0, 14, 13, 12, 12, 12, 10, 10, 8, 8, 7, 7, 5, 4, 3, 3, 2},	//71st data  
		{4, 2, 0, 15, 14, 13, 12, 12, 10, 12, 9, 8, 6, 5, 4, 3, 3},
		{2, 2, 0, 0, 14, 13, 12, 11, 11, 10, 9, 7, 6, 5, 4, 3, 2},
		{1, 15, 14, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},
		{2, 1, 1, 15, 13, 12, 11, 10, 11, 8, 8, 8, 5, 4, 4, 2},
		{3, 2, 0, 0, 14, 13, 11, 11, 8, 7, 6, 4, 3, 3},
		{2, 0, 0, 14, 14, 13, 12, 11, 10, 10, 9, 7, 6, 4, 4, 3, 3},
		{3, 2, 1, 0, 14, 13, 11, 10, 10, 10, 8, 7, 6, 4, 4, 3, 3},
		{2, 1, 15, 13, 13, 12, 10, 10, 9, 8, 7, 4, 4, 4, 3 },
		{3, 3, 0, 15, 0, 13, 12, 12, 10, 9, 9, 7, 6, 4, 3},				//80st data  
		{2, 2, 1, 0, 14, 13, 13, 11, 10, 10, 9, 7, 7, 5, 4, 4, 3},
		{2, 1, 0, 0, 14, 12, 13, 11, 11, 10, 10, 9, 8, 8, 6, 6, 5, 5, 4, 4, 4, 3, 2},
		{2, 2, 1, 15, 13, 12, 11, 11, 11, 10, 9, 7, 7, 6, 5, 4, 3, 3 },
		{2, 2, 1, 0, 15, 13, 12, 12, 11, 11, 9, 8, 8, 6, 4, 4, 3, 3 }, 
		{2, 0, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 6, 5, 4, 4, 3},
		{2, 1, 0, 15, 15, 13, 12, 11, 12, 10, 10, 10, 8, 8, 7, 7, 5, 4, 4, 3, 3, 2, 2, 2},
		{1, 1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},					
		{1, 1, 15, 14, 13, 12, 10, 10, 9, 7, 7, 7, 5, 4, 3, 3},
		{4, 3, 0, 3, 15, 13, 13, 12, 11, 9, 7, 7, 5, 4, 4},
		{3, 2, 1, 0, 0, 15, 13, 13, 12, 11, 10, 11, 7, 8, 6, 5, 4, 4, 3},
		{3, 2, 0, 0, 13, 12, 12, 10, 10, 9, 6, 5, 4, 3 },
		{2, 1, 0, 15, 13, 12, 11, 11, 9, 8, 6, 6, 5, 4},
		{2, 0, 15, 14, 12, 12, 10, 9, 9, 7, 6, 5, 5, 4, 3},
		{2, 1, 0, 15, 14, 12, 10, 10, 9, 7, 7, 5, 4, 3},
		{3, 2, 0, 14, 13, 12, 10, 9, 7, 6, 5, 3},
		{3,2,0,14,13,12,10,9,7,6,5,3},
		{3, 2, 2, 0, 15, 14, 12, 12, 11, 9, 8, 8, 6, 5, 4},
		{3, 2, 1, 0, 15, 13, 13, 12, 11, 10, 8, 8, 7, 6, 5, 4, 3},
		{2, 1, 0, 15, 14, 13, 12, 11, 10, 10, 10, 7, 7, 6, 4, 4, 4, 3, 3},
		{4, 4, 2, 2, 1, 15, 13, 13, 12, 10, 10, 10, 8, 7, 5, 4, 3, 3},	
		{3, 4, 3, 2, 2, 1, 0, 15, 15, 15, 13, 12, 12, 12, 11, 10, 8, 9, 9, 8, 8, 6, 4, 4, 4, 3},
		{2, 1, 0, 14, 13, 12, 11, 11, 10, 8, 7, 6, 5, 4, 3},		
		{2, 2, 0, 15, 14, 13, 12, 12, 11, 11, 10, 8, 8, 6, 5, 5, 4, 3},
		{2, 2, 2, 15, 15, 14, 13, 12, 11, 10, 9, 8, 8, 6, 4, 5, 4, 4, 3},
		{2, 1, 0, 14, 15, 13, 12, 11, 11, 9, 8, 8, 6, 5, 4, 4, 3},
		{2, 1, 0, 14, 14, 12, 12, 11, 10, 10, 7, 8, 8, 5, 4, 4, 4 }, 
		{2, 1, 0, 15, 13, 13, 11, 10, 10, 8, 9, 6, 4, 5, 4, 3},
		{3, 2, 0, 2, 15, 15, 13, 13, 12, 11, 10, 8, 8, 7, 7, 4, 6, 5, 4, 3, 3},
		{2, 1, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 7, 4, 7, 4, 5, 4, 2},	
		{2, 2, 15, 14, 14, 12, 11, 10, 9, 8, 6, 6, 5, 4, 3, 2},
		{2, 1, 1, 15, 15, 13, 12, 11, 10, 9, 8, 7, 5, 5, 4, 4, 3 },
		{3, 1, 0, 15, 14, 13, 12, 11, 10, 8, 7, 4, 6, 4, 3},
		{2, 1, 1, 14, 13, 13, 12, 12, 10, 8, 8, 5, 5, 4, 3, 3 }, 
		{2, 0, 1, 14, 14, 14, 13, 12, 11, 10, 10, 10, 7, 6, 6, 5, 4, 4, 3},
		{3, 1, 2, 15, 15, 13, 13, 13, 12, 10, 10, 10, 8, 7, 6, 5, 4, 3, 3, 3},
		{2, 0, 1, 14, 15, 15, 13, 12, 10, 11, 9, 9, 7, 7, 6, 5, 5, 3, 3, 3 },
		{2, 1, 0, 0, 14, 13, 12, 12, 10, 10, 9, 7, 8, 6, 5, 4, 4, 3, 3 },
		{2, 1, 0, 15, 14, 13, 12, 11, 11, 10, 10, 9, 10, 6, 7, 5, 7, 5, 4, 3, 2},
		{2, 2, 0, 15, 14, 13, 12, 11, 11, 10, 8, 8, 6, 5, 4, 3, 3},
		{2, 0, 15, 13, 13, 12, 11, 10, 8, 8, 6, 6, 5, 4, 4, 4, 2},
		{2, 2, 1, 15, 13, 12, 11, 10, 10, 9, 7, 6, 4, 5, 4 },
		{4, 1, 2, 15, 15, 13, 12, 10, 10, 9, 6, 8, 6, 4, 4 },
		{2, 1, 1, 15, 14, 14, 12, 11, 10, 10, 9, 8, 7, 5, 5, 4, 3 },
		{3, 1, 2, 14, 0, 14, 12, 12, 11, 10, 10, 8, 6, 5, 5, 4},
		{2, 2, 15, 15, 13, 13, 12, 10, 10, 9, 7, 8, 6, 4, 4, 4, 4, 3},		
		{2, 2, 1, 0, 15, 15, 14, 13, 13, 11, 11, 10, 10, 9, 9, 6, 8, 6, 5, 5, 4, 4 },
		{2, 2, 1, 0, 14, 14, 12, 13, 11, 11, 10, 9, 9, 7, 6, 5, 4, 4, 4, 3},
		{2, 2, 1, 0, 14, 14, 13, 14, 11, 10, 10, 10, 8, 7, 6, 6, 5, 4, 4, 3},		
		{3,2,1,0,0,0,0,14,15,13,15,12,13,12,12,11,10,11,8,9,8,8,8,7,7,8,6,5,4,4,4,4,3,3},	
		{1, 0, 1, 0, 0, 15, 13, 14, 13, 12, 11, 11, 11, 10, 9, 9, 7, 8, 7, 6, 5, 4, 4, 3, 3 },
		{2, 1, 1, 0, 0, 14, 13, 0, 13, 13, 12, 11, 12, 10, 7, 9, 8, 7, 8, 7, 6, 4, 4, 4, 3, 3 },
		{1, 0, 0, 0, 15, 15, 13, 13, 13, 12, 12, 11, 10, 9, 9, 7, 8, 8, 7, 6, 5, 4, 4, 4, 3},
		{ 2, 1, 1, 0, 15, 13, 0, 13, 12, 12, 11, 12, 10, 8, 10, 8, 7, 8, 7, 6, 4, 4, 4, 4, 3, 3},												
		{3,2,1,0,15,0,0,14,14,12,13,12,12,11,10,10,9,8,8,7,7,6,4,4,4,4,3},
		{1,1,0,0,15,14,14,12,12,11,11,9,10,8,8,7,7,6,5,4,3},		
		{2, 2, 0, 0, 15, 14, 14, 12, 12, 11, 11, 10, 9, 10, 7, 7, 7, 6, 4, 5, 4},
		{2, 2, 0, 0, 15, 14, 14, 13, 12, 11, 10, 10, 7, 8, 8, 8, 5, 5, 4, 4, 4 },
		{2, 2, 1, 0, 14, 13, 12, 12, 11, 11, 9, 8, 8, 7, 6, 4, 4, 4},
		{3, 2, 2, 1, 0, 15, 13, 13, 12, 11, 11, 11, 8, 8, 8, 7, 5, 5, 4, 4},
		{2, 2, 1, 0, 15, 13, 13, 11, 10, 9, 8, 8, 7, 5, 4, 4 },
		{2, 3, 1, 1, 15, 15, 13, 12, 11, 12, 9, 9, 8, 6, 6, 4, 4 },
		{3, 2, 1, 0, 15, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 6, 5, 4, 4},
		{3, 3, 1, 2, 0, 15, 14, 15, 13, 13, 11, 12, 10, 9, 8, 8, 7, 6, 5, 4, 4},
		{2, 2, 1, 0, 14, 13, 13, 11, 11, 10, 9, 7, 7, 5, 4, 4},
		{2, 2, 2, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3 },
		{1, 1, 0, 0, 13, 12, 11, 10, 9, 7, 7, 6, 4, 3 },
		{2, 3, 1, 0, 0, 15, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4},
		{2, 1, 0, 0, 13, 13, 12, 10, 10, 10, 8, 6, 6, 5, 4, 3},
		{2, 1, 1, 0, 15, 14, 13, 12, 10, 9, 10, 7, 7, 6, 4, 4, 3, 4},
		{2, 3, 1, 0, 0, 15, 14, 12, 11, 10, 10, 8, 7, 7, 6, 4, 4, 3},
		{2, 2, 2, 14, 15, 14, 14, 13, 11, 11, 10, 10, 8, 8, 7, 6, 4, 5, 3, 3},
		{1, 1, 15, 14, 13, 12, 11, 9, 9, 5, 5, 3 },
		{3, 2, 2, 0, 15, 15, 13, 13, 12, 12, 10, 12, 10, 10, 8, 6, 8, 5, 5},
		{2, 2, 0, 14, 15, 15, 13, 13, 11, 11, 11, 10, 9, 9, 6, 6, 5, 5, 4, 4 },//154th data circledata end.
										 
 
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
										
										
		//{4, 4, 4, 4, 3, 0, 0, 15, 0, 15, 13, 12, 12, 12, 12, 12, 11, 12, 8, 8, 8, 8, 7, 8, 8},	//-ve Square data..
		//{4, 3, 3, 4, 3, 1, 0, 0, 0, 0, 15, 11, 12, 12, 12, 11, 12, 8, 7, 8, 8, 8},	//Square data..
		//{4, 3, 4, 4, 3, 3, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 15, 13, 12, 11, 12, 12, 12, 12, 11, 11, 8, 8, 7, 8, 8, 8, 8, 8, 7, 7, 7, 8, 7, 7},	//square data.
		//{4, 3, 3, 4, 3, 3, 3, 0, 1, 0, 0, 0, 15, 0, 0, 0, 0, 0, 13, 12, 11, 12, 12, 12, 11, 11, 12, 11, 8, 9, 8, 8, 8, 8, 7, 8, 7, 8, 7},	//square data.
		//{4, 4, 3, 0, 0, 15, 0, 15, 12, 12, 12, 11, 12, 8, 7, 8, 7, 8, 8, 7},	//square data.
		//{4, 4, 3, 4, 3, 0, 0, 0, 0, 0, 0, 15, 15, 0, 13, 12, 12, 12, 12, 12, 12, 8, 8, 8, 7, 8, 8, 7, 8, 7, 8},	//square data.
		//{4, 4, 4, 3, 0, 0, 0, 0, 0, 14, 0, 12, 12, 12, 11, 11, 7, 8, 9, 7, 8, 7, 8},	//square data.
		//{4, 4, 4, 3, 0, 0, 0, 15, 12, 12, 12, 11, 8, 8, 8, 7},	//square data. 
		//{4, 5, 4, 3, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 
		//{4, 5, 4, 5, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 
		//{4, 4, 4, 3, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 										
		//{4,4,3,0,15,0,12,11,12,12,7,8,7,8},	//square data		 
		//{ 4, 3, 4, 3, 3, 4, 0, 15, 0, 0, 15, 0, 0, 0, 13, 12, 12, 12, 11, 12, 12, 11, 7, 8, 8, 8, 8, 7, 7, 7},
		//{4, 4, 3, 3, 0, 15, 0, 0, 0, 0, 15, 0, 13, 12, 12, 12, 11, 12, 9, 7, 8, 8, 7, 7, 8, 7},
		//{4, 4, 3, 3, 3, 15, 0, 0, 0, 0, 15, 13, 12, 11, 12, 11, 12, 8, 8, 8, 8, 7, 7, 7, 8},		
		//{ 4, 3, 3, 3, 1, 15, 0, 15, 15, 0, 13, 12, 12, 12, 12, 12, 7, 7, 9, 8, 8, 8, 8, 7 },	//square data  
		//{4, 3, 4, 3, 0, 15, 0, 15, 15, 0, 11, 12, 12, 12, 11, 7, 8, 7, 8, 7, 7},
		//{4, 4, 3, 4, 0, 15, 0, 0, 0, 15, 0, 13, 12, 12, 11, 12, 11, 8, 7, 8, 8, 7, 8, 7 },
		//{4, 3, 3, 3, 4, 0, 0, 15, 0, 15, 0, 15, 12, 12, 12, 12, 11, 11, 7, 9, 8, 8, 7, 8},
		//{4, 3, 3, 4, 4, 1, 0, 0, 0, 15, 15, 13, 12, 12, 11, 11, 8, 8, 8, 8, 7, 7},
		//{4, 3, 3, 15, 15, 15, 15, 15, 15, 12, 12, 12, 11, 8, 8, 7, 7, 7},
		//{4, 4, 3, 3, 3, 15, 15, 0, 15, 15, 0, 13, 11, 12, 12, 12, 8, 8, 8, 8, 8, 7, 8, 7, 7},
		//{4, 3, 4, 3, 3, 0, 15, 0, 15, 0, 15, 0, 0, 12, 11, 12, 12, 12, 12, 7, 7, 9, 8, 7, 8, 7, 7},
		//{4, 4, 3, 3, 15, 0, 0, 0, 0, 15, 15, 12, 12, 12, 12, 8, 9, 8, 8, 7, 7, 7},
		//{4, 4, 3, 4, 0, 15, 0, 0, 15, 15, 15, 12, 11, 12, 11, 11, 7, 8, 8, 7, 7, 7 },
		//{4,4,0,0,12,12,8,8},	
		//{4, 4, 4, 3, 3, 0, 15, 0, 15, 15, 0, 0, 12, 11, 12, 12, 11, 7, 9, 8, 8, 7, 7, 7 },
		//{4, 4, 4, 4, 3, 1, 0, 15, 15, 15, 0, 0, 15, 12, 12, 12, 12, 11, 8, 7, 8, 9, 8, 8, 7, 7},	//square data  
		//{4, 3, 4, 4, 3, 0, 0, 0, 0, 15, 0, 15, 0, 12, 12, 12, 11, 12, 12, 12, 9, 8, 7, 9, 9, 7, 7, 8},
		//{4, 4, 4, 4, 4, 3, 15, 0, 0, 0, 0, 0, 0, 13, 12, 12, 12, 12, 12, 11, 6, 8, 8, 9, 7, 9, 7, 7, 7},
		//{4, 4, 4, 4, 4, 1, 0, 0, 0, 15, 15, 0, 12, 12, 12, 12, 12, 12, 12, 7, 7, 9, 7, 9, 8, 7, 8, 7},
		//{4, 4, 4, 4, 3, 15, 0, 0, 0, 0, 0, 0, 15, 12, 12, 12, 11, 12, 12, 12, 7, 7, 8, 8, 8, 8, 8, 8},
		//{4, 4, 3, 3, 0, 0, 15, 15, 15, 0, 12, 12, 12, 12, 12, 8, 8, 8, 8, 7, 7, 7},
		//{4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 15, 12, 12, 12, 11, 12, 12, 12, 12, 7, 7, 8, 8, 7, 8, 7, 7},
		//{4, 4, 4, 3, 4, 4, 1, 0, 0, 15, 0, 0, 15, 0, 0, 15, 12, 11, 12, 12, 11, 12, 12, 12, 12, 8, 7, 8, 8, 8, 8, 7, 8, 7, 7},
		//{ 4, 4, 3, 4, 3, 1, 15, 0, 0, 15, 0, 0, 0, 15, 12, 12, 12, 12, 12, 12, 11, 8, 7, 8, 8, 8, 8, 8, 7, 7, 7, 7 },
		//{4, 4, 4, 3, 3, 15, 0, 0, 0, 15, 0, 0, 13, 12, 12, 12, 12, 12, 7, 8, 8, 8, 7, 7, 7},
		//{4, 4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 15, 0, 0, 0, 13, 11, 12, 11, 12, 12, 12, 12, 9, 7, 7, 8, 8, 8, 8, 7, 7, 7},
		//{4, 4, 4, 15, 0, 15, 15, 15, 13, 12, 12, 11, 8, 8, 8, 7, 7, 7, 8},
		//{4, 3, 4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 15, 15, 12, 12, 12, 12, 12, 12, 12, 12, 8, 7, 8, 8, 8, 7, 8, 8, 8},
		//{4, 3, 4, 4, 3, 4, 1, 0, 0, 0, 0, 15, 0, 0, 13, 12, 11, 11, 12, 12, 11, 8, 7, 8, 8, 8, 7, 7, 7, 8},
		//{4, 3, 3, 3, 3, 4, 15, 15, 0, 0, 15, 0, 15, 11, 12, 12, 12, 12, 12, 8, 7, 8, 8, 8, 8, 7, 7, 7},
		//{4, 3, 4, 0, 15, 0, 0, 0, 15, 13, 12, 12, 12, 8, 8, 8, 7, 7, 7, 8},
		//{4, 3, 4, 3, 3, 15, 15, 0, 0, 15, 15, 13, 12, 11, 12, 12, 8, 8, 8, 8, 7, 7},
		//{4, 4, 3, 4, 3, 3, 3, 1, 15, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11, 8, 8, 8, 8, 7, 8, 8, 8, 7, 7, 7},
		//{ 4, 4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 12, 12, 11, 7, 8, 8, 7, 8, 7, 7, 7},
		//{4, 4, 3, 4, 15, 15, 0, 15, 0, 0, 15, 0, 15, 12, 11, 12, 12, 12, 12, 8, 8, 8, 8, 8, 7, 8, 9},
		//{4, 4, 4, 4, 0, 0, 0, 0, 0, 15, 12, 12, 12, 12, 11, 8, 7, 8, 8, 7, 7 },		//square data 48th data end.
										
		//{4, 4, 4, 4, 4, 4, 13, 13, 13, 13, 8, 7, 8, 7, 8},	//1st triangle data		-ve
		//	{5, 4, 5, 5, 4, 4, 14, 13, 13, 13, 8, 7, 8, 7, 8},	//triangle data
		//	{3, 4, 3, 3, 4, 4, 14, 13, 13, 13, 8, 7, 8, 7, 8},	//triangle data
		//	{4,4,4,3,4,3,0,15,13,13,14,13,13,13,12,8,8,8,8,7,7,8},	//triangle data
		//	{4,4,4,4,3,4,0,13,14,13,14,13,13,13,9,8,8,8,8,8},	//triangle data
		//	{3, 4, 3, 3, 3, 3, 2, 2, 0, 13, 13, 13, 12, 13, 12, 12, 12, 4, 8, 8, 8, 8, 8, 8},	//triangle data
		//	{3, 3, 3, 3, 3, 1, 2, 14, 13, 13, 12, 12, 12, 12, 11, 7, 8, 8, 8, 6, 8},	//triangle data
		//	{4, 4, 4, 4, 3, 2, 0, 14, 13, 12, 13, 13, 12, 13, 12, 8, 8, 8, 7, 8, 6, 8},	//triangle data
		//	{4, 4, 3, 2, 2, 0, 13, 14, 12, 13, 13, 13, 12, 13, 8, 8, 8, 7, 7},	//triangle data
		//	{3, 3, 2, 2, 2, 3, 2, 0, 12, 12, 12, 13, 12, 13, 13, 12, 8, 8, 8, 7, 9, 7, 8 },	//triangle data
		//	{ 3, 3, 3, 2, 2, 2, 2, 1, 13, 12, 13, 13, 13, 12, 12, 8, 7, 8, 8, 8, 7, 8, 8},	//triangle data
		//	{ 3, 3, 4, 4, 3, 1, 13, 13, 13, 13, 13, 13, 12, 9, 8, 8, 7, 7 },	//triangle data
		//	{4, 4, 4, 3, 4, 3, 0, 13, 13, 13, 13, 13, 13, 12, 8, 8, 8, 8, 8, 7, 7},	//triangle data
		//	{4, 4, 4, 4, 4, 3, 0, 14, 13, 13, 13, 13, 13, 12, 9, 8, 8, 8, 7, 7},	//triangle data
		//	{ 3, 3, 3, 2, 3, 3, 2, 3, 14, 13, 12, 12, 13, 12, 12, 12, 12, 10, 9, 9, 8, 8, 4, 8},	//triangle data
		//	{3, 3, 3, 3, 3, 3, 15, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 7},
		//	{3, 3, 3, 3, 2, 3, 13, 13, 12, 12, 12, 12, 9, 8, 8, 7, 8},
		//	{3, 3, 3, 3, 2, 13, 13, 13, 12, 13, 12, 8, 8, 6, 8, 8, 7},
		//	{3, 3, 4, 4, 3, 1, 15, 15, 13, 13, 13, 13, 12, 14, 12, 9, 8, 8, 7, 7, 8, 7 },
		//	{3, 4, 4, 4, 4, 3, 14, 14, 13, 13, 13, 13, 13, 13, 8, 8, 8, 8, 8 },
		//	{3, 3, 3, 3, 3, 1, 14, 13, 13, 12, 12, 13, 11, 8, 8, 8, 7, 8, 8},
		//	{4, 4, 3, 4, 4, 1, 15, 14, 13, 13, 13, 12, 12, 12, 8, 8, 7, 8, 7, 7, 7 },
		//	{3, 3, 2, 3, 2, 0, 14, 13, 13, 12, 12, 12, 12, 8, 9, 8, 8, 8, 7, 8, 7},
		//	{3, 3, 2, 3, 2, 0, 14, 13, 13, 12, 12, 12, 12, 8, 9, 8, 8, 8, 7, 8, 7},
		//	{3, 3, 3, 3, 3, 3, 1, 0, 14, 13, 12, 13, 13, 12, 13, 8, 8, 8, 8, 7, 7, 8, 7, 7 },
		//	//{2, 2, 2, 0, 13, 13, 12, 12, 8, 7, 8, 8},
		//	{3, 3, 3, 3, 2, 2, 2, 15, 13, 12, 13, 13, 13, 12, 10, 12, 12, 6, 8, 7, 15, 8, 7, 8, 7},
		//	{3, 3, 3, 2, 3, 3, 2, 1, 13, 13, 13, 12, 13, 12, 12, 13, 12, 8, 8, 8, 8, 7, 8, 7, 8},
		//	//{2, 2, 2, 2, 13, 13, 12, 13, 12, 12, 8, 8, 7, 7},
		//	{3, 2, 3, 2, 2, 2, 13, 12, 13, 12, 12, 13, 12, 9, 8, 8, 7, 7},
		//	{4, 4, 4, 4, 4, 4, 4, 3, 0, 15, 14, 13, 13, 13, 13, 12, 12, 13, 8, 8, 8, 15, 7, 9, 7, 7},	//triangle data
		//	{3,3,3,3,2,2,13,12,12,12,13,12,12,12,11,9,7,8,7,7},	
		//	//{2, 3, 3, 3, 3, 2, 15, 13, 13, 13, 13, 11, 12, 12, 8, 8, 7, 7, 7, 8},
		//	{3, 3, 3, 3, 3, 2, 3, 2, 13, 13, 13, 12, 13, 12, 12, 11, 8, 8, 7, 7, 7, 8},
		//	{ 3, 3, 3, 3, 3, 2, 2, 14, 15, 13, 13, 12, 13, 12, 12, 12, 7, 8, 8, 8, 7, 8},
		//	{4, 4, 4, 4, 4, 4, 0, 0, 15, 14, 14, 13, 13, 13, 12, 13, 13, 6, 9, 8, 9, 8, 6, 7},
		//	{4, 3, 4, 4, 4, 0, 14, 13, 13, 13, 13, 13, 12, 13, 11, 7, 8, 8, 7, 7, 7 },
		//	{4, 4, 4, 4, 4, 4, 4, 0, 0, 13, 13, 12, 13, 14, 12, 15, 10, 7, 8, 7, 8, 6, 8 },
		//	{3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 14, 12, 13, 12, 12, 8, 8, 8, 7, 7, 7 },
		//	{3, 3, 4, 3, 3, 3, 3, 1, 13, 13, 13, 13, 13, 12, 13, 12, 12, 7, 8, 8, 8, 7, 8},
		//	//{2, 3, 3, 3, 3, 3, 3, 14, 14, 13, 12, 13, 12, 12, 12, 9, 8, 8, 8, 8, 7, 8},
		//	{3, 2, 2, 2, 2, 2, 2, 2, 2, 0, 13, 13, 13, 13, 13, 13, 12, 13, 12, 9, 7, 7, 8, 8, 8, 8, 7, 8 },
		//	{3, 3, 3, 2, 3, 2, 2, 2, 13, 13, 13, 14, 13, 12, 12, 12, 12, 9, 8, 8, 7, 8, 8, 8},
		//	{4, 4, 4, 4, 4, 4, 15, 14, 13, 13, 13, 13, 11, 8, 9, 8, 7, 6 },
		//	{4, 3, 3, 4, 3, 15, 14, 13, 13, 13, 15, 13, 12, 12, 13, 12, 7, 7, 8, 8, 8, 7, 7},
		//	{ 4, 3, 3, 4, 4, 15, 14, 13, 13, 14, 13, 13, 12, 8, 8, 7, 8, 7, 7, 6},		 
		//	{4, 3, 3, 3, 4, 2, 15, 13, 13, 13, 13, 14, 13, 13, 12, 8, 8, 8, 8, 7, 7, 8},
		//	{4, 4, 4, 4, 3, 3, 1, 13, 14, 13, 13, 13, 14, 13, 12, 10, 6, 9, 7, 8, 7, 8, 7},
		//	{3, 4, 4, 3, 3, 3, 3, 14, 13, 13, 13, 13, 12, 14, 13, 12, 12, 7, 8, 8, 8, 6, 7, 8 },
		//	{4, 4, 4, 4, 3, 3, 3, 3, 0, 13, 13, 12, 14, 13, 13, 14, 13, 12, 10, 7, 8, 7, 8, 7, 9},
		//	{4, 4, 3, 3, 4, 4, 2, 2, 14, 13, 13, 13, 13, 13, 13, 12, 12, 8, 9, 7, 7, 8, 7, 8},
		//	{3, 3, 4, 3, 4, 3, 3, 15, 13, 13, 13, 14, 14, 13, 12, 12, 13, 8, 8, 7, 8, 8, 8, 8, 7},
		//	{4, 3, 4, 4, 3, 3, 15, 14, 14, 13, 13, 14, 13, 13, 13, 12, 8, 8, 9, 8, 7, 7, 6, 7, 8 },
		//	{4, 3, 3, 3, 3, 3, 3, 3, 0, 14, 13, 13, 12, 13, 13, 14, 12, 12, 6, 8, 8, 7, 8, 8, 8, 7, 7},
		//	{4, 3, 3, 4, 3, 4, 3, 15, 13, 13, 13, 13, 14, 13, 13, 13, 6, 9, 8, 8, 7, 8, 8, 6},//52nd triangle data -ve end
	};	//44th random -ve data

											

	Mat trainingDataMatTemp(noOfTrainingSamplesCircle, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatCircle);
	 
}

 
void GestureRecognizer::copyArrayDataToVec(){
	
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

void GestureRecognizer::squareTrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesSquare][50] = {	
		{4, 4, 4, 4, 3, 0, 0, 15, 0, 15, 13, 12, 12, 12, 12, 12, 11, 12, 8, 8, 8, 8, 7, 8, 8},	//+ve Square data..
		{4, 3, 3, 4, 3, 1, 0, 0, 0, 0, 15, 11, 12, 12, 12, 11, 12, 8, 7, 8, 8, 8},	//Square data..
		{4, 3, 4, 4, 3, 3, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 15, 13, 12, 11, 12, 12, 12, 12, 11, 11, 8, 8, 7, 8, 8, 8, 8, 8, 7, 7, 7, 8, 7, 7},	//square data.
		{4, 3, 3, 4, 3, 3, 3, 0, 1, 0, 0, 0, 15, 0, 0, 0, 0, 0, 13, 12, 11, 12, 12, 12, 11, 11, 12, 11, 8, 9, 8, 8, 8, 8, 7, 8, 7, 8, 7},	//square data.
		{4, 4, 3, 0, 0, 15, 0, 15, 12, 12, 12, 11, 12, 8, 7, 8, 7, 8, 8, 7},	//square data.
		{4, 4, 3, 4, 3, 0, 0, 0, 0, 0, 0, 15, 15, 0, 13, 12, 12, 12, 12, 12, 12, 8, 8, 8, 7, 8, 8, 7, 8, 7, 8},	//square data.
		{4, 4, 4, 3, 0, 0, 0, 0, 0, 14, 0, 12, 12, 12, 11, 11, 7, 8, 9, 7, 8, 7, 8},	//square data.
		{4, 4, 4, 3, 0, 0, 0, 15, 12, 12, 12, 11, 8, 8, 8, 7},	//square data. 
		{4, 5, 4, 3, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 
		{4, 5, 4, 5, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 
		{4, 4, 4, 3, 1, 0, 0, 15, 13, 12, 12, 11, 9, 8, 8, 7},	//square data. 										
		{4,4,3,0,15,0,12,11,12,12,7,8,7,8},	//square data		 
		{ 4, 3, 4, 3, 3, 4, 0, 15, 0, 0, 15, 0, 0, 0, 13, 12, 12, 12, 11, 12, 12, 11, 7, 8, 8, 8, 8, 7, 7, 7},
		{4, 4, 3, 3, 0, 15, 0, 0, 0, 0, 15, 0, 13, 12, 12, 12, 11, 12, 9, 7, 8, 8, 7, 7, 8, 7},
		{4, 4, 3, 3, 3, 15, 0, 0, 0, 0, 15, 13, 12, 11, 12, 11, 12, 8, 8, 8, 8, 7, 7, 7, 8},		
		{ 4, 3, 3, 3, 1, 15, 0, 15, 15, 0, 13, 12, 12, 12, 12, 12, 7, 7, 9, 8, 8, 8, 8, 7 },	//square data  
		{4, 3, 4, 3, 0, 15, 0, 15, 15, 0, 11, 12, 12, 12, 11, 7, 8, 7, 8, 7, 7},
		{4, 4, 3, 4, 0, 15, 0, 0, 0, 15, 0, 13, 12, 12, 11, 12, 11, 8, 7, 8, 8, 7, 8, 7 },
		{4, 3, 3, 3, 4, 0, 0, 15, 0, 15, 0, 15, 12, 12, 12, 12, 11, 11, 7, 9, 8, 8, 7, 8},
		{4, 3, 3, 4, 4, 1, 0, 0, 0, 15, 15, 13, 12, 12, 11, 11, 8, 8, 8, 8, 7, 7},
		{4, 3, 3, 15, 15, 15, 15, 15, 15, 12, 12, 12, 11, 8, 8, 7, 7, 7},
		{4, 4, 3, 3, 3, 15, 15, 0, 15, 15, 0, 13, 11, 12, 12, 12, 8, 8, 8, 8, 8, 7, 8, 7, 7},
		{4, 3, 4, 3, 3, 0, 15, 0, 15, 0, 15, 0, 0, 12, 11, 12, 12, 12, 12, 7, 7, 9, 8, 7, 8, 7, 7},
		{4, 4, 3, 3, 15, 0, 0, 0, 0, 15, 15, 12, 12, 12, 12, 8, 9, 8, 8, 7, 7, 7},
		{4, 4, 3, 4, 0, 15, 0, 0, 15, 15, 15, 12, 11, 12, 11, 11, 7, 8, 8, 7, 7, 7 },
		{4,4,0,0,12,12,8,8},	
		{4, 4, 4, 3, 3, 0, 15, 0, 15, 15, 0, 0, 12, 11, 12, 12, 11, 7, 9, 8, 8, 7, 7, 7 },
		{4, 4, 4, 4, 3, 1, 0, 15, 15, 15, 0, 0, 15, 12, 12, 12, 12, 11, 8, 7, 8, 9, 8, 8, 7, 7},	//square data  
		{4, 3, 4, 4, 3, 0, 0, 0, 0, 15, 0, 15, 0, 12, 12, 12, 11, 12, 12, 12, 9, 8, 7, 9, 9, 7, 7, 8},
		{4, 4, 4, 4, 4, 3, 15, 0, 0, 0, 0, 0, 0, 13, 12, 12, 12, 12, 12, 11, 6, 8, 8, 9, 7, 9, 7, 7, 7},
		{4, 4, 4, 4, 4, 1, 0, 0, 0, 15, 15, 0, 12, 12, 12, 12, 12, 12, 12, 7, 7, 9, 7, 9, 8, 7, 8, 7},
		{4, 4, 4, 4, 3, 15, 0, 0, 0, 0, 0, 0, 15, 12, 12, 12, 11, 12, 12, 12, 7, 7, 8, 8, 8, 8, 8, 8},
		{4, 4, 3, 3, 0, 0, 15, 15, 15, 0, 12, 12, 12, 12, 12, 8, 8, 8, 8, 7, 7, 7},
		{4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 15, 12, 12, 12, 11, 12, 12, 12, 12, 7, 7, 8, 8, 7, 8, 7, 7},
		{4, 4, 4, 3, 4, 4, 1, 0, 0, 15, 0, 0, 15, 0, 0, 15, 12, 11, 12, 12, 11, 12, 12, 12, 12, 8, 7, 8, 8, 8, 8, 7, 8, 7, 7},
		{ 4, 4, 3, 4, 3, 1, 15, 0, 0, 15, 0, 0, 0, 15, 12, 12, 12, 12, 12, 12, 11, 8, 7, 8, 8, 8, 8, 8, 7, 7, 7, 7 },
		{4, 4, 4, 3, 3, 15, 0, 0, 0, 15, 0, 0, 13, 12, 12, 12, 12, 12, 7, 8, 8, 8, 7, 7, 7},
		{4, 4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 15, 0, 0, 0, 13, 11, 12, 11, 12, 12, 12, 12, 9, 7, 7, 8, 8, 8, 8, 7, 7, 7},
		{4, 4, 4, 15, 0, 15, 15, 15, 13, 12, 12, 11, 8, 8, 8, 7, 7, 7, 8},
		{4, 3, 4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 15, 15, 12, 12, 12, 12, 12, 12, 12, 12, 8, 7, 8, 8, 8, 7, 8, 8, 8},
		{4, 3, 4, 4, 3, 4, 1, 0, 0, 0, 0, 15, 0, 0, 13, 12, 11, 11, 12, 12, 11, 8, 7, 8, 8, 8, 7, 7, 7, 8},
		{4, 3, 3, 3, 3, 4, 15, 15, 0, 0, 15, 0, 15, 11, 12, 12, 12, 12, 12, 8, 7, 8, 8, 8, 8, 7, 7, 7},
		{4, 3, 4, 0, 15, 0, 0, 0, 15, 13, 12, 12, 12, 8, 8, 8, 7, 7, 7, 8},
		{4, 3, 4, 3, 3, 15, 15, 0, 0, 15, 15, 13, 12, 11, 12, 12, 8, 8, 8, 8, 7, 7},
		{4, 4, 3, 4, 3, 3, 3, 1, 15, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11, 8, 8, 8, 8, 7, 8, 8, 8, 7, 7, 7},
		{ 4, 4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 12, 12, 11, 7, 8, 8, 7, 8, 7, 7, 7},
		{4, 4, 3, 4, 15, 15, 0, 15, 0, 0, 15, 0, 15, 12, 11, 12, 12, 12, 12, 8, 8, 8, 8, 8, 7, 8, 9},
		{4, 4, 4, 4, 0, 0, 0, 0, 0, 15, 12, 12, 12, 12, 11, 8, 7, 8, 8, 7, 7 },		//square data 48th data end.



		{0, 15, 0, 14, 0, 3, 4, 6, 8, 9, 12, 15, 0, 14, 15, 0, 10, 9, 8, 7},		//random -ve samples 0th
		{3, 13, 12, 13, 12, 2, 3, 2, 3, 2, 13, 12},		 
		{3, 12, 11, 12, 11, 5, 4, 3, 0, 13, 12},
		{8, 4, 13, 3, 1, 0, 1, 7, 8, 7, 9, 4, 2, 1, 0, 15, 8},  
		{3, 4, 3, 13, 12, 13, 12, 4, 3, 4, 3, 2, 3, 2, 0, 13, 12, 8, 4 },
		{2, 1, 0, 1, 2, 12, 11, 13, 12, 8, 6, 8, 7, 8 },
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
		{4,12,11},					//45th -ve random data..

		{2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 9, 9, 8, 8, 7, 8, 4, 4, 4, 3, 2},
		{2, 2, 2, 1, 0, 15, 13, 13, 12, 12, 11, 11, 10, 9, 9, 7, 8, 6, 7, 5, 4, 4, 4, 3, 3, 3},
		{3, 3, 2, 1, 1, 0, 15, 14, 13, 12, 12, 11, 12, 10, 8, 11, 8, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3},
		{2, 1, 0, 13, 12, 11, 10, 9, 8, 6, 4, 3, 3 },
		{1, 0, 0, 13, 12, 12, 12, 11, 12, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		{3, 3, 1, 0, 0, 14, 13, 13, 12, 12, 11, 11, 8, 9, 7, 6, 6, 4, 4, 4, 3},
		{3, 2, 1, 1, 0, 0, 15, 15, 14, 13, 12, 13, 12, 11, 11, 9, 10, 9, 9, 8, 7, 8, 8, 7, 6, 4, 4, 4, 4, 4},
		{2, 1, 2, 1, 0, 0, 0, 15, 14, 14, 12, 13, 12, 12, 11, 11, 11, 10, 9, 9, 8, 7, 8, 8, 7, 6, 7, 4, 4, 4, 4, 3, 4, 3, 3},
		{2, 2, 1, 0, 0, 15, 14, 14, 12, 12, 12, 11, 11, 10, 10, 10, 9, 6, 8, 7, 7, 6, 5, 4, 4, 4, 3, 3, 3 }, 					 
		{4, 2, 3, 3, 2, 0, 0, 14, 13, 13, 12, 12, 12, 12, 11, 10, 9, 8, 8, 8, 7, 6, 5, 4}, 
		{2, 3, 2, 3, 0, 14, 13, 13, 12, 11, 10, 10, 8, 8, 7, 5, 4, 4},
		{2, 1, 15, 14, 12, 10, 9, 8, 7, 5, 2},
		{3, 1, 0, 15, 13, 13, 12, 11, 9, 8, 7, 6, 4, 4, 3, 3},
		{3, 2, 1, 15, 13, 12, 12, 10, 9, 7, 7, 4, 3 },
		{3, 2, 1, 15, 14, 12, 12, 11, 11, 8, 8, 7, 6, 5, 4, 3},
		{3, 1, 0, 13, 12, 11, 9, 8, 8, 4, 3 },
		{2, 0, 13, 12, 10, 9, 6, 4, 3 },//18th data.cicle data end
		{2, 0, 15, 13, 12, 12, 11, 9, 8, 7, 5, 3, 4 },
		{3, 2, 0, 15, 13, 12, 12, 11, 10, 8, 7, 6, 4, 4},
		{3, 2, 1, 0, 14, 13, 12, 11, 12, 10, 10, 8, 6, 6, 5, 4, 4, 2},
		{2, 1, 15, 15, 13, 12, 12, 11, 11, 9, 7, 6, 5, 4},
		{3, 2, 0, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 6, 6, 4, 3},
		{2, 1, 0, 14, 13, 12, 12, 11, 11, 10, 8, 7, 6, 7, 5, 4, 4, 3},
		{3, 2, 1, 0, 15, 13, 12, 12, 10, 10, 9, 8, 7, 6, 5, 4, 3, 3},
		{3, 3, 1, 15, 13, 13, 11, 10, 10, 8, 7, 6, 4, 4, 3},
		{4, 4, 3, 1, 1, 0, 15, 14, 13, 12, 12, 11, 11, 8, 8, 7, 7, 6, 4},// 27th data.
		{3, 2, 1, 0, 15, 14, 12, 12, 10, 11, 9, 8, 7, 7, 5, 4, 4, 3, 2 },
		{2, 1, 0, 14, 13, 12, 12, 11, 9, 9, 7, 7, 5, 4, 3, 3},
		{2, 1, 0, 15, 13, 12, 12, 11, 9, 9, 7, 6, 4, 4, 4, 4, 3},
		{3, 2, 1, 1, 15, 14, 13, 12, 10, 11, 9, 7, 7, 6, 4, 4, 4, 3},
		{4, 2, 1, 14, 13, 12, 11, 11, 9, 8, 8, 5, 4, 4, 3},
		{4, 3, 2, 2, 0, 0, 15, 12, 11, 11, 11, 9, 7, 8, 6, 5, 4, 3},
		{3, 3, 2, 0, 0, 14, 13, 12, 11, 10, 8, 7, 7, 5, 4, 4},
		{4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5}, 
		{2, 1, 0, 0, 13, 12, 12, 11, 10, 9, 8, 7, 7, 5, 4, 4, 4},
		{2, 3, 2, 1, 15, 15, 13, 12, 11, 11, 9, 9, 7, 7, 6, 4, 4, 3},
		{3, 2, 2, 1, 15, 15, 14, 12, 12, 12, 11, 11, 11, 9, 8, 6, 6, 5, 4, 3},
		{2, 2, 0, 0, 14, 13, 13, 12, 11, 9, 8, 8, 6, 5, 4, 3},
		{2, 1, 15, 15, 13, 12, 12, 11, 9, 9, 8, 7, 6, 4, 4, 3, 3},	
		{3, 2, 1, 0, 0, 0, 13, 13, 12, 11, 11, 11, 10, 9, 8, 6, 6, 5, 4, 4, 2},
		{2, 2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 9, 7, 7, 5, 4, 4, 3},
		{3, 2, 2, 0, 0, 14, 13, 12, 12, 11, 9, 9, 8, 7, 6, 5, 4},
		{2, 2, 0, 0, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 4},
		{1, 1, 0, 14, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 3, 2},
		{3, 2, 3, 1, 1, 0, 0, 14, 13, 13, 13, 12, 11, 11, 10, 9, 8, 8, 8, 7, 7, 6, 5, 4, 4, 3, 3},
		{1, 2, 1, 0, 14, 13, 13, 12, 11, 11, 8, 9, 8, 7, 6, 5, 4, 4, 4, 4 },
		{2, 2, 1, 0, 0, 15, 13, 12, 11, 11, 11, 8, 8, 8, 8, 7, 5, 4, 3},
		{2, 1, 0, 0, 15, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6, 5, 4, 3, 3 },
		{3, 3, 1, 1, 15, 14, 13, 13, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2},
		{3, 3, 2, 2, 0, 14, 14, 13, 12, 12, 11, 10, 8, 8, 8, 6, 6, 4, 4, 3, 3},
		{3, 3, 1, 0, 0, 14, 13, 12, 12, 10, 9, 8, 8, 6, 5, 5, 4, 3 },
		{2, 2, 1, 0, 15, 14, 12, 12, 12, 10, 8, 8, 8, 6, 6, 4, 4, 4, 3},
		{2, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 6, 5, 4, 4, 3 },
		{2, 2, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6, 5, 4, 4, 2},
		{3, 1, 1, 0, 15, 13, 13, 12, 12, 10, 9, 8, 8, 5, 6, 4, 4, 4},
		{4, 4, 3, 2, 1, 0, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6 },
		{4, 3, 2, 1, 0, 0, 15, 14, 13, 12, 12, 11, 10, 9, 7, 8, 7, 6, 5, 3},
		{2, 3, 2, 1, 1, 0, 0, 14, 14, 13, 12, 12, 12, 11, 11, 10, 9, 9, 7, 8, 7, 7, 6, 5, 4, 3, 3},		//until this works well..
		{4,4,3,2,1,0,15,13,13,12,11,10,10,9,8,7,6,5},
		{2,3,0,15,15,13,12,12,11,10,8,7,7,7,4,4,3},
		{3,2,1,15,13,12,12,11,11,9,6,5,4,4,3},	
		{2,1,0,14,12,11,11,10,10,8,6,5,5,4,3,3,2},	
		{2,1,0,15,13,12,12,12,10,11,9,8,7,5,4,3,3},	 
		{3,2,0,15,13,12,10,9,6,7,4,4},	//65th data.circle data end.
		{2, 1, 0, 14, 13, 12, 11, 10, 9, 7, 7, 4, 4, 3},
		{3, 1, 1, 15, 13, 13, 12, 11, 10, 10, 9, 7, 5, 4, 4, 4, 3, 0},
		{2, 1, 0, 14, 12, 12, 11, 10, 10, 8, 6, 7, 5, 4, 4, 3, 3},
		{3, 2, 0, 14, 13, 12, 12, 11, 10, 8, 8, 7, 5, 4, 4, 3, 2},
		{3, 2, 1, 15, 14, 12, 11, 10, 10, 7, 7, 5, 4, 3, 4},
		{3, 1, 0, 14, 13, 12, 12, 12, 10, 10, 8, 8, 7, 7, 5, 4, 3, 3, 2},	//71st data  
		{4, 2, 0, 15, 14, 13, 12, 12, 10, 12, 9, 8, 6, 5, 4, 3, 3},
		{2, 2, 0, 0, 14, 13, 12, 11, 11, 10, 9, 7, 6, 5, 4, 3, 2},
		{1, 15, 14, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},
		{2, 1, 1, 15, 13, 12, 11, 10, 11, 8, 8, 8, 5, 4, 4, 2},
		{3, 2, 0, 0, 14, 13, 11, 11, 8, 7, 6, 4, 3, 3},
		{2, 0, 0, 14, 14, 13, 12, 11, 10, 10, 9, 7, 6, 4, 4, 3, 3},
		{3, 2, 1, 0, 14, 13, 11, 10, 10, 10, 8, 7, 6, 4, 4, 3, 3},
		{2, 1, 15, 13, 13, 12, 10, 10, 9, 8, 7, 4, 4, 4, 3 },
		{3, 3, 0, 15, 0, 13, 12, 12, 10, 9, 9, 7, 6, 4, 3},				//80st data  
		{2, 2, 1, 0, 14, 13, 13, 11, 10, 10, 9, 7, 7, 5, 4, 4, 3},
		{2, 1, 0, 0, 14, 12, 13, 11, 11, 10, 10, 9, 8, 8, 6, 6, 5, 5, 4, 4, 4, 3, 2},
		{2, 2, 1, 15, 13, 12, 11, 11, 11, 10, 9, 7, 7, 6, 5, 4, 3, 3 },
		{2, 2, 1, 0, 15, 13, 12, 12, 11, 11, 9, 8, 8, 6, 4, 4, 3, 3 }, 
		{2, 0, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 6, 5, 4, 4, 3},
		{2, 1, 0, 15, 15, 13, 12, 11, 12, 10, 10, 10, 8, 8, 7, 7, 5, 4, 4, 3, 3, 2, 2, 2},
		{1, 1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 3},					
		{1, 1, 15, 14, 13, 12, 10, 10, 9, 7, 7, 7, 5, 4, 3, 3},
		{4, 3, 0, 3, 15, 13, 13, 12, 11, 9, 7, 7, 5, 4, 4},
		{3, 2, 1, 0, 0, 15, 13, 13, 12, 11, 10, 11, 7, 8, 6, 5, 4, 4, 3},
		{3, 2, 0, 0, 13, 12, 12, 10, 10, 9, 6, 5, 4, 3 },
		{2, 1, 0, 15, 13, 12, 11, 11, 9, 8, 6, 6, 5, 4},
		{2, 0, 15, 14, 12, 12, 10, 9, 9, 7, 6, 5, 5, 4, 3},
		{2, 1, 0, 15, 14, 12, 10, 10, 9, 7, 7, 5, 4, 3},
		{3, 2, 0, 14, 13, 12, 10, 9, 7, 6, 5, 3},
		{3,2,0,14,13,12,10,9,7,6,5,3},
		{3, 2, 2, 0, 15, 14, 12, 12, 11, 9, 8, 8, 6, 5, 4},
		{3, 2, 1, 0, 15, 13, 13, 12, 11, 10, 8, 8, 7, 6, 5, 4, 3},
		{2, 1, 0, 15, 14, 13, 12, 11, 10, 10, 10, 7, 7, 6, 4, 4, 4, 3, 3},
		{4, 4, 2, 2, 1, 15, 13, 13, 12, 10, 10, 10, 8, 7, 5, 4, 3, 3},	
		{3, 4, 3, 2, 2, 1, 0, 15, 15, 15, 13, 12, 12, 12, 11, 10, 8, 9, 9, 8, 8, 6, 4, 4, 4, 3},
		{2, 1, 0, 14, 13, 12, 11, 11, 10, 8, 7, 6, 5, 4, 3},		
		{2, 2, 0, 15, 14, 13, 12, 12, 11, 11, 10, 8, 8, 6, 5, 5, 4, 3},
		{2, 2, 2, 15, 15, 14, 13, 12, 11, 10, 9, 8, 8, 6, 4, 5, 4, 4, 3},
		{2, 1, 0, 14, 15, 13, 12, 11, 11, 9, 8, 8, 6, 5, 4, 4, 3},
		{2, 1, 0, 14, 14, 12, 12, 11, 10, 10, 7, 8, 8, 5, 4, 4, 4 }, 
		{2, 1, 0, 15, 13, 13, 11, 10, 10, 8, 9, 6, 4, 5, 4, 3},
		{3, 2, 0, 2, 15, 15, 13, 13, 12, 11, 10, 8, 8, 7, 7, 4, 6, 5, 4, 3, 3},
		{2, 1, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 7, 4, 7, 4, 5, 4, 2},	
		{2, 2, 15, 14, 14, 12, 11, 10, 9, 8, 6, 6, 5, 4, 3, 2},
		{2, 1, 1, 15, 15, 13, 12, 11, 10, 9, 8, 7, 5, 5, 4, 4, 3 },
		{3, 1, 0, 15, 14, 13, 12, 11, 10, 8, 7, 4, 6, 4, 3},
		{2, 1, 1, 14, 13, 13, 12, 12, 10, 8, 8, 5, 5, 4, 3, 3 }, 
		{2, 0, 1, 14, 14, 14, 13, 12, 11, 10, 10, 10, 7, 6, 6, 5, 4, 4, 3},
		{3, 1, 2, 15, 15, 13, 13, 13, 12, 10, 10, 10, 8, 7, 6, 5, 4, 3, 3, 3},
		{2, 0, 1, 14, 15, 15, 13, 12, 10, 11, 9, 9, 7, 7, 6, 5, 5, 3, 3, 3 },
		{2, 1, 0, 0, 14, 13, 12, 12, 10, 10, 9, 7, 8, 6, 5, 4, 4, 3, 3 },
		{2, 1, 0, 15, 14, 13, 12, 11, 11, 10, 10, 9, 10, 6, 7, 5, 7, 5, 4, 3, 2},
		{2, 2, 0, 15, 14, 13, 12, 11, 11, 10, 8, 8, 6, 5, 4, 3, 3},
		{2, 0, 15, 13, 13, 12, 11, 10, 8, 8, 6, 6, 5, 4, 4, 4, 2},
		{2, 2, 1, 15, 13, 12, 11, 10, 10, 9, 7, 6, 4, 5, 4 },
		{4, 1, 2, 15, 15, 13, 12, 10, 10, 9, 6, 8, 6, 4, 4 },
		{2, 1, 1, 15, 14, 14, 12, 11, 10, 10, 9, 8, 7, 5, 5, 4, 3 },
		{3, 1, 2, 14, 0, 14, 12, 12, 11, 10, 10, 8, 6, 5, 5, 4},
		{2, 2, 15, 15, 13, 13, 12, 10, 10, 9, 7, 8, 6, 4, 4, 4, 4, 3},		
		{2, 2, 1, 0, 15, 15, 14, 13, 13, 11, 11, 10, 10, 9, 9, 6, 8, 6, 5, 5, 4, 4 },
		{2, 2, 1, 0, 14, 14, 12, 13, 11, 11, 10, 9, 9, 7, 6, 5, 4, 4, 4, 3},
		{2, 2, 1, 0, 14, 14, 13, 14, 11, 10, 10, 10, 8, 7, 6, 6, 5, 4, 4, 3},		
		{3,2,1,0,0,0,0,14,15,13,15,12,13,12,12,11,10,11,8,9,8,8,8,7,7,8,6,5,4,4,4,4,3,3},	
		{1, 0, 1, 0, 0, 15, 13, 14, 13, 12, 11, 11, 11, 10, 9, 9, 7, 8, 7, 6, 5, 4, 4, 3, 3 },
		{2, 1, 1, 0, 0, 14, 13, 0, 13, 13, 12, 11, 12, 10, 7, 9, 8, 7, 8, 7, 6, 4, 4, 4, 3, 3 },
		{1, 0, 0, 0, 15, 15, 13, 13, 13, 12, 12, 11, 10, 9, 9, 7, 8, 8, 7, 6, 5, 4, 4, 4, 3},
		{ 2, 1, 1, 0, 15, 13, 0, 13, 12, 12, 11, 12, 10, 8, 10, 8, 7, 8, 7, 6, 4, 4, 4, 4, 3, 3},												
		{3,2,1,0,15,0,0,14,14,12,13,12,12,11,10,10,9,8,8,7,7,6,4,4,4,4,3},
		{1,1,0,0,15,14,14,12,12,11,11,9,10,8,8,7,7,6,5,4,3},		
		{2, 2, 0, 0, 15, 14, 14, 12, 12, 11, 11, 10, 9, 10, 7, 7, 7, 6, 4, 5, 4},
		{2, 2, 0, 0, 15, 14, 14, 13, 12, 11, 10, 10, 7, 8, 8, 8, 5, 5, 4, 4, 4 },
		{2, 2, 1, 0, 14, 13, 12, 12, 11, 11, 9, 8, 8, 7, 6, 4, 4, 4},
		{3, 2, 2, 1, 0, 15, 13, 13, 12, 11, 11, 11, 8, 8, 8, 7, 5, 5, 4, 4},
		{2, 2, 1, 0, 15, 13, 13, 11, 10, 9, 8, 8, 7, 5, 4, 4 },
		{2, 3, 1, 1, 15, 15, 13, 12, 11, 12, 9, 9, 8, 6, 6, 4, 4 },
		{3, 2, 1, 0, 15, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 6, 5, 4, 4},
		{3, 3, 1, 2, 0, 15, 14, 15, 13, 13, 11, 12, 10, 9, 8, 8, 7, 6, 5, 4, 4},
		{2, 2, 1, 0, 14, 13, 13, 11, 11, 10, 9, 7, 7, 5, 4, 4},
		{2, 2, 2, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3 },
		{1, 1, 0, 0, 13, 12, 11, 10, 9, 7, 7, 6, 4, 3 },
		{2, 3, 1, 0, 0, 15, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4},
		{2, 1, 0, 0, 13, 13, 12, 10, 10, 10, 8, 6, 6, 5, 4, 3},
		{2, 1, 1, 0, 15, 14, 13, 12, 10, 9, 10, 7, 7, 6, 4, 4, 3, 4},
		{2, 3, 1, 0, 0, 15, 14, 12, 11, 10, 10, 8, 7, 7, 6, 4, 4, 3},
		{2, 2, 2, 14, 15, 14, 14, 13, 11, 11, 10, 10, 8, 8, 7, 6, 4, 5, 3, 3},
		{1, 1, 15, 14, 13, 12, 11, 9, 9, 5, 5, 3 },
		{3, 2, 2, 0, 15, 15, 13, 13, 12, 12, 10, 12, 10, 10, 8, 6, 8, 5, 5},
		{2, 2, 0, 14, 15, 15, 13, 13, 11, 11, 11, 10, 9, 9, 6, 6, 5, 5, 4, 4 },//154th data circledata end.
	};
	Mat trainingDataMatTemp(noOfTrainingSamplesSquare, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatSquare);

}

void GestureRecognizer::triangleTrainingData(){
	float trainingDataTotal[noOfTrainingSamplesTriangle][50] = {
		//{4, 4, 4, 4, 4, 4, 13, 13, 13, 13, 8, 7, 8, 7, 8},	//1st triangle data with different direction.(Right side)
	//	{5, 4, 5, 5, 4, 4, 14, 13, 13, 13, 8, 7, 8, 7, 8},	//triangle data
	//	{3, 4, 3, 3, 4, 4, 14, 13, 13, 13, 8, 7, 8, 7, 8},	//triangle data				 
	//	{4,4,4,3,4,3,15,15,13,13,14,13,13,13,13,8,8,8,8,7,7,8},	//triangle data
	//	{4,4,4,4,3,4,3,13,14,13,14,13,13,13,9,8,8,8,8,8},	//triangle data
	//	{3, 3, 3, 3, 3, 14, 14, 14, 13, 13, 14, 14, 14, 14, 7, 7, 8, 8, 8, 7, 8},	//triangle data
	//	{4, 4, 4, 4, 3, 15, 15, 14, 13, 13, 13, 13, 13, 13, 13, 8, 8, 8, 7, 8, 7, 8},	//triangle data
	//	{4, 4, 3, 3, 3, 14, 13, 14, 13, 13, 13, 13, 13, 13, 8, 8, 8, 7, 7},	//triangle data
	//	{3, 3, 4, 4, 3, 15, 13, 13, 13, 13, 13, 13, 13, 9, 8, 8, 7, 7 },	//triangle data
	//	{4, 4, 4, 3, 4, 3, 15, 13, 13, 13, 13, 13, 13, 13, 8, 8, 8, 8, 8, 7, 7},	//triangle data
	//	{4, 4, 4, 4, 4, 3, 15, 14, 13, 13, 13, 13, 13, 13, 9, 8, 8, 8, 7, 7},	//triangle data
	//	{3, 3, 3, 4, 3, 3, 4, 3, 14, 13, 14, 14, 13, 14, 14, 13, 13, 9, 9, 9, 8, 8, 7, 8},	//triangle data
	//	{3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14, 13, 13, 13, 13, 8, 8, 8, 7},
	//	{3, 3, 3, 3, 4, 3, 13, 13, 13, 14, 14, 14, 9, 8, 8, 7, 8},
	//	{3, 3, 3, 3, 3, 13, 13, 13, 14, 13, 13, 8, 8, 9, 8, 8, 7},
	//	{3, 3, 4, 4, 3, 15, 14, 14, 13, 13, 13, 13, 13, 14, 13, 9, 8, 8, 7, 7, 8, 7 },
	//	{3, 4, 4, 4, 4, 3, 14, 14, 13, 13, 13, 13, 13, 13, 8, 8, 8, 8, 8 },
	//	{3, 3, 3, 3, 3, 15, 14, 13, 13, 13, 13, 13, 9, 8, 8, 8, 7, 8, 8},
	//	{4, 4, 3, 4, 4, 15, 14, 14, 13, 13, 13, 13, 13, 13, 8, 8, 7, 8, 7, 7, 7 },
	//	{3, 3, 3, 3, 3, 15, 14, 13, 13, 13, 13, 13, 13, 8, 9, 8, 8, 8, 7, 8, 7},
	//	{3, 3, 3, 3, 3, 3, 14, 13, 13, 13, 13, 13, 13, 8, 9, 8, 8, 8, 7, 8, 7},
	//	{3, 3, 3, 3, 3, 3, 3, 3, 14, 13, 13, 13, 13, 13, 13, 8, 8, 8, 8, 7, 7, 8, 7, 7 },
	//	{3, 3, 3, 3, 3, 3, 3, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 9, 8, 7, 14, 8, 7, 8, 7},
	//	{3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 8, 8, 8, 8, 7, 8, 7, 8},
	//	{3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 13, 13, 9, 8, 8, 7, 7},
	//	{4, 4, 4, 4, 4, 4, 4, 3, 3, 15, 14, 13, 13, 13, 13, 13, 13, 13, 8, 8, 8, 14, 7, 9, 7, 7},	//triangle data
	//	{3,3,3,3,3,3,13,13,13,13,13,13,13,13,9,9,7,8,7,7},	
	//	{3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 13, 13, 9, 8, 8, 7, 7, 7, 8},
	//	{ 3, 3, 3, 3, 3, 3, 3, 14, 14, 13, 13, 13, 13, 13, 13, 13, 7, 8, 8, 8, 7, 8},
	//	{4, 4, 4, 4, 4, 4, 3, 3, 14, 14, 14, 13, 13, 13, 13, 13, 13, 9, 9, 8, 9, 8, 9, 7},
	//	{4, 3, 4, 4, 4, 3, 14, 13, 13, 13, 13, 13, 13, 13, 9, 7, 8, 8, 7, 7, 7 },
	//	{4, 4, 4, 4, 4, 4, 4, 3, 3, 13, 13, 13, 13, 14, 14, 14, 7, 8, 7, 8, 9, 8 },
	//	{3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 14, 13, 13, 13, 13, 8, 8, 8, 7, 7, 7 },
	//	{3, 3, 4, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 7, 8, 8, 8, 7, 8},
	//	{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 9, 7, 7, 8, 8, 8, 8, 7, 8 },
	//	{3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 13, 14, 13, 13, 13, 13, 13, 9, 8, 8, 7, 8, 8, 8},
	//	{4, 4, 4, 4, 4, 4, 15, 14, 13, 13, 13, 13, 9, 8, 9, 8, 7, 9 },
	//	{4, 3, 3, 4, 3, 15, 14, 13, 13, 13, 15, 13, 13, 13, 13, 13, 7, 7, 8, 8, 8, 7, 7},
	//	{ 4, 3, 3, 4, 4, 15, 14, 13, 13, 14, 13, 13, 13, 8, 8, 7, 8, 7, 7, 9},		 
	//	{4, 3, 3, 3, 4, 3, 15, 13, 13, 13, 13, 14, 13, 13, 13, 8, 8, 8, 8, 7, 7, 8},
	//	{4, 4, 4, 4, 3, 3, 3, 13, 14, 13, 13, 13, 14, 13, 13, 13, 9, 9, 7, 8, 7, 8, 7},
	//	{3, 4, 4, 3, 3, 3, 3, 14, 13, 13, 13, 13, 13, 14, 13, 13, 13, 7, 8, 8, 8, 9, 7, 8 },
	//	{4, 4, 4, 4, 3, 3, 3, 3, 3, 13, 13, 13, 14, 13, 13, 14, 13, 13, 13, 7, 8, 7, 8, 7, 9},
	//	{4, 4, 3, 3, 4, 4, 3, 3, 14, 13, 13, 13, 13, 13, 13, 13, 13, 8, 9, 7, 7, 8, 7, 8},
	//	{3, 3, 4, 3, 4, 3, 3, 15, 13, 13, 13, 14, 14, 13, 13, 13, 13, 8, 8, 7, 8, 8, 8, 8, 7},
	//	{4, 3, 4, 4, 3, 3, 15, 14, 14, 13, 13, 14, 13, 13, 13, 13, 8, 8, 9, 8, 7, 7, 9, 7, 8 },
	//	{4, 3, 3, 3, 3, 3, 3, 3, 3, 14, 13, 13, 13, 13, 13, 14, 13, 13, 9, 8, 8, 7, 8, 8, 8, 7, 7},
	//	{4, 3, 3, 4, 3, 4, 3, 15, 13, 13, 13, 13, 14, 13, 13, 13, 9, 9, 8, 8, 7, 8, 8, 9},//53nd triangle data
	//	{3, 3, 4, 4, 4, 3, 13, 13, 13, 13, 14, 13, 9, 7, 8, 7, 7, 8, 8},
	//	{4, 4, 4, 3, 4, 3, 14, 13, 13, 13, 13, 13, 13, 9, 7, 7, 7, 8},
	//	{4, 4, 4, 3, 3, 13, 13, 13, 13, 13, 13, 9, 7, 9, 7, 8},
	//	{4, 4, 4, 4, 4, 4, 15, 13, 13, 13, 14, 13, 13, 13, 8, 8, 8, 7, 7, 8} ,
	//	{4, 3, 4, 3, 4, 3, 3, 13, 13, 13, 13, 13, 13, 13, 8, 8, 8, 7, 8, 7},	//53th triangle data
		
		{3, 4, 3, 4, 4, 3, 9, 10, 10, 10, 9, 10, 11, 12, 15, 15, 0, 0, 15, 0 },		//+ve triangle data with different direction.(Left side)
		{4, 4, 4, 4, 3, 3, 4, 9, 10, 10, 10, 10, 10, 10, 11, 11, 0, 15, 0, 15, 0, 0, 1},
		{4, 4, 4, 4, 3, 3, 3, 9, 9, 10, 11, 11, 11, 10, 10, 11, 10, 15, 0, 0, 0, 0, 0},
		{4, 3, 4, 4, 9, 10, 10, 10, 11, 10, 0, 15, 0, 0, 0},
		{4, 4, 4, 4, 4, 3, 4, 11, 10, 10, 10, 10, 10, 10, 11, 12, 0, 15, 0, 0, 0},
		{4, 4, 4, 3, 4, 4, 3, 10, 10, 11, 10, 10, 10, 10, 12, 15, 0, 0, 0, 0},
		{4, 4, 4, 4, 4, 4, 4, 4, 3, 8, 10, 10, 10, 10, 10, 10, 11, 10, 0, 15, 0, 0, 0, 0, 15, 0, 0},
		{4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 15, 0, 0, 0, 0, 0, 0, 15},
		{4, 4, 4, 4, 3, 4, 9, 10, 10, 10, 10, 10, 10, 11, 0, 0, 15, 0, 0, 0, 0, 0},
		{4, 4, 3, 4, 3, 4, 10, 11, 10, 10, 10, 10, 10, 9, 11, 15, 0, 0, 0, 0, 0, 0},
		{3, 4, 4, 4, 4, 4, 10, 10, 10, 10, 11, 10, 11, 15, 0, 0, 0, 15, 0, 15},
		{4, 4, 3, 4, 3, 4, 10, 10, 10, 10, 10, 10, 10, 0, 0, 15, 0 },
		{4, 3, 4, 3, 4, 4, 5, 8, 10, 10, 10, 10, 11, 11, 11, 10, 11, 0, 0, 0, 0, 15, 0, 15},
		{4, 4, 4, 4, 3, 4, 4, 9, 10, 10, 11, 11, 10, 10, 15, 0, 15, 15, 0, 0},
		{3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 15, 0, 0, 0, 0, 0, 0, 15, 1},
		{4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 11, 10, 11, 9, 10, 10, 15, 0, 15, 0, 0, 0},
		{4, 4, 3, 4, 4, 3, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 15, 0, 0, 0, 0 },
		{4, 4, 4, 3, 4, 10, 10, 10, 10, 10, 10, 15, 15, 15, 0, 0 },
		{4, 4, 4, 4, 4, 4, 3, 4, 3, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 15, 0, 15, 0, 15, 0, 0, 0, 0},
		{4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 15, 0, 15, 15, 0, 0 },			//20th data end.
   
		{0, 15, 0, 14, 0, 3, 4, 6, 8, 9, 12, 15, 0, 14, 15, 0, 10, 9, 8, 7},		//random -ve samples 0th
		{3, 13, 12, 13, 12, 2, 3, 2, 3, 2, 13, 12},		 
		{3, 12, 11, 12, 11, 5, 4, 3, 0, 13, 12},
		{8, 4, 13, 3, 1, 0, 1, 7, 8, 7, 9, 4, 2, 1, 0, 15, 8},  
		{3, 4, 3, 13, 12, 13, 12, 4, 3, 4, 3, 2, 3, 2, 0, 13, 12, 8, 4 },
		{2, 1, 0, 1, 2, 12, 11, 13, 12, 8, 6, 8, 7, 8 },
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
		{4,12,11},					//45th -ve random data..
	};

	Mat trainingDataMatTemp(noOfTrainingSamplesTriangle, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatTriangle);
}

void GestureRecognizer::LetterSTrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesLetterS][50] = {	
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
		{8, 7, 9, 10, 11, 14, 15, 0, 15, 13, 12, 12, 10, 10, 8, 8, 7, 5, 4 },		
		{9, 11, 12, 13, 14, 12, 12, 11, 9, 8, 5},
		{7, 10, 12, 12, 13, 15, 13, 12, 11, 10, 9, 7},
		{6, 8, 11, 12, 12, 14, 14, 14, 11, 11, 8, 8, 6},
		{9, 11, 12, 14, 15, 12, 11, 10, 9, 7},
		{7, 10, 10, 12, 12, 13, 13, 12, 10, 9, 8},
		{9, 11, 12, 13, 13, 13, 12, 10, 9, 8, 6 },
		{10, 11, 12, 14, 13, 12, 11, 10, 9, 8},
		{8, 11, 12, 12, 14, 13, 11, 10, 10, 8, 6},
		{8, 11, 12, 12, 14, 13, 11, 10, 10, 8, 6},
		{9, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8},
		{10, 11, 12, 13, 13, 13, 12, 10, 9, 7},
		{8, 10, 11, 12, 13, 15, 14, 13, 12, 11, 11, 9, 8, 7},
		{10, 12, 13, 12, 12, 10, 8},
		{10, 11, 12, 13, 13, 12, 11, 10, 8 },
		{9, 11, 12, 13, 13, 13, 11, 10, 9},
		{7, 6, 9, 10, 12, 12, 12, 13, 15, 15, 14, 13, 12, 12, 10, 9, 8, 7, 7},
		{9, 11, 13, 14, 13, 11, 11, 10, 7},
		{10, 11, 13, 13, 12, 11, 11, 9, 7},
		{8, 8, 11, 11, 13, 15, 15, 14, 13, 12, 11, 10, 8, 7, 8, 6, 5},
		{8, 10, 11, 13, 14, 15, 13, 12, 10, 10, 9, 8},
		{8, 11, 12, 13, 14, 12, 11, 10, 10, 9, 8},
		{10, 13, 14, 12, 10, 10, 8},
		{8, 10, 12, 13, 13, 12, 11, 10, 9, 7 },
		{10, 12, 14, 13, 12, 11, 9, 8},
		{10, 11, 12, 14, 13, 12, 11, 10, 8, 8},
		{9, 11, 13, 15, 13, 11, 11, 9, 8, 6},
		{7, 8, 11, 13, 0, 15, 13, 12, 10, 8, 7, 7, 5},		
		{8, 11, 12, 13, 14, 15, 13, 11, 10, 9, 7, 8},
		{7, 9, 10, 12, 14, 14, 13, 12, 11, 10, 8, 6},
		{8, 9, 12, 13, 14, 14, 12, 11, 10, 8, 7},
		{8, 10, 12, 12, 13, 13, 14, 13, 11, 10, 9, 8, 7 },
		{6, 10, 11, 12, 14, 14, 12, 12, 10, 9, 8, 7},
		{9, 10, 11, 13, 0, 13, 12, 11, 10, 9, 8, 8},
		{9, 11, 11, 12, 13, 14, 13, 12, 11, 11, 10, 9, 8, 8},
		{8, 10, 11, 12, 13, 12, 12, 11, 9, 9},					
		{9, 11, 14, 13, 12, 11, 11, 9, 8},
		{10, 10, 12, 14, 12, 11, 10, 9, 7},
		{10, 11, 13, 14, 12, 12, 11, 10, 8, 7},
		{10, 11, 13, 13, 13, 11, 10, 9, 8, 6},
		{8, 9, 10, 12, 14, 15, 13, 12, 11, 10, 8, 7, 6},
		{8, 10, 12, 15, 15, 13, 12, 10, 10, 7, 7},
		{10, 12, 13, 12, 11, 11, 10, 8},
		{6, 9, 10, 12, 13, 14, 13, 11, 11, 10, 8, 7 },
		{9, 11, 12, 15, 14, 12, 11, 11, 9, 8, 8},		
		{10, 13, 13, 13, 11, 11, 9, 7},
		{10, 12, 13, 13, 12, 11, 10, 8, 7, 7, 12},
		{9, 11, 12, 14, 13, 12, 11, 10, 9, 8},
		{8, 10, 12, 14, 14, 13, 10, 9, 8, 6},
		{8, 10, 13, 15, 14, 13, 10, 9, 8, 6},
		{10, 12, 14, 13, 12, 10, 9, 8, 6},
		{10, 12, 12, 12, 10, 9, 9},
		{8, 11, 13, 13, 14, 12, 10, 8, 8, 8},
		{10, 12, 12, 14, 12, 11, 10, 8, 8, 8},
		{9, 12, 14, 12, 11, 10, 9, 7},				
		{8, 12, 12, 15, 13, 12, 11, 11, 9, 9, 7},
		{10, 12, 12, 14, 12, 12, 10, 9, 8, 8},
		{9, 11, 12, 14, 13, 11, 11, 11, 10, 8, 7 },
		{10, 11, 13, 13, 13, 12, 11, 10, 10, 8, 7},
		{10, 11, 12, 0, 12, 11, 10, 8, 7},
		{10, 13, 13, 12, 10, 10, 8, 7},
		{8, 11, 12, 12, 14, 13, 11, 10, 10, 8, 8},
		{8, 10, 12, 13, 14, 13, 12, 11, 9, 8, 8, 8},
		{10, 12, 13, 13, 11, 11, 9, 7},
		{8, 11, 12, 14, 13, 12, 11, 10, 8},
		{7, 10, 12, 14, 14, 12, 11, 10, 9, 7, 7},
		{9, 11, 13, 12, 13, 11, 12, 11, 10, 8, 8},
		{8, 10, 12, 15, 14, 12, 11, 11, 9, 8, 7},
		{10, 11, 13, 13, 13, 12, 11, 10, 9, 8, 7},		//100st +ve data end

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
	Mat trainingDataMatTemp(noOfTrainingSamplesLetterS, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatLetterS);

}

void GestureRecognizer::Digit1TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit1][50] = {	
		{2, 2, 2, 1, 13, 12, 12, 12, 12, 12, 12, 12, 11, 12 },			//+ve data 
		{3, 2, 3, 2, 2, 13, 12, 12, 12, 12, 12, 11, 11, 11 },
		{3, 2, 3, 3, 2, 13, 12, 12, 12, 12, 12, 12, 12, 11 },
		{3, 2, 3, 13, 12, 12, 12, 12, 12, 11},
		{2, 2, 3, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{2, 2, 2, 1, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11},
		{2, 2, 2, 0, 12, 12, 12, 11, 11, 12, 12, 12, 11 },
		{3, 1, 2, 13, 12, 12, 12, 12, 11},
		{2, 3, 12, 12, 12, 11, 11, 12 },
		{2, 2, 13, 12, 12, 12, 12, 11, 12, 12 },
		{2, 2, 13, 12, 12, 12, 12, 11, 12, 12 },
		{2, 2, 13, 12, 12, 11, 12, 11 },
		{2, 2, 2, 2, 12, 12, 12, 12, 11, 12, 11},
		{3, 2, 2, 12, 12, 12, 12, 12, 11, 12, 11 },
		{2, 2, 2, 1, 12, 12, 12, 12, 12, 11, 12, 11},
		{2, 2, 2, 2, 1, 12, 12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12},
		{3, 3, 2, 2, 12, 12, 12, 12, 12, 11, 12 },
		{3, 2, 2, 12, 12, 11, 12, 12, 12, 12, 11 },
		{3, 2, 3, 2, 12, 12, 12, 11, 12, 12, 12, 12},
		{2, 2, 2, 12, 11, 11, 12, 12, 12, 12},
		{2, 2, 2, 13, 12, 11, 11, 12, 12, 12, 12, 12 },			
		{2,2,2,11,12,12,12,12,12,12,11,12,11},	
		{2, 2, 2, 12, 12, 12, 11, 12, 11, 10},
		{1, 2, 2, 2, 12, 12, 12, 11, 12, 12, 11, 12, 12, 10 },
		{2, 3, 2, 2, 13, 12, 12, 11, 12, 11, 12, 10 },
		{2, 3, 2, 2, 12, 12, 12, 12, 11, 12, 12, 11, 12},
		{2, 3, 2, 2, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11},
		{2, 2, 2, 2, 2, 12, 11, 12, 11, 11, 12, 11, 12, 11 },
		{2, 2, 2, 2, 2, 12, 12, 11, 11, 12, 12, 12, 12, 12, 11, 11},
		{2, 2, 2, 1, 2, 12, 11, 12, 12, 11, 12, 12, 11, 11, 10 },
		{2, 2, 2, 2, 12, 12, 11, 12, 12, 12, 12, 11, 11 },
		{2, 2, 2, 3, 2, 12, 11, 12, 12, 11, 11, 11, 12, 10 },		//32 th data end..
		{2, 1, 2, 12, 12, 12, 12, 12, 11, 11, 12},
		{1, 3, 0, 14, 12, 11, 12, 12, 12, 11, 11, 11 },
		{1, 2, 2, 13, 12, 12, 11, 12, 11, 12, 11, 12 },
		{2, 2, 2, 2, 13, 12, 12, 12, 12, 12, 11, 12, 11, 11},
		{2, 1, 1, 2, 13, 11, 11, 12, 11, 11, 11, 12, 11},
		{2, 0, 2, 0, 12, 12, 12, 12, 12, 11, 12, 12, 11, 12},
		{3, 0, 2, 2, 3, 13, 12, 11, 12, 12, 12, 11, 12, 12, 10 },
		{1, 2, 1, 1, 11, 11, 12, 12, 12, 11, 11, 11 },
		{2, 1, 2, 2, 3, 12, 12, 12, 12, 11, 11, 11, 12},
		{2, 1, 2, 13, 12, 11, 12, 11, 11, 12, 11, 12, 11},		
		{3, 2, 2, 2, 12, 12, 11, 11, 12, 12, 12, 12, 11},
		{3, 2, 2, 2, 2, 12, 12, 12, 12, 12, 11, 12, 11, 11, 12, 11, 12, 12},		//44nd data end
		{2, 2, 12, 11, 12, 12, 12, 12, 12, 12, 12, 11},
		{2, 2, 0, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11 },
		{2, 2, 12, 11, 12, 11, 12, 12, 12, 11, 11},
		{2, 2, 2, 13, 12, 11, 11, 11, 12, 12, 12, 12, 12 },		//48nd data end
		{2, 2, 0, 11, 11, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11},	
		{1, 2, 13, 12, 12, 12, 11, 12, 12, 11, 12},
		{2, 2, 2, 12, 12, 11, 12, 11, 12, 11, 12, 12, 12, 12},
		{2, 2, 13, 11, 12, 12, 12, 12, 12, 11, 12},
		{2, 2, 13, 12, 12, 11, 13, 10, 11, 12, 13},
		{2, 2, 3, 12, 12, 11, 11, 12, 12, 11, 12, 12, 12, 12, 11},	
		{1, 2, 13, 12, 12, 12, 12, 11, 12, 11},
		{2, 2, 13, 12, 11, 12, 11, 12, 12, 12},
		{2, 2, 1, 12, 12, 12, 11, 11, 12, 12},
		{2, 2, 12, 12, 12, 12, 12, 12, 12, 12},
		{3, 2, 2, 12, 12, 11, 12, 12, 12, 12, 11, 12},
		{2, 2, 13, 12, 12, 12, 12, 11, 12, 12, 12},
		{2, 2, 13, 11, 12, 11, 12, 11, 11, 12},
		{2, 2, 13, 12, 12, 11, 12, 12, 12},
		{2, 3, 2, 13, 11, 12, 12, 11, 11, 12, 12},
		{2, 2, 12, 12, 11, 11, 12, 12},
		{2, 2, 13, 12, 12, 12, 11, 12, 11},
		{2, 2, 2, 12, 12, 11, 12, 12, 11},
		{2, 2, 2, 2, 4, 12, 11, 12, 12, 12, 12, 12, 12, 12},
		{2, 2, 2, 12, 12, 12, 12, 12, 12, 11},
		{2, 2, 13, 12, 11, 12, 12, 11, 12, 12},
		{2, 2, 13, 11, 12, 12, 12, 12, 12},
		{2, 3, 15, 12, 11, 12, 12, 12, 12, 12},		
		{2, 3, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{2, 2, 2, 12, 12, 11, 12, 12, 12, 12, 12, 11},
		{2, 2, 13, 12, 11, 12, 12, 12, 11, 12},
		{2, 2, 12, 12, 11, 12, 12, 12},
		{2, 2, 2, 12, 12, 11, 12, 11, 12, 12, 11},
		{2, 2, 2, 12, 12, 12, 11, 12, 11, 12, 11, 11},
		{2, 2, 12, 12, 12, 12, 11, 12, 12 },
		{2, 2, 3, 13, 12, 12, 12, 11, 12, 12},
		{3, 2, 13, 12, 12, 12, 11},		//80nd data end

		 
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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit1, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit1);

}

void GestureRecognizer::Digit2TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit2][50] = {	
		
		{1, 2, 1, 1, 0, 15, 12, 12, 11, 10, 11, 9, 9, 11, 7, 9, 0, 14, 0, 0, 0, 0, 15},		//+ve digit2 data
		{2, 2, 1, 0, 0, 14, 12, 12, 10, 10, 9, 9, 10, 9, 9, 14, 0, 0, 0, 15, 0, 0, 0, 15},
		{1, 2, 0, 0, 15, 13, 12, 11, 11, 10, 10, 8, 9, 10, 0, 0, 0, 0, 0, 0},
		{3, 2, 0, 0, 0, 14, 12, 12, 10, 10, 9, 9, 10, 10, 15, 0, 0, 15, 0, 0, 0, 0},
		{2, 2, 1, 0, 15, 13, 12, 10, 10, 10, 9, 10, 10, 15, 0, 0, 1, 0, 15 },
		{1, 1, 0, 0, 14, 12, 12, 10, 10, 10, 9, 9, 10, 10, 15, 15, 1, 0, 0, 0, 15},
		{2, 2, 1, 1, 0, 0, 13, 12, 11, 10, 10, 10, 10, 10, 9, 15, 0, 0, 15, 0, 15},
		{2, 2, 0, 0, 14, 12, 12, 11, 10, 9, 9, 10, 10, 15, 15, 0, 15, 1},
		{2, 1, 0, 15, 14, 13, 11, 11, 10, 9, 10, 10, 9, 11, 11, 0, 0, 0, 0, 1},
		{2, 2, 0, 0, 15, 14, 12, 12, 10, 10, 10, 10, 10, 10, 10, 14, 14, 1, 15, 1, 0 },
		{ 2, 1, 0, 14, 13, 12, 10, 10, 9, 9, 15, 0, 0, 0, 15, 0},
		{2, 1, 0, 13, 11, 11, 10, 10, 9, 14, 15, 0, 15}, 
		{2, 0, 0, 14, 13, 11, 11, 10, 9, 10, 10, 10, 15, 0, 0, 15, 15, 1},
		{1, 1, 15, 15, 13, 12, 11, 10, 10, 10, 10, 9, 15, 0, 0, 0, 0, 15, 0, 0},
		{1, 1, 0, 15, 13, 11, 10, 11, 10, 9, 10, 12, 0, 0, 0, 0, 15, 0, 15},
		{3, 1, 1, 0, 14, 11, 11, 10, 10, 10, 10, 10, 10, 15, 0, 0, 15, 0, 15},
		{ 2, 3, 2, 1, 0, 15, 13, 12, 10, 10, 9, 10, 11, 10, 14, 1, 0, 0, 0, 0, 15},
		{3, 3, 2, 2, 0, 0, 0, 12, 12, 11, 10, 10, 10, 10, 15, 0, 15, 0, 0},
		{3, 2, 1, 0, 0, 14, 12, 12, 10, 11, 10, 9, 10, 10, 15, 15, 0, 0, 15 },	//// 19th +ve digit2 data
		{3, 2, 0, 15, 13, 12, 12, 11, 10, 10, 10, 10, 0, 0, 0, 15, 0},
		{2, 1, 3, 0, 15, 13, 13, 11, 10, 11, 10, 10, 9, 9, 0, 0, 0, 0, 15},
		{2, 2, 0, 15, 14, 12, 11, 10, 10, 10, 10, 14, 0, 15, 15 },
		{3, 0, 1, 15, 15, 13, 11, 10, 10, 10, 10, 10, 8, 15, 0, 0, 0, 0, 0 },
		{2, 2, 1, 0, 15, 13, 12, 11, 10, 10, 10, 11, 10, 10, 8, 0, 0, 0, 0, 15, 0},
		{2, 0, 0, 14, 14, 12, 11, 10, 10, 10, 10, 11, 10, 14, 0, 1, 15 },
		{1, 0, 15, 14, 13, 12, 11, 10, 10, 9, 9, 9, 9, 8, 15, 0, 0, 0, 0, 0 },
		{1, 2, 0, 0, 14, 14, 12, 11, 11, 11, 10, 10, 10, 9, 15, 1, 0, 0, 0 },
		{2, 2, 0, 1, 14, 15, 13, 12, 11, 10, 11, 10, 10, 9, 9, 0, 0, 0, 0, 0 },
		{2, 0, 0, 0, 15, 13, 12, 12, 11, 10, 10, 10, 10, 10, 9, 9, 9, 0, 1, 15, 1, 15, 0, 1},
		{4, 1, 1, 0, 15, 14, 13, 12, 11, 11, 10, 10, 10, 10, 9, 9, 0, 0, 0, 1, 0, 0, 0},		//// 30th +ve digit2 data
		{3, 3, 3, 1, 0, 0, 15, 14, 12, 11, 12, 10, 11, 10, 10, 10, 9, 10, 0, 0, 0, 0, 0},
		{4, 2, 0, 0, 0, 0, 13, 12, 11, 11, 11, 9, 11, 10, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 15},
		{2, 1, 0, 14, 13, 12, 11, 10, 9, 9, 8, 15, 0, 15, 0, 0},
		{2, 2, 1, 0, 14, 13, 12, 11, 10, 10, 9, 9, 8, 15, 0, 0, 0, 0, 1 },
		{2, 1, 1, 15, 13, 12, 11, 10, 9, 9, 15, 15, 0, 0, 1},		
		{2, 0, 15, 12, 12, 11, 11, 10, 10, 10, 15, 0, 0, 15, 0},
		{2, 1, 15, 15, 14, 13, 11, 10, 10, 10, 10, 9, 11, 0, 0, 0, 15, 15, 0, 15},
		{2, 0, 0, 15, 15, 13, 11, 11, 10, 10, 10, 9, 9, 0, 15, 15, 0, 0},
		{3, 3, 1, 15, 0, 15, 13, 11, 10, 10, 10, 10, 10, 14, 0, 15, 0, 1},
		{2, 2, 1, 15, 13, 12, 11, 10, 10, 10, 9, 15, 0, 15, 0, 0, 1, 15 },		
		{2, 1, 0, 13, 12, 11, 10, 10, 10, 10, 10, 15, 0, 0, 0, 0 },
		{1, 1, 13, 12, 11, 11, 10, 10, 10, 10, 1, 15, 15, 0, 15 },
		{2, 1, 15, 13, 11, 10, 11, 10, 10, 10, 0, 0, 15, 15, 0},
		{2, 1, 0, 13, 12, 11, 10, 10, 10, 11, 11, 14, 0, 0, 0, 0, 0},		//// 44th +ve digit2 data
		{2, 2, 1, 0, 15, 13, 12, 12, 11, 11, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0},
		{1, 15, 12, 11, 11, 10, 10, 10, 10, 12, 0, 15, 0, 0, 15},
		{1, 14, 12, 11, 10, 10, 11, 10, 14, 0, 0, 15 },
		{2, 1, 13, 12, 11, 10, 10, 10, 10, 0, 0, 14, 1, 0},		
		{0, 0, 13, 11, 11, 11, 10, 9, 10, 12, 0, 0, 15, 1},
		{ 2, 1, 15, 12, 12, 11, 10, 11, 10, 14, 15, 1, 0 },
		{2, 1, 14, 13, 11, 11, 10, 11, 9, 10, 10, 0, 0, 0, 0, 0, 0},
		{1, 0, 15, 12, 12, 10, 10, 10, 10, 10, 13, 15, 0, 1, 1},
		{1, 15, 12, 11, 10, 10, 10, 14, 15, 15, 1, 0 },
		{2, 1, 14, 12, 12, 11, 11, 11, 10, 10, 10, 15, 0, 1, 0, 0 },		//// 54th +ve digit2 data
		{3, 1, 0, 14, 12, 12, 11, 11, 10, 10, 9, 10, 15, 0, 0, 0, 0 },
		{ 2, 15, 15, 12, 11, 10, 10, 10, 10, 10, 15, 15, 0, 0},
		{2, 0, 15, 12, 12, 11, 10, 10, 10, 10, 0, 15, 0, 0 },
		{2, 0, 15, 12, 12, 10, 10, 10, 11, 15, 0, 15},
		{1, 0, 13, 11, 10, 10, 10, 10, 10, 15, 0, 0, 0, 0},
		{2, 0, 15, 13, 11, 10, 10, 10, 10, 15, 0, 0, 0, 0 },
		{1, 1, 15, 12, 11, 11, 10, 10, 10, 15, 0, 0, 0, 0},
		{2, 1, 0, 13, 12, 12, 11, 10, 11, 10, 10, 15, 15, 0, 0},
		{1, 0, 12, 11, 10, 10, 10, 10, 0, 0, 0, 0, 15 },
		{3, 1, 0, 13, 11, 11, 10, 10, 10, 9, 10, 15, 0, 0, 0},
		{1, 14, 12, 11, 10, 10, 10, 10, 14, 0, 0, 0 },
		{2, 0, 0, 14, 12, 12, 11, 10, 10, 9, 9, 9, 0, 0, 0, 0, 0},		//66nd data end
		 
		 

		{2, 2, 2, 1, 13, 12, 12, 12, 12, 12, 12, 12, 11, 12 },			//-ve data 1
		{3, 2, 3, 2, 2, 13, 12, 12, 12, 12, 12, 11, 11, 11 },
		{3, 2, 3, 3, 2, 13, 12, 12, 12, 12, 12, 12, 12, 11 },
		{3, 2, 3, 13, 12, 12, 12, 12, 12, 11},
		{2, 2, 3, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{2, 2, 2, 1, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11},
		{2, 2, 2, 0, 12, 12, 12, 11, 11, 12, 12, 12, 11 },
		{3, 1, 2, 13, 12, 12, 12, 12, 11},
		{2, 3, 12, 12, 12, 11, 11, 12 },
		{2, 2, 13, 12, 12, 12, 12, 11, 12, 12 },
		{2, 2, 13, 12, 12, 12, 12, 11, 12, 12 },
		{2, 2, 13, 12, 12, 11, 12, 11 },
		{2, 2, 2, 2, 12, 12, 12, 12, 11, 12, 11},
		{3, 2, 2, 12, 12, 12, 12, 12, 11, 12, 11 },
		{2, 2, 2, 1, 12, 12, 12, 12, 12, 11, 12, 11},
		{2, 2, 2, 2, 1, 12, 12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12},
		{ 3, 3, 2, 2, 12, 12, 12, 12, 12, 11, 12 },
		{3, 2, 2, 12, 12, 11, 12, 12, 12, 12, 11 },
		{3, 2, 3, 2, 12, 12, 12, 11, 12, 12, 12, 12},
		{2, 2, 2, 12, 11, 11, 12, 12, 12, 12},
		{2, 2, 2, 13, 12, 11, 11, 12, 12, 12, 12, 12 },			
		{2,2,2,11,12,12,12,12,12,12,11,12,11},	
		{2, 2, 2, 12, 12, 12, 11, 12, 11, 10},
		{0, 2, 2, 2, 12, 12, 12, 11, 12, 12, 11, 12, 12, 10 },
		{2, 3, 2, 2, 13, 12, 12, 11, 12, 11, 12, 10 },
		{2, 3, 2, 2, 12, 12, 12, 12, 11, 12, 12, 11, 12},
		{2, 3, 2, 2, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11},
		{2, 2, 2, 2, 2, 12, 11, 12, 11, 11, 12, 11, 12, 11 },
		{0, 2, 2, 2, 2, 12, 12, 11, 11, 12, 12, 12, 12, 12, 11, 11},
		{2, 2, 2, 1, 2, 12, 11, 12, 12, 11, 12, 12, 11, 11, 10 },
		{ 2, 2, 2, 2, 12, 12, 11, 12, 12, 12, 12, 11, 11 },
		{2, 2, 2, 3, 2, 12, 11, 12, 12, 11, 11, 11, 12, 10 },		//32 th data end..
		{2, 1, 2, 12, 12, 12, 12, 12, 11, 11, 12},
		{0, 3, 0, 14, 12, 11, 12, 12, 12, 11, 11, 11 },
		{1, 2, 2, 13, 12, 12, 11, 12, 11, 12, 11, 12 },
		{2, 2, 2, 2, 13, 12, 12, 12, 12, 12, 11, 12, 11, 11},
		{2, 1, 1, 2, 13, 11, 11, 12, 11, 11, 11, 12, 11},
		{2, 0, 2, 0, 12, 12, 12, 12, 12, 11, 12, 12, 11, 12},
		{3, 0, 2, 2, 3, 13, 12, 11, 12, 12, 12, 11, 12, 12, 10 },
		{0, 2, 1, 1, 11, 11, 12, 12, 12, 11, 11, 11 },
		{ 2, 1, 2, 2, 3, 12, 12, 12, 12, 11, 11, 11, 12},
		{2, 1, 2, 13, 12, 11, 12, 11, 11, 12, 11, 12, 11},		
		{3, 2, 2, 2, 12, 12, 11, 11, 12, 12, 12, 12, 11},
		{3, 2, 2, 2, 2, 12, 12, 12, 12, 12, 11, 12, 11, 11, 12, 11, 12, 12},		//44nd data end
		{2, 2, 12, 11, 12, 12, 12, 12, 12, 12, 12, 11},
		{2, 2, 0, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11 },
		{2, 2, 12, 11, 12, 11, 12, 12, 12, 11, 11},
		{2, 2, 2, 13, 12, 11, 11, 11, 12, 12, 12, 12, 12 },		//48nd data end
		{2, 2, 0, 11, 11, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11},	
		{1, 2, 13, 12, 12, 12, 11, 12, 12, 11, 12},
		{2, 2, 2, 12, 12, 11, 12, 11, 12, 11, 12, 12, 12, 12},
		{2, 2, 13, 11, 12, 12, 12, 12, 12, 11, 12},
		{2, 2, 13, 12, 12, 11, 13, 10, 11, 12, 13},
		{2, 2, 3, 12, 12, 11, 11, 12, 12, 11, 12, 12, 12, 12, 11},	//54nd data end
		

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit2, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit2);

}


void GestureRecognizer::Digit3TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit3][50] = {	
		
		{1, 0, 14, 13, 12, 11, 8, 10, 6, 7, 1, 14, 14, 12, 11, 9, 8, 9, 7, 6, 7},
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
		{15, 12, 11, 10, 12, 15, 13, 12, 12, 10, 9, 7, 8},		//74th +ve 3 data end

		  
	 

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit3, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit3);

}


void GestureRecognizer::Digit4TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit4][50] = {	

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit4, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit4);

}

void GestureRecognizer::Digit5TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit5][50] = {	

		//data for digit 5 : not added yet
		{9, 8, 8, 8, 13, 11, 12, 11, 11, 1, 1, 15, 14, 13, 12, 12, 10, 9, 8, 8, 7, 6, 5},
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
		{7, 8, 8, 10, 11, 11, 0, 1, 0, 15, 13, 12, 12, 12, 11, 10, 9, 8, 8, 5, 4},		
		{7, 7, 8, 7, 8, 12, 12, 11, 12, 12, 0, 0, 0, 14, 13, 12, 12, 11, 10, 10, 8, 7, 7, 4 },
		{8, 8, 8, 7, 12, 12, 12, 12, 12, 12, 1, 0, 15, 13, 13, 12, 11, 10, 10, 8, 7, 7, 6},		//63th data end
		{8,8,11,12,12,0,0,15,14,12,11,9,8,7,5},
		{8,8,12,12,12,0,0,14,12,12,11,10,8,7},
		{7,8,12,12,1,0,14,13,12,11,10,8,7,4},		//66th data end
		//{1, 0, 14, 13, 12, 11, 8, 10, 6, 7, 1, 14, 14, 12, 11, 9, 8, 9, 7, 6, 7},			//Digit3 data as -ve data
		//{2, 0, 15, 14, 12, 12, 11, 10, 9, 8, 1, 15, 12, 11, 9, 8, 7, 7 },
		//{1, 0, 0, 14, 12, 11, 12, 10, 9, 8, 0, 15, 13, 12, 11, 9, 8, 9, 7, 7},
		//{0, 15, 14, 13, 12, 11, 10, 9, 8, 8, 1, 15, 14, 12, 12, 10, 9, 8, 7, 7, 8},
		//{0, 0, 15, 14, 12, 12, 11, 11, 8, 9, 6, 0, 0, 14, 12, 12, 9, 9, 8, 7, 7},
		//{0, 0, 14, 13, 13, 11, 10, 9, 10, 7, 0, 14, 12, 11, 10, 9, 8, 8, 7},
		//{0, 0, 14, 13, 13, 12, 11, 10, 8, 8, 7, 0, 15, 13, 0, 12, 12, 10, 10, 8, 8, 8, 6},
		//{1, 0, 15, 14, 13, 12, 11, 10, 10, 7, 8, 7, 0, 0, 14, 13, 12, 10, 10, 8, 8, 8, 7},
		//{0, 0, 13, 13, 12, 11, 11, 10, 8, 8, 1, 0, 14, 12, 12, 12, 10, 9, 8, 7, 7 },
		//{0, 0, 15, 13, 13, 11, 11, 9, 8, 1, 14, 13, 12, 10, 9, 8, 8, 7 },		
		//{2, 15, 14, 13, 12, 11, 10, 9, 8, 5, 1, 0, 15, 14, 12, 11, 10, 9, 8, 8 },
		//{1, 14, 14, 13, 12, 11, 10, 9, 8, 0, 14, 0, 13, 10, 10, 9, 8, 7, 8},
		//{15, 0, 15, 13, 12, 11, 10, 9, 8, 7, 1, 0, 0, 14, 12, 11, 10, 11, 9, 8, 8, 7, 8},
		//{1, 1, 0, 15, 13, 12, 11, 11, 10, 9, 8, 1, 15, 12, 11, 10, 9, 9, 7, 8 },
		//{0, 0, 0, 15, 14, 12, 12, 11, 10, 8, 8, 8, 5, 15, 0, 0, 13, 12, 11, 11, 10, 11, 7, 7, 7},
		//{0, 0, 14, 13, 12, 11, 11, 9, 9, 8, 1, 15, 14, 12, 11, 9, 9, 8, 8, 7, 7 },
		//{0, 4, 15, 0, 0, 15, 13, 13, 12, 11, 11, 9, 9, 7, 1, 15, 14, 12, 12, 11, 9, 8, 8, 8, 8, 7},
		//{15, 1, 0, 15, 13, 12, 11, 11, 10, 9, 7, 0, 0, 14, 12, 11, 12, 10, 9, 8, 8, 7, 7},
		//{1, 1, 1, 15, 15, 13, 12, 12, 11, 11, 10, 11, 8, 7, 7, 1, 14, 14, 12, 11, 11, 9, 9, 8, 8, 7, 6},
		//{1, 0, 15, 15, 13, 11, 10, 10, 8, 0, 15, 14, 12, 11, 10, 9, 8, 7, 7},		//20th +ve 3 data end
		//{15, 14, 12, 11, 10, 9, 8, 1, 15, 13, 12, 10, 11, 10, 7, 8, 7, 9, 7, 5},
		//{0, 14, 13, 12, 10, 10, 8, 15, 14, 12, 10, 9, 7, 8},
		//{0, 0, 14, 13, 11, 11, 10, 9, 9, 3, 15, 14, 13, 11, 10, 10, 7, 8, 7, 7 },
		//{1, 0, 13, 12, 12, 10, 9, 0, 14, 12, 12, 10, 10, 8, 8, 7, 7 },
		//{2, 0, 15, 14, 13, 12, 11, 11, 9, 8, 0, 14, 12, 12, 10, 10, 8, 7, 7, 8, 6},
		//{0, 15, 13, 12, 11, 9, 7, 15, 14, 13, 12, 10, 10, 10, 8, 7},
		//{0, 0, 14, 12, 11, 10, 8, 8, 0, 15, 15, 12, 12, 10, 10, 9, 8, 7, 5 },
		//{0, 0, 14, 12, 10, 9, 9, 15, 15, 13, 11, 10, 10, 8, 6, 7},
		//{ 0, 15, 14, 12, 11, 10, 9, 8, 15, 15, 14, 12, 12, 12, 10, 10, 8, 8, 6 },
		//{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4},
		//{1, 15, 15, 14, 13, 11, 10, 10, 8, 14, 14, 13, 12, 10, 9, 9, 8, 7, 4 },
		//{ 0, 15, 15, 13, 12, 10, 8, 15, 15, 13, 12, 10, 10, 8, 8, 8, 6 },
		//{0, 15, 14, 13, 11, 10, 8, 8, 15, 14, 14, 12, 12, 10, 9, 8, 7, 7},
		//{0, 0, 15, 13, 12, 11, 10, 9, 7, 0, 14, 14, 13, 12, 10, 8, 9, 8, 7, 7},
		//{0, 0, 14, 12, 10, 9, 8, 14, 14, 13, 12, 10, 9, 8, 7, 5},
		//{0, 15, 15, 12, 11, 10, 9, 14, 14, 13, 11, 10, 8, 7, 7},
		//{0, 15, 14, 12, 10, 9, 9, 15, 14, 12, 12, 12, 9, 9, 7},
		//{1, 0, 14, 13, 12, 10, 9, 0, 14, 12, 10, 10, 8, 8, 8, 8, 6, 5, 4 },
		//{0, 14, 14, 12, 12, 10, 8, 15, 14, 12, 10, 9, 9, 7, 5, 6, 4},
		//{0, 14, 14, 12, 12, 10, 8, 0, 14, 12, 9, 8, 8, 7, 8},
		//{0, 15, 13, 11, 9, 15, 15, 14, 12, 11, 10, 10, 8, 7, 7 },		//41th +ve 3 data end
		//{15, 13, 11, 9, 9, 8, 0, 15, 14, 14, 13, 10, 9, 8, 8, 7, 4 },
		//{0, 15, 13, 12, 11, 9, 9, 8, 0, 15, 13, 12, 11, 9, 9, 8},
		//{0, 15, 14, 13, 12, 11, 10, 9, 8, 0, 14, 15, 12, 11, 11, 10, 9, 8, 8},
		//{0, 15, 14, 13, 12, 11, 9, 8, 8, 15, 15, 14, 12, 11, 10, 9, 8, 7 },
		//{15, 14, 15, 14, 13, 12, 10, 8, 8, 0, 15, 13, 13, 11, 11, 10, 8, 8},
		//{15, 14, 13, 13, 11, 9, 8, 7, 15, 0, 15, 14, 13, 11, 11, 10, 8, 7, 8, 5 },
		//{0, 0, 15, 14, 13, 12, 11, 10, 8, 8, 7, 1, 15, 14, 13, 12, 12, 11, 10, 8, 8, 8 },
		//{0, 15, 13, 12, 10, 10, 8, 15, 13, 12, 11, 11, 10, 9, 7},
		//{0, 15, 15, 14, 12, 11, 10, 9, 14, 15, 13, 12, 11, 10, 9, 8, 7, 7 },
		//{2, 0, 14, 13, 13, 12, 10, 8, 10, 5, 0, 15, 14, 13, 12, 10, 10, 9, 7, 7 },		//51th +ve 3 data end
		//{0, 15, 14, 12, 11, 10, 10, 8, 0, 15, 13, 12, 11, 10, 9, 7 },
		//{0, 0, 15, 0, 13, 12, 11, 11, 9, 8, 15, 15, 13, 12, 11, 11, 10, 9, 8, 7, 7, 5 },
		//{1, 0, 0, 13, 13, 12, 11, 10, 9, 14, 13, 12, 12, 11, 11, 10, 8, 7, 8, 6},			
		//{0, 0, 13, 11, 9, 15, 13, 11, 10, 8, 8},
		//{15, 15, 12, 11, 9, 9, 0, 15, 13, 11, 10, 9, 8},
		//{15, 13, 12, 10, 9, 15, 14, 12, 11, 10, 9, 8 },	
		//{0, 13, 12, 11, 10, 15, 12, 11, 11, 8, 7, 7, 5 },
		//{15, 15, 13, 12, 11, 11, 9, 0, 14, 12, 11, 11, 10, 8, 7, 6},
		//{1, 15, 0, 13, 12, 11, 10, 13, 13, 12, 11, 10, 8, 6, 5},
		//{0, 15, 12, 11, 11, 10, 14, 12, 12, 11, 10, 8, 7, 5},
		//{0, 15, 13, 11, 10, 13, 13, 12, 11, 12, 10, 9, 6, 5 },
		//{0, 15, 12, 11, 9, 15, 13, 12, 11, 10, 9, 7, 5, 4},
		//{1, 15, 13, 11, 10, 8, 15, 14, 12, 12, 10, 8, 7, 6},
		//{2, 1, 0, 15, 12, 12, 11, 10, 11, 14, 13, 11, 11, 9, 8, 7, 6 },
		//{2, 1, 0, 14, 12, 11, 11, 10, 15, 14, 12, 11, 10, 9, 7, 7, 6, 4},
		//{0, 15, 12, 11, 10, 14, 13, 13, 10, 8, 8, 7, 5},
		//{0, 15, 13, 12, 10, 9, 14, 13, 12, 10, 10, 9, 6, 6, 4},
		//{2, 1, 0, 15, 14, 12, 12, 11, 11, 11, 11, 9, 9, 14, 13, 12, 12, 11, 11, 10, 8, 8, 7 },
		//{1, 0, 0, 15, 15, 14, 12, 11, 10, 9, 13, 14, 13, 12, 12, 11, 11, 11, 10, 8, 7, 8},
		//{2, 0, 0, 0, 15, 12, 11, 11, 11, 10, 10, 9, 15, 15, 13, 12, 12, 11, 11, 10, 7, 8, 7, 6},
		//{0, 3, 2, 15, 15, 13, 12, 11, 11, 10, 9, 9, 15, 15, 14, 13, 12, 11, 11, 10, 8, 7, 7, 8, 4},
		//{2, 0, 0, 12, 13, 12, 12, 11, 10, 10, 9, 9, 15, 14, 13, 13, 11, 10, 8, 8, 7, 5},
		//{15, 12, 11, 10, 12, 15, 13, 12, 12, 10, 9, 7, 8},		//74th -ve 3 data end

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit5, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit5);

}

void GestureRecognizer::Digit6TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit6][50] = {	

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit6, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit6);

}

void GestureRecognizer::Digit7TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit7][50] = {	

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit7, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit7);

}

void GestureRecognizer::Digit8TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit8][50] = {	

		//data for digit 8 : not added yet
		{9, 10, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 5, 4, 2, 2, 1, 2, 4, 4, 6, 8, 6},
		{8, 10, 11, 13, 15, 15, 15, 14, 13, 11, 11, 10, 8, 9, 6, 5, 2, 1, 1, 2, 3, 4, 5, 4},
		{9, 8, 9, 10, 13, 15, 15, 14, 14, 12, 11, 10, 9, 7, 6, 5, 4, 2, 1, 1, 2, 2, 3, 4, 5},
		{8, 8, 9, 10, 12, 14, 15, 14, 13, 13, 11, 11, 9, 11, 8, 6, 5, 2, 2, 1, 2, 2, 4, 4, 4, 4, 3},
		{9, 10, 12, 15, 15, 14, 13, 12, 10, 9, 7, 6, 3, 2, 1, 1, 1, 2, 4, 5, 5},
		{8, 10, 12, 14, 15, 13, 13, 12, 10, 9, 8, 6, 4, 2, 1, 1, 2, 3, 3, 4, 4, 4 },
		{7, 9, 9, 10, 12, 15, 15, 15, 13, 13, 12, 10, 10, 8, 7, 5, 4, 2, 2, 0, 2, 2, 2, 4},
		{8, 9, 10, 13, 15, 14, 14, 12, 11, 10, 8, 7, 6, 4, 2, 1, 2, 2, 3},
		{8, 9, 11, 14, 15, 15, 13, 12, 10, 10, 9, 8, 6, 4, 2, 2, 1, 2, 3, 3},
		{10, 14, 15, 15, 13, 11, 9, 8, 6, 3, 1, 2, 4, 6},
		{8, 8, 10, 13, 15, 15, 14, 14, 12, 11, 10, 9, 7, 5, 4, 3, 2, 1, 2, 4},
		{8, 9, 9, 11, 12, 12, 14, 15, 15, 14, 14, 12, 11, 10, 9, 8, 8, 7, 4, 2, 2, 1, 2, 3, 3, 4, 4},
		{8, 9, 10, 11, 12, 0, 14, 14, 14, 13, 11, 10, 9, 10, 8, 7, 5, 4, 2, 2, 1, 2, 2, 3, 3, 6},
		{8, 9, 10, 12, 13, 14, 0, 14, 13, 13, 11, 10, 8, 7, 5, 4, 3, 2, 2, 2, 2, 5},
		{8, 10, 10, 13, 15, 13, 12, 10, 9, 9, 5, 3, 2, 2, 3, 3, 4, 4},
		{9, 10, 11, 11, 13, 14, 14, 15, 13, 12, 10, 9, 8, 7, 5, 3, 2, 2, 2, 2, 3, 3, 4},
		{9, 10, 10, 12, 13, 14, 14, 13, 12, 11, 10, 9, 7, 5, 4, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4},
		{8, 9, 11, 15, 15, 15, 13, 12, 11, 8, 8, 5, 3, 2, 1, 3, 3, 4},
		{8, 11, 11, 13, 15, 15, 13, 12, 11, 10, 8, 8, 6, 4, 2, 1, 2, 2, 5, 6},
		{10, 12, 15, 14, 14, 13, 11, 9, 7, 5, 4, 2, 1, 2, 4, 6},
		{9, 9, 11, 15, 14, 13, 12, 11, 9, 7, 5, 5, 2, 1, 1, 3, 3, 5, 7},
		{8, 8, 10, 13, 14, 0, 15, 15, 13, 13, 12, 11, 9, 9, 6, 3, 2, 2, 2, 5, 7},
		{10, 12, 13, 14, 13, 12, 11, 7, 4, 3, 2, 2, 1, 1, 3, 4, 7},
		{10, 10, 12, 15, 15, 14, 14, 12, 12, 10, 9, 8, 5, 3, 2, 2, 2, 2, 4, 5, 6},
		{8, 8, 10, 11, 14, 15, 0, 14, 12, 11, 10, 10, 8, 5, 5, 3, 1, 1, 1, 2, 4, 5, 5},
		{8, 8, 10, 12, 14, 0, 13, 14, 13, 12, 11, 10, 9, 6, 6, 4, 3, 2, 2, 2, 2, 4, 5},
		{9, 11, 14, 14, 13, 10, 8, 4, 1, 1, 2, 3, 4, 4},
		{7, 8, 11, 11, 14, 15, 14, 13, 11, 10, 8, 8, 5, 4, 2, 2, 2, 2, 2, 6, 9},
		{8, 10, 12, 14, 14, 13, 13, 12, 12, 9, 8, 7, 4, 2, 2, 2, 2, 2, 3, 4, 4},		
		{8, 11, 13, 14, 15, 14, 12, 11, 10, 9, 8, 5, 4, 2, 2, 2, 3, 3, 5},
		{9, 10, 14, 14, 13, 12, 11, 9, 11, 8, 5, 5, 3, 2, 1, 1, 3, 4, 4, 7},
		{8, 9, 10, 11, 13, 14, 13, 14, 12, 10, 10, 7, 5, 3, 2, 2, 2, 2, 2, 3, 5},
		{8, 9, 9, 12, 14, 14, 14, 13, 12, 12, 10, 9, 7, 5, 4, 3, 2, 1, 2, 3, 3, 4, 5},
		{6, 9, 9, 11, 12, 13, 14, 14, 12, 12, 11, 8, 7, 5, 3, 2, 2, 2, 2, 3, 3, 4, 4},
		{7, 9, 10, 12, 14, 13, 14, 14, 12, 10, 9, 8, 5, 4, 2, 2, 2, 2, 3, 4, 6},
		{8, 8, 9, 11, 10, 13, 14, 14, 14, 15, 12, 12, 10, 10, 9, 7, 7, 6, 4, 3, 2, 2, 2, 2, 3, 4, 4, 6},
		{8, 9, 11, 14, 14, 13, 14, 12, 11, 9, 8, 6, 5, 4, 2, 2, 1, 2, 3, 6, 7},
		{8, 8, 9, 11, 13, 15, 14, 14, 13, 13, 11, 10, 8, 7, 7, 5, 4, 2, 1, 2, 2, 2, 2, 6, 7},
		{8, 8, 9, 13, 15, 15, 14, 14, 14, 13, 10, 9, 7, 6, 4, 2, 2, 1, 2, 3, 4, 6, 4},
		{10, 14, 14, 15, 13, 10, 8, 6, 2, 1, 2, 3, 4, 6, 8},
		{10, 13, 15, 15, 15, 12, 10, 10, 8, 5, 3, 2, 2, 1, 2, 4, 8},		//41th data end

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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit8, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit8);

}

void GestureRecognizer::Digit9TrainingData(){
	   
	float trainingDataTotal[noOfTrainingSamplesDigit9][50] = {	

		//data for digit 9 : not added yet
		{8, 10, 12, 12, 14, 15, 0, 3, 3, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 11},
		{8, 9, 12, 12, 14, 15, 3, 4, 4, 4, 11, 12, 12, 12, 11, 12, 11, 12, 12},
		{8, 10, 12, 12, 12, 14, 0, 3, 4, 4, 4, 4, 12, 12, 12, 12, 12, 11, 12, 12, 11, 12},
		{8, 11, 12, 14, 0, 4, 4, 4, 4, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{8, 10, 12, 14, 0, 3, 4, 4, 12, 11, 12, 12, 12, 12},
		{9, 11, 11, 12, 12, 14, 15, 2, 3, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12},
		{9, 10, 11, 12, 12, 12, 14, 0, 2, 3, 4, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 12, 11, 12, 12, 4},
		{9, 10, 11, 13, 13, 2, 3, 3, 4, 4, 12, 12, 12, 12, 12, 12, 12, 11, 11, 12},
		{9, 11, 6, 12, 12, 13, 15, 3, 2, 3, 4, 4, 12, 12, 12, 11, 12, 12, 12, 11, 12 },
		{8, 10, 10, 12, 1, 15, 2, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 11, 12, 4, 12, 3, 11, 3},
		{10, 10, 12, 14, 15, 2, 3, 4, 4, 12, 11, 12, 12, 12, 12, 12, 12, 11, 12, 4},
		{8, 9, 12, 12, 14, 1, 2, 4, 4, 12, 12, 12, 12, 12, 12, 12},
		{8, 9, 9, 11, 13, 12, 14, 14, 0, 2, 3, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12, 11},
		{8, 10, 10, 13, 13, 15, 2, 3, 4, 4, 4, 12, 12, 12, 11, 11, 12, 12, 12, 11},
		{9, 10, 11, 12, 14, 15, 2, 2, 4, 4, 4, 11, 12, 11, 11, 12, 12, 12, 12, 11},
		{8, 9, 12, 12, 15, 1, 3, 4, 4, 11, 12, 12, 11, 11, 12, 12, 12},
		{7, 8, 11, 10, 12, 13, 13, 14, 0, 3, 3, 4, 4, 4, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{8, 9, 11, 12, 14, 15, 1, 3, 3, 4, 11, 12, 12, 11, 12, 12},
		{9, 10, 12, 13, 15, 0, 2, 3, 4, 4, 12, 12, 12, 12, 11, 12, 12},
		{8, 9, 12, 12, 12, 13, 0, 2, 3, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 12},
		{8, 10, 10, 12, 12, 13, 14, 15, 2, 4, 4, 4, 4, 11, 12, 12, 12, 11, 11, 12, 12, 11, 12},
		{9, 9, 11, 13, 0, 0, 3, 3, 4, 4, 11, 11, 12, 12, 11, 12, 11, 12, 12},
		{8, 8, 11, 12, 12, 13, 14, 0, 2, 3, 3, 4, 12, 12, 12, 11, 12, 12, 11, 12, 12},
		{8, 9, 11, 11, 12, 13, 15, 1, 4, 3, 4, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 11},
		{9, 11, 11, 13, 14, 1, 3, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12},
		{10, 10, 12, 14, 15, 2, 3, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 11, 12, 12},
		{9, 8, 11, 11, 13, 14, 0, 0, 3, 3, 4, 4, 12, 11, 12, 12, 13, 12, 12, 12, 12},
		{8, 8, 10, 11, 13, 12, 15, 0, 2, 3, 4, 4, 4, 12, 12, 12, 11, 12, 12, 12, 11, 12},
		{8, 9, 10, 12, 12, 13, 14, 0, 3, 3, 4, 4, 11, 12, 12, 12, 12, 12, 12, 12},
		{8, 9, 10, 11, 13, 14, 15, 1, 3, 4, 3, 4, 5, 12, 12, 12, 11, 12, 12, 12, 11, 12, 12},
		{9, 11, 11, 13, 14, 0, 2, 4, 4, 4, 12, 12, 12, 12, 12, 12},
		{8, 10, 12, 12, 15, 0, 1, 3, 4, 4, 11, 12, 11, 12, 11, 12, 12, 12},
		{8, 10, 6, 11, 11, 12, 13, 0, 1, 4, 3, 4, 11, 12, 12, 11, 12, 12, 11},
		{10, 10, 12, 14, 15, 1, 3, 4, 4, 4, 11, 12, 12, 11, 12, 11, 12, 12},
		{8, 10, 10, 13, 0, 1, 3, 3, 4, 12, 12, 12, 12, 11, 12, 12},
		{9, 10, 11, 12, 0, 13, 0, 2, 4, 3, 4, 4, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
		{8, 10, 10, 14, 12, 14, 2, 2, 4, 4, 11, 11, 12, 12, 11, 12, 12},
		{8, 9, 10, 11, 12, 14, 15, 2, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12},
		{8, 8, 9, 11, 11, 11, 12, 12, 15, 15, 15, 1, 2, 4, 4, 4, 4, 4, 3, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12, 11 },
		{8, 8, 10, 12, 11, 12, 13, 13, 14, 0, 1, 3, 3, 4, 4, 4, 4, 11, 12, 12, 12, 12, 12, 12, 12, 11, 12},
		{7, 8, 9, 10, 11, 12, 13, 13, 14, 1, 2, 3, 4, 4, 4, 4, 12, 12, 12, 11, 12, 11, 11, 11, 12, 12, 12, 12},
		{7, 8, 11, 8, 12, 12, 13, 13, 15, 1, 2, 3, 3, 4, 4, 4, 12, 12, 12, 12, 11, 12, 12, 12, 12, 11, 11},
		{7, 9, 9, 11, 12, 12, 13, 13, 0, 2, 2, 4, 4, 4, 4, 4, 4, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11, 12, 11, 12, 12},
		{9, 10, 12, 11, 14, 14, 2, 3, 4, 4, 4, 12, 12, 11, 12, 11, 12, 12, 12, 12, 11, 12},
		{8, 10, 11, 12, 14, 0, 2, 3, 3, 4, 11, 12, 12, 12, 11, 11, 12, 12},
		{9, 11, 10, 12, 12, 13, 15, 1, 2, 3, 3, 4, 4, 4, 12, 12, 12, 12, 12, 11, 11, 12, 12, 12},
		{8, 9, 12, 11, 12, 13, 14, 0, 2, 3, 3, 4, 4, 4, 4, 4, 12, 11, 12, 12, 12, 12, 12, 11, 12, 12, 11, 12, 11},		//47th data end


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
	Mat trainingDataMatTemp(noOfTrainingSamplesDigit9, 50, CV_32FC1, trainingDataTotal);
	trainingDataMatTemp.copyTo(trainingDataMatDigit9);

}
void GestureRecognizer::writeCircleAngleStreamInFile(vector<int>& returnReducedVector,int& indexOfGestures,String gestureText){
	
	if(indexOfGesturesGlobal != indexOfGestures){
		
		//ostringstream convert;   // stream used for the conversion
		//convert << indexOfGestures;
		gestureTextVec.push_back(gestureText);
		 Mat tempMat;
		 tempMat.create(1,returnReducedVector.size(),CV_8U);

		 for(int i = 0; i< returnReducedVector.size(); i++){
			tempMat.at<uchar>(0,i) = returnReducedVector[i];
		 
		 }
		 circleAngleVec.push_back(tempMat);
		// file<<"mat"+convert.str()<<tempMat ; 
		 indexOfGesturesGlobal = indexOfGestures;
	}  

	if(indexOfGestures == 10){
		writeCircleAngleStreamInFile(circleAngleVec,gestureTextVec);
			indexOfGestures = 0;
	}
}

void GestureRecognizer::writeCircleAngleStreamInFile(vector<Mat>& circleAngleVec,vector<String>& gestureTextVecSrc){
	cv::FileStorage file;
	file.open("some_name.text", cv::FileStorage::WRITE);

	if(gestureTextVecSrc.size() == circleAngleVec.size())
	for(int i = 0; i< circleAngleVec.size(); i++){
		ostringstream convert;   // stream used for the conversion
		convert << i; 
		file<<"mat"+convert.str()<<circleAngleVec[i] ; 
		file<<"gestureText"<<gestureTextVecSrc[i]; 
	}
	circleAngleVec.clear();
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
bool myfunction (int i, int j) {
  return (i==j);
}

//problem with this method is that it finds pattern only with same digit, ie  if there is pattern of 12 for 2 times 
//then it returns the 12 but if it has 12,11,12,11 then it does not understand it as pattern
void GestureRecognizer::findRepetedPatternsInASrcVec(vector<int>& srcVec,int& returnFirst,int& returnSecond){
	std::vector<int>::iterator it;

  // using default comparison:
  it = std::adjacent_find (srcVec.begin(), srcVec.end());

  if (it!=srcVec.end()){
	  returnFirst = *it;
    std::cout << "the first pair of repeated elements are: " << *it << '\n';
  }
  //using predicate comparison:
  it = adjacent_find (++it, srcVec.end(), myfunction);
	
  if (it!=srcVec.end()){
	  returnSecond = *it;
	  std::cout << "the second pair of repeated elements are: " << *it << '\n';
  }
}

//this method will search left pattern like : 1,0,15,15,1,0 
//param : srcVec: Angle vec chain
//param : return{attern: return str vec with direction
//param : limit: required threshold that needs to be matched in order to define direction..
void GestureRecognizer::findRepetedPatternsInASrcVecRight(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterRight = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 1-0-15 range?
		if(srcVec[i] == 1 || srcVec[i] == 0 || srcVec[i] == 15){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 1 || srcVec[i+1] == 0 || srcVec[i+1] == 15){
				counterRight++;
				if(counterRight >=limit){	
					returnPattern.push_back("Right");		
				}
			} else {	//if next element is not in the range, then reset counter..
				counterRight = 0;
			}
		}
	}
}

//this method will search left pattern like : 7,8,9,9,8,7 
//param : srcVec: Angle vec chain
//param : return{attern: return str vec with direction
//param : limit: required threshold that needs to be matched in order to define direction..
void GestureRecognizer::findRepetedPatternsInASrcVecLeft(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterLeft = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 7-8-9 range?
		if(srcVec[i] == 7 || srcVec[i] == 8 || srcVec[i] == 9){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 7 || srcVec[i+1] == 8 || srcVec[i+1] == 9){
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
void GestureRecognizer::findRepetedPatternsInASrcVecUp(vector<int>& srcVec,vector<String>& returnPattern,int limit){

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
void GestureRecognizer::findRepetedPatternsInASrcVecDown(vector<int>& srcVec,vector<String>& returnPattern,int limit){

	int counterDown = 0;
	for(int i = 0; i< srcVec.size()-1; i++){
		//check if element if in 11-12-13 range?
		if(srcVec[i] == 11 || srcVec[i] == 12 || srcVec[i] == 13){
			//check if the next element is also in the same range? if yes inc counter
			if(srcVec[i+1] == 11 || srcVec[i+1] == 12 || srcVec[i+1] == 13){
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


String GestureRecognizer::understandTheGestureWithShapeDescriptor(vector<Point3i>& srcVector){
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

//Para: srcVectorScattered is a 3D points with scattered trajectory. 
//		ie they are not that contineous they are atleast 5 pixels seperated from each other in x or y direction.
//Para: srcReducedVec is a vector of codes from scattered points, helps to train and test SVM
//Para: srcVectorOriginal is a 3D points with scattered trajectory helps to find shapes from shape descriptors.
String GestureRecognizer::understandTheGesture(vector<Point3i>& srcVectorScattered,vector<int>& srcReducedVec,
	vector<Point3i>& srcVectorOriginal,Mat& srcMatForShapes){
	
	//convert 3D vector to 2D vector for shape recognision..
	vector<Point> srcVector2D;

	for(int cont = 0; cont < srcVectorScattered.size(); cont++){
		srcVector2D.push_back(Point(srcVectorScattered[cont].x,srcVectorScattered[cont].y));
	}
	
	//convert 3D vector to 2D vector for shape recognision..
	/*vector<Point> srcVector2DOriginal;

	for(int cont = 0; cont < srcVectorOriginal.size(); cont++){
		srcVector2DOriginal.push_back(Point(srcVectorOriginal[cont].x,srcVectorOriginal[cont].y));
	}*/

	String returnString;	
	 
	bool flag = false;	//flag if any of the following gesture is detected.?

	//check out the width of gesture
	if(width < 10){
		
		flag = detectUp(srcVectorScattered,returnString);
		if(flag == false)
		flag = detectDown(srcVectorScattered,returnString);
	}

	  
	//check out the height of gesture
	if(height < 10){
		if(flag == false)
		flag = detectLeft(srcVectorScattered,returnString);
		if(flag == false)
		flag = detectRight(srcVectorScattered,returnString);
	
	}
 
	//Common Mat for all SVM classifiers
	Mat testMatForSVM;
	testMatForSVM = Mat::zeros(1,50,CV_32FC1);

	int sizeOfsrcReducedVec = srcReducedVec.size();
	if(sizeOfsrcReducedVec > 50){		//it blocks the no of column greater than 50, because in training data, we train our chain for max 50 elements.
		sizeOfsrcReducedVec = 50;
	}
	for(int reducedVecCnt = 0; reducedVecCnt < sizeOfsrcReducedVec ;reducedVecCnt++){
		testMatForSVM.at<float>(0,reducedVecCnt) = srcReducedVec[reducedVecCnt];
	}

	//Gesture 5(circle) & 6(Square) by SVM
	//Rule1: start point and stop points should be close enough if not same.!
	if( closenessCriteria < 20){
		//Gesture  (8-Eight): condition is very imp..  
		if(srcReducedVec[0] == 7 || srcReducedVec[0] == 8 || srcReducedVec[0] == 9 ||
			srcReducedVec[0] == 10 || srcReducedVec[0] == 11 ){	//condition 1: chain must start with 7,8,9,10,11 
			float response = SVMDigit8.predict(testMatForSVM);
			if (response == 1){
				returnString += "Digit 8, ";		//Gesture  (7-Seven) 
				flag = true;
			}
		}

		// The array for storing the approximation curve
		std::vector<cv::Point> approx;
		 
		//approxPolyDP(srcVector2D, approx, 10, true);
		double digit = arcLength(Mat(srcVector2D), true)*0.02;
		cout<<"digit : "<<digit<<endl;
		approxPolyDP(srcVector2D, approx, 6.5, true);
		for (int i = 0; i < approx.size(); i++) {
			Point point1;
			point1.x = approx[i].x*4;
			point1.y = approx[i].y*4;
			putText(srcMatForShapes,"angle",point1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,0,0),2,1);
			circle(srcMatForShapes,point1,1,Scalar(255,255,0),3,8);
		}

		 
		if (approx.size() == 3 && returnString.empty()){
			returnString += "Triangle";    // Triangles
			flag = true;
		} else if (approx.size() >= 4 && approx.size() <= 6 && returnString.empty()) {
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
			 
			if (((vtc == 4  && maxCosine < 0.3) || returnString.empty()) && detectSquareWithSimpleTechnique(srcReducedVec)) {
				// Detect rectangle or square

				if(abs(width - height) <= 10){
					returnString += "Square";    // Square
					flag = true;
				} else {
					returnString += "Rectangle";    // Rectangle
					flag = true;
				}
				//setLabel(dst, ratio <= 0.02 ? "SQU" : "RECT", contours[i]);
			}
			 
		} //else if (approx.size() >= 4 && approx.size() <= 6)

		float responseForCircle = SVMCircle.predict(testMatForSVM);
		if (responseForCircle == 1 && returnString.empty()){
			//Gesture 5(circle) & 6(Ellipse/Zero) by Manual Methods
			if(height <= width*1.4){
				 
				returnString += "circle";
			} else {
				returnString += "ellipse or Digit 0";		//ellipse or 0
			}
			
			flag = true;		//initialize falg == true for string output
		}

		 
	}	//if(sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 20)

	if(flag == false){
	  
			//Rule1: start point has to be on top-left of stop point.
			if(startPoint.x < stopPoint.x  && startPoint.y < stopPoint.y){					//common condition for Digit1 & 2
				if(stopPoint.x == extreamBottom.x && stopPoint.y == extreamBottom.y &&	//condition1 : for Digit1
					(srcReducedVec[srcReducedVec.size()-1] == 11 || srcReducedVec[srcReducedVec.size()-1] == 12 || srcReducedVec[srcReducedVec.size()-1] == 13) && //condition2 : chain must stop with 11,12,13
						(srcReducedVec[0] == 1 || srcReducedVec[0] == 2 || srcReducedVec[0] == 3)){	//condition 3: chain must start with 1,2,3 
						float response = SVMDigit1.predict(testMatForSVM);
						vector<String> returnStrVecDown,returnStrVecRight;
						findRepetedPatternsInASrcVecDown(srcReducedVec,returnStrVecDown,2);		//additional condition for Digit1
						 

						if(returnStrVecDown.size() > 0){
								if (response == 1 && returnStrVecDown[0] == "Down"){
									returnString += "Digit 1, ";		//Gesture  (1-One) 
									flag = true;
								}
						}
				} //condi for digit 1

				if((srcReducedVec[srcReducedVec.size()-1] == 0 || srcReducedVec[srcReducedVec.size()-1] == 1 || srcReducedVec[srcReducedVec.size()-1] == 15)) {//condition1 : chain must stop with 0,1,15
					float response = SVMDigit2.predict(testMatForSVM);
					vector<String> returnStrVec;
					findRepetedPatternsInASrcVecRight(srcReducedVec,returnStrVec,2);		//additional condition for Digit2
					if(returnStrVec.size() > 0){
						if (response == 1 && returnStrVec[0] == "Right"){
							returnString += "Digit 2, ";		//Gesture  (1-Two) 
							flag = true;
						} 
					}
					  
				} //condi for digit 2
			}	//commmon condition for 1 & 2
		 
			//Gesture  (3-Three): condition is very imp.. 
			if(startPoint.y < stopPoint.y &&  // condi 1 : y start must be smaller than y stop
				((startPoint.x == extreamLeft.x && startPoint.y == extreamLeft.y) ||  (stopPoint.x == extreamLeft.x && stopPoint.y == extreamLeft.y)) &&  // condition2 : start or stop must be extream left
				(srcReducedVec[0] == 0 || srcReducedVec[0] == 1 || srcReducedVec[0] == 2 || srcReducedVec[0] == 15)	&& //condition3 : chain must start with 0,1,2 or 15
				(srcReducedVec[srcReducedVec.size()-1] == 4 || srcReducedVec[srcReducedVec.size()-1] == 5 || srcReducedVec[srcReducedVec.size()-1] == 6 || srcReducedVec[srcReducedVec.size()-1] == 7 || 
				srcReducedVec[srcReducedVec.size()-1] == 8 || srcReducedVec[srcReducedVec.size()-1] == 8 || srcReducedVec[srcReducedVec.size()-1] == 9)	//condition4 : chain must stop with 4,5,6,7,8,9
				&& returnString.empty() ) {//condition5 : the string must be empty
					
				float response = SVMDigit3.predict(testMatForSVM);
				 
					if (response == 1){
						returnString += "Digit 3, ";		//Gesture  (3-Three) 
						flag = true;
					}
			}

			//Gesture  (4-Four): condition is very imp.. 
			//FiXED: in future add one more condition, that the start point and stop points are extream low and height values in a vector
			if(startPoint.y < stopPoint.y && startPoint.x < stopPoint.x){	//condition 1: start point must be top left of stop point
				if(stopPoint.x == extreamBottom.x && stopPoint.y == extreamBottom.y &&	//condition 2: stop point must be extream bottom point
					(srcReducedVec[0] == 10 || srcReducedVec[0] == 11 || srcReducedVec[0] == 12) ){	//condition 3: chain must start with 11,12
					float response = SVMDigit4.predict(testMatForSVM);
					vector<String> returnStrVec;
					findRepetedPatternsInASrcVecDown(srcReducedVec,returnStrVec,1);		//additional condition for Digit4
					if(returnStrVec.size() > 0){
						if (response == 1 && returnStrVec[0] == "Down"){
							returnString += "Digit 4, ";		//Gesture  (4-Four) 
							flag = true;
						}
					}
				}
			}

			//Gesture 5,S,6
			if(startPoint.y < stopPoint.y && startPoint.x > stopPoint.x ){
			//Gesture  (5-Five): condition is very imp.. It is similar to Letter S
				if((srcReducedVec[0] == 7 || srcReducedVec[0] == 8 || srcReducedVec[0] == 9) &&		//chain must start with 7,8,9
					(srcReducedVec[srcReducedVec.size()-1] == 4 || srcReducedVec[srcReducedVec.size()-1] == 5 || srcReducedVec[srcReducedVec.size()-1] == 6
					|| srcReducedVec[srcReducedVec.size()-1] == 7 ||srcReducedVec[srcReducedVec.size()-1] == 8)) {//chain must stop with 4,5,6,7 or 8
					float response = SVMDigit5.predict(testMatForSVM);
					vector<String> returnStrVecLeft,returnStrVecDown;
					findRepetedPatternsInASrcVecLeft(srcReducedVec,returnStrVecLeft,1);		//additional condition for Digit5
					findRepetedPatternsInASrcVecDown(srcReducedVec,returnStrVecDown,1);		//additional condition for Digit5
						if(returnStrVecLeft.size() > 0 && returnStrVecDown.size() > 0){
							if (response == 1 && returnStrVecLeft[0] == "Left" && returnStrVecDown[0] == "Down" && returnString != "Digit 3, "){
								returnString += "Digit 5, ";		//Gesture  (5-Five) 
								flag = true;
							}
						} 
				}
				//Gesture  (S-Letter) 
				//Rule1: start point has to be on top-left of stop point.
				if(stopPoint.x == extreamLeft.x && stopPoint.y == extreamLeft.y &&	//end point must be extream left point.
					(srcReducedVec[0] == 6 || srcReducedVec[0] == 7 || srcReducedVec[0] == 8 || srcReducedVec[0] == 9 || srcReducedVec[0] == 10)){	//chain must start with 6,7,8,9 or 10
						
					float response = SVMLetterS.predict(testMatForSVM);
					 
					if (response == 1 && returnString != "Digit 5, "){
						returnString += "Letter S, ";
						flag = true;
					}
				} 
				
				if((srcReducedVec[0] == 7 || srcReducedVec[0] == 8 || srcReducedVec[0] == 9 || srcReducedVec[0] == 10) && //chain must start with 7,8,9 or 10		
					(srcReducedVec[srcReducedVec.size()-1] == 8 || srcReducedVec[srcReducedVec.size()-1] == 9 || srcReducedVec[srcReducedVec.size()-1] == 10 
					|| srcReducedVec[srcReducedVec.size()-1] == 11 ||srcReducedVec[srcReducedVec.size()-1] == 12) ){	//chain must stop with 8,9,10,11 or 12		
					//Gesture  (6-Six): condition is very imp.. 
					float response = SVMDigit6.predict(testMatForSVM);
					if (response == 1 && returnString != "Letter S, "){
						returnString += "Digit 6, ";		//Gesture  (6-Six) 
						flag = true;
					}
				}
					
			}
		  
			//Gesture  (7-Seven): condition is very imp.. 
			if((srcReducedVec[0] == 0 || srcReducedVec[0] == 1 || srcReducedVec[0] == 15) && 	//chain must start with 0,1, or 15
				(srcReducedVec[srcReducedVec.size()-1] == 9 || srcReducedVec[srcReducedVec.size()-1] == 10 || srcReducedVec[srcReducedVec.size()-1] == 11)) {//chain must stop with 9,10,11
				float response = SVMDigit7.predict(testMatForSVM);
				vector<String> returnStrVec;
				findRepetedPatternsInASrcVecRight(srcReducedVec,returnStrVec,1);
				if(returnStrVec.size() > 0){
					if (response == 1 && returnStrVec[0] == "Right" && returnString != "Digit 4, " && returnString != "Digit 1, "
						&& returnString != "Digit 2, "){
						returnString += "Digit 7, ";		//Gesture  (7-Seven) 
						flag = true;
					}
				}
			}

			//Gesture  (9-Nine): condition is very imp.. 
			 
			if(startPoint.y < stopPoint.y){	//condition 1: start point must have smaller y than stop point
				if(stopPoint.x == extreamBottom.x && stopPoint.y == extreamBottom.y && 		//condition 2: stoppoint must be extream bottom point
					(srcReducedVec[0] == 7 || srcReducedVec[0] == 8 || srcReducedVec[0] == 9 || srcReducedVec[0] == 10) && 	//condition 3: chain must start with 7,8,9,10 
						(srcReducedVec[srcReducedVec.size()-1] == 11 || srcReducedVec[srcReducedVec.size()-1] == 12 || srcReducedVec[srcReducedVec.size()-1] == 13)) {//chain must stop with 11,12,13
					float response = SVMDigit9.predict(testMatForSVM);
					if (response == 1 && returnString != "Digit 4, "){
						returnString += "Digit 9, ";		//Gesture  (9-Nine) 
						flag = true;
					}
				}
			}
			
 
	}

	//flag = detectLater5(srcReducedVec,returnString);
	 
	if(flag == true){
		return returnString;
	}	

	 
	return "not configured!";
}
 
//this manual detection works very well.  
bool GestureRecognizer::detectSquareWithSimpleTechnique(vector<int>& srcVector){

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
double GestureRecognizer::findAngleBetweenTwoPoints(Point3i& A,Point3i& B){
// This function calculates the angle of the line from A to B with respect to the positive X-axis in degrees
 
 
	//condition 1: Quarter 1..
	 if(B.x > A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			return acos (param) * 180.0 / PI;
		}
		return 45;	//if denominator is 0 then assum first quarter angle 45'
	} 
	 

	//condition 2: Quarter 2..
	if(B.x < A.x && B.y < A.y){

		//cos(theta) = abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			return 180 - theta;
		}
		return 135;	//if denominator is 0 then assum second quarter angle 135'
	}
	 
	//condition 3: Quarter 3..
	if(B.x < A.x && B.y > A.y){

		//cos(theta) = 180 + abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			return 180 + theta;
		}
		return 225;	//if denominator is 0 then assum third quarter angle 225'
	}

	//condition 4: Quarter 4..
	if(B.x > A.x && B.y > A.y){

		//cos(theta) = 360 - abs(B.x - A.x)/ sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0)) 
		double denominator = sqrt (pow ((B.x - A.x),2.0) + pow ((B.y - A.y),2.0));  
		if(denominator != 0){
			double diff = abs(B.x - A.x);
			double param = diff/denominator;
			double theta = acos (param) * 180.0 / PI;
			return 360 - theta;
		}
		return 315;	//if denominator is 0 then assum fourth quarter angle 315'
	}

	//condition 5: 0 position..
	if(B.y == A.y){
		if(B.x > A.x)
		return 0;	//if both y are same and B.x > A.x then it should be 0'
	}
	//condition 6: 180 position..
	if(B.y == A.y){
		if(B.x < A.x)
		return 180;	//if both y are same and B.x < A.x then it should be 180'
	}
	//condition 7: 90 position..
	if(B.x == A.x){
		if(B.y < A.y)
		return 90;	//if both x are same and B.y < A.y then it should be 90'
	}
	//condition 8: 270 position..
	if(B.x == A.x){
		if(B.y > A.y)
		return 270;	//if both x are same and B.y > A.y then it should be 270'
	}
}

//Param:	srcVec is input vec with all codes
int GestureRecognizer::findNoOfStreightSidesInVector(vector<int>& srcVec){

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





bool GestureRecognizer::detectEllipse(vector<int>& returnReducedVector,String& returnString){
	//gesture 6: ellipse
	//Rule1: start point and stop points should be close enough if not same.!
	if( sqrt (pow ((startPoint.x - stopPoint.x),2.0) + pow ((startPoint.y - stopPoint.y),2.0)) < 10){
		//Rule 2: Analyze angle vector smartly..
	 
		vector<Mat> rulesForEllipse;
		
		Mat firstElementRule,secondElementRule,thirdElementRule,fourthElementRule,fifthElementRule,sixthElementRule,sevenththElementRule,eigthElementRule,ninthElementRule;
		
		firstElementRule.create(1,1,CV_8U);secondElementRule.create(2,1,CV_8U);thirdElementRule.create(3,1,CV_8U);
		fourthElementRule.create(4,1,CV_8U);fifthElementRule.create(5,1,CV_8U);sixthElementRule.create(4,1,CV_8U);
		sevenththElementRule.create(3,1,CV_8U);eigthElementRule.create(2,1,CV_8U);ninthElementRule.create(1,1,CV_8U);
		
		firstElementRule.at<uchar>(0,0) = 1;	 
		secondElementRule.at<uchar>(0,0) = 0;	secondElementRule.at<uchar>(1,0) = 7;	  
		thirdElementRule.at<uchar>(0,0) = 5;	thirdElementRule.at<uchar>(1,0) = 6;	thirdElementRule.at<uchar>(2,0) = 7; 
		fourthElementRule.at<uchar>(0,0) = 6;	fourthElementRule.at<uchar>(1,0) = 5;	fourthElementRule.at<uchar>(2,0) = 4;	fourthElementRule.at<uchar>(3,0) = 3; 
		fifthElementRule.at<uchar>(0,0) = 5;	fifthElementRule.at<uchar>(1,0) = 4;	fifthElementRule.at<uchar>(2,0) = 3;	fifthElementRule.at<uchar>(3,0) = 2;	fifthElementRule.at<uchar>(4,0) = 1;
		sixthElementRule.at<uchar>(0,0) = 4;	sixthElementRule.at<uchar>(1,0) = 3;	sixthElementRule.at<uchar>(2,0) = 2;	sixthElementRule.at<uchar>(3,0) = 1;
		sevenththElementRule.at<uchar>(0,0) = 3;sevenththElementRule.at<uchar>(1,0) = 2;sevenththElementRule.at<uchar>(2,0) = 1; 
		eigthElementRule.at<uchar>(0,0) = 2;	eigthElementRule.at<uchar>(1,0) = 1;	
		ninthElementRule.at<uchar>(0,0) = 1;	

		rulesForEllipse.push_back(firstElementRule);
		rulesForEllipse.push_back(secondElementRule);
		rulesForEllipse.push_back(thirdElementRule);
		rulesForEllipse.push_back(fourthElementRule);
		rulesForEllipse.push_back(fifthElementRule);
		rulesForEllipse.push_back(sixthElementRule);
		rulesForEllipse.push_back(sevenththElementRule);
		rulesForEllipse.push_back(eigthElementRule);
		rulesForEllipse.push_back(ninthElementRule);


		int counter = 0;
		int sizeOfReducedVec =  returnReducedVector.size();
		int sizeOfRuleVec =  rulesForEllipse.size();

		for(int indx = 0; indx <  sizeOfReducedVec; indx++){
			if(indx < sizeOfRuleVec){
				for(int indxRule = 0; indxRule < rulesForEllipse[indx].rows ; indxRule++){
					if(returnReducedVector[indx] == rulesForEllipse[indx].at<uchar>(indxRule,0)){
						counter++;
							//return "circle";
					}
				}
			}
		} 
		
		if(counter >= rulesForEllipse.size()*0.9){
			returnString = "Ellipse";
			return true;
		}
	}//Rule1 condition end..
	return false;
}

bool GestureRecognizer::detectCharacterS(vector<int>& returnReducedVector,String& returnString){
	//gesture 6: Character S
	//Rule1: start point has to be on top-left of stop point.
	if(startPoint.x > stopPoint.x  && startPoint.y < stopPoint.y){
		//Rule 2: Analyze angle vector smartly..
		vector<Mat> rulesForS;
		
		Mat firstElementRule,secondElementRule,thirdElementRule,fourthElementRule,
			fifthElementRule,sixthElementRule,sevenththElementRule,eigthElementRule,ninthElementRule;
		
		firstElementRule.create(3,1,CV_8U);secondElementRule.create(3,1,CV_8U);thirdElementRule.create(5,1,CV_8U);
		fourthElementRule.create(4,1,CV_8U);fifthElementRule.create(5,1,CV_8U);/*sixthElementRule.create(4,1,CV_8U);*/
		/*sevenththElementRule.create(3,1,CV_8U);eigthElementRule.create(2,1,CV_8U);ninthElementRule.create(1,1,CV_8U);*/
		
		firstElementRule.at<uchar>(0,0) = 3; firstElementRule.at<uchar>(1,0) = 4;	firstElementRule.at<uchar>(2,0) = 5;	 
		secondElementRule.at<uchar>(0,0) = 5;	secondElementRule.at<uchar>(1,0) = 6;	   secondElementRule.at<uchar>(2,0) = 7;
		thirdElementRule.at<uchar>(0,0) = 7;	thirdElementRule.at<uchar>(1,0) = 0;	thirdElementRule.at<uchar>(2,0) = 1;	thirdElementRule.at<uchar>(3,0) = 5;	thirdElementRule.at<uchar>(4,0) = 6; 
		fourthElementRule.at<uchar>(0,0) = 6;	fourthElementRule.at<uchar>(1,0) = 5;	fourthElementRule.at<uchar>(2,0) = 4;	fourthElementRule.at<uchar>(3,0) = 3;
		fifthElementRule.at<uchar>(0,0) = 3;	fifthElementRule.at<uchar>(1,0) = 4;	fifthElementRule.at<uchar>(2,0) = 5;
		
		/*sixthElementRule.at<uchar>(0,0) = 4;	sixthElementRule.at<uchar>(1,0) = 3;	sixthElementRule.at<uchar>(2,0) = 2;	sixthElementRule.at<uchar>(3,0) = 1;
		sevenththElementRule.at<uchar>(0,0) = 3;sevenththElementRule.at<uchar>(1,0) = 2;sevenththElementRule.at<uchar>(2,0) = 1; 
		eigthElementRule.at<uchar>(0,0) = 2;	eigthElementRule.at<uchar>(1,0) = 1;	
		ninthElementRule.at<uchar>(0,0) = 1;	*/

		rulesForS.push_back(firstElementRule);
		rulesForS.push_back(secondElementRule);
		rulesForS.push_back(thirdElementRule);
		rulesForS.push_back(fourthElementRule);
		rulesForS.push_back(fifthElementRule);
		/*rulesForS.push_back(sixthElementRule);
		rulesForS.push_back(sevenththElementRule);
		rulesForS.push_back(eigthElementRule);
		rulesForS.push_back(ninthElementRule);*/


		int counter = 0;
		int sizeOfReducedVec =  returnReducedVector.size();
		int sizeOfRuleVec =  rulesForS.size();

		for(int indx = 0; indx <  sizeOfReducedVec; indx++){
			if(indx < sizeOfRuleVec){
				for(int indxRule = 0; indxRule < rulesForS[indx].rows ; indxRule++){
					if(returnReducedVector[indx] == rulesForS[indx].at<uchar>(indxRule,0)){
						counter++;
							 
					}
				}
			}
		} 
		
		if(counter >= returnReducedVector.size()*0.8){
			returnString = "Charactor S";
			return true;
		}

	}
		
	return false;
}

bool GestureRecognizer::detectLater5(vector<int>& returnReducedVector,String& returnString){
	//gesture 6: Character S
	//Rule1: start point has to be on top-left of stop point.
	if(startPoint.x > stopPoint.x  && startPoint.y < stopPoint.y){
		//Rule 2: Analyze angle vector smartly..
		vector<Mat> rulesForS;
		
		Mat firstElementRule,secondElementRule,thirdElementRule,fourthElementRule,fifthElementRule,sixthElementRule,sevenththElementRule,eigthElementRule,ninthElementRule;
		
		firstElementRule.create(2,1,CV_8U);secondElementRule.create(2,1,CV_8U);thirdElementRule.create(3,1,CV_8U);
		fourthElementRule.create(4,1,CV_8U);fifthElementRule.create(5,1,CV_8U);sixthElementRule.create(4,1,CV_8U);
		sevenththElementRule.create(3,1,CV_8U);eigthElementRule.create(2,1,CV_8U);ninthElementRule.create(1,1,CV_8U);
		
		firstElementRule.at<uchar>(0,0) = 4;	firstElementRule.at<uchar>(0,0) = 5;	 
		secondElementRule.at<uchar>(0,0) = 0;	secondElementRule.at<uchar>(1,0) = 7;	  
		thirdElementRule.at<uchar>(0,0) = 5;	thirdElementRule.at<uchar>(1,0) = 6;	thirdElementRule.at<uchar>(2,0) = 7; 
		fourthElementRule.at<uchar>(0,0) = 6;	fourthElementRule.at<uchar>(1,0) = 5;	fourthElementRule.at<uchar>(2,0) = 4;	fourthElementRule.at<uchar>(3,0) = 3; 
		fifthElementRule.at<uchar>(0,0) = 5;	fifthElementRule.at<uchar>(1,0) = 4;	fifthElementRule.at<uchar>(2,0) = 3;	fifthElementRule.at<uchar>(3,0) = 2;	fifthElementRule.at<uchar>(4,0) = 1;
		sixthElementRule.at<uchar>(0,0) = 4;	sixthElementRule.at<uchar>(1,0) = 3;	sixthElementRule.at<uchar>(2,0) = 2;	sixthElementRule.at<uchar>(3,0) = 1;
		sevenththElementRule.at<uchar>(0,0) = 3;sevenththElementRule.at<uchar>(1,0) = 2;sevenththElementRule.at<uchar>(2,0) = 1; 
		eigthElementRule.at<uchar>(0,0) = 2;	eigthElementRule.at<uchar>(1,0) = 1;	
		ninthElementRule.at<uchar>(0,0) = 1;	

		rulesForS.push_back(firstElementRule);
		rulesForS.push_back(secondElementRule);
		rulesForS.push_back(thirdElementRule);
		rulesForS.push_back(fourthElementRule);
		rulesForS.push_back(fifthElementRule);
		rulesForS.push_back(sixthElementRule);
		rulesForS.push_back(sevenththElementRule);
		rulesForS.push_back(eigthElementRule);
		rulesForS.push_back(ninthElementRule);


		int counter = 0;
		int sizeOfReducedVec =  returnReducedVector.size();
		int sizeOfRuleVec =  rulesForS.size();

		for(int indx = 0; indx <  sizeOfReducedVec; indx++){
			if(indx <= sizeOfRuleVec){
				for(int indxRule = 0; indxRule < rulesForS[indx].rows ; indxRule++){
					if(returnReducedVector[indx] == rulesForS[indx].at<uchar>(indxRule,0)){
						counter++;
							 
					}
				}
			}
		} 
		
		if(counter >= returnReducedVector.size()*0.8){
			returnString = "Later 5";
			return true;
		}

	}
		
	return false;
}