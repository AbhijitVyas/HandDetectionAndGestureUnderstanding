#include "MainMethodClass.h"
//constructor
 MainMethodClass::MainMethodClass(KalmanFilter& kfLeft,KalmanFilter& kfRight){
	selectObject = false;
	trackObjectLeft = 0;
	trackObjectRight = 0;
	showHist = true;
	firstTimeFlagForGlobalDepthMedianDistance = true;
	flagForGesture = false;						//this flag is used for gesture start stop indication..

	flagForDirectionLeft = "top";
	flagForDirectionRight = "top";
	firstTimeDirectionFlag = true;
	flagForFirstTime	= true;		//this flag is for cuurent depth frame and previous depth frame and operation  
	
	//set prevTrajectoryPoint to initial condition
	prevTrajectoryPoint.x = 0;
	prevTrajectoryPoint.y = 0;
	prevTrajectoryPoint.z = 0;

	countForStartStopGesture = 0;	//this is the counter which will indicate start stop after certain value.
	dynamicHueMin1 = 2;
	dynamicHueMax1 = 12;			// keeps lower range of Hue to red/ skin region, stays away from yellow(hue = 12 to 20)
	dynamicHueMin2 = 160;
	dynamicHueMax2 = 170;

	Y_MIN  = 0;
	Y_MAX  = 255;
	Cr_MIN = 145;   //148 is for stable noise free and 142 for more skin pixels in dark environment
	Cr_MAX = 170;
	Cb_MIN = 75;
	Cb_MAX = 255;

	//global hue min & max ranges for initial filtering
	staticHueMin1 = 2;
	staticHueMax1 = 12;
	staticHueMin2 = 160;
	staticHueMax2 = 170;

	satMin = 65;
	satMax = 256;
	valMin = 20;		// valmin = 80, reduces noise in kinect also stays away from black color detection..
	// but valmin = 20 works well in dark and not bright situations..
	valMax = 256;
 
	medianBlurCnt = 0;
	erodeAmt = 0;
	dilateAmt = 0;
	backGroundSubtractionFlag = 0;
	flagForImage = 0;
	flagForKalman = 0;
	 
	flickering = 0;
	flagForWaterShade = 0;
	displayHistogramFlag = 0;
	handDetectionFlag = 0;

	gesture = 0;
	globalCounter = 0;
	 
	MINNOOFPIXELS = 200;
	hsize = 16;
	histimg = Mat::zeros(200, 320, CV_8UC3);

	counterForLeftTWNull = 0;		//this is counter for no of frames the left TW is null, if its more than certain value reinitialize the left TW..
	counterForRightTWNull = 0;		//this is counter for no of frames the Right TW is null, if its more than certain value reinitialize the Right TW..

	findFingersFlag = 0;

	dynamicThresholding = 0;		//flag for dynamic thresholding
	
	widthHeightRationFactor = 1.6;	//this is very imp for creating new Rect from width or height info.. currently the height will be 1.6 times the width

	kalman = Kalman(kfLeft,kfRight);
	this->kfLeft = kfLeft;  
	this->kfRight = kfRight;
	
	trackWindowFromDepthLeft = Rect(0,0,0,0);
	trackWindowFromDepthRight= Rect(0,0,0,0);
	depthDistanceLeft = 0;
	
	indexOfGestures = 0;

	//initialize global variables..
	ORGCOLS = 640;
	ORGROWS = 480;
	PYRDOWNCOLS = 160;
	PYRDOWNROWS = 120;
	depthIplBackGrndSubMask = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS), IPL_DEPTH_8U,1);
	edgeIplDepthImage = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS), IPL_DEPTH_16U,1);
	depthIplImagePyrDown = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS), IPL_DEPTH_16U,1);
	tempImageLeft = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS), IPL_DEPTH_8U,1);
	tempImageRight = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS), IPL_DEPTH_8U,1);
	tempImageEdgeImage = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS),IPL_DEPTH_8U,1);

	edgeMaskImageLeft = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS),IPL_DEPTH_8U,1);
	edgeMaskImageRight = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS),IPL_DEPTH_8U,1);

	prevdeflickerIplImage = cvCreateImage(cvSize(PYRDOWNCOLS,PYRDOWNROWS),IPL_DEPTH_16U,1);
	cvSet(prevdeflickerIplImage, cvScalar(0)); 

	mask1PreviousFrame = Mat::zeros(Size(PYRDOWNCOLS,PYRDOWNROWS),CV_8U);
}

 MainMethodClass::~MainMethodClass(){
	cvReleaseImage(&depthIplBackGrndSubMask);
	cvReleaseImage(&depthIplImagePyrDown);
	cvReleaseImage(&depthIplImageOriginalSize); 
	cvReleaseImage(&edgeIplDepthImage);
	cvReleaseImage(&tempImageLeft);
	cvReleaseImage(&tempImageRight);
	cvReleaseImage(&tempImageEdgeImage);
	cvReleaseImage(&edgeMaskImageLeft);
	cvReleaseImage(&edgeMaskImageRight);
	cvReleaseImage(&prevdeflickerIplImage);
 }
void MainMethodClass::onMouseStatic( int event, int x, int y, int flags, void* that)
{
	MainMethodClass *self = static_cast<MainMethodClass*>(that);
    self->onMouse( event,  x, y, flags );
}
void MainMethodClass::onMouse( int event, int x, int y, int flags )
{
	//cout<<" MOUSE INITIALIZED.................................................."<<endl;
	 
		  if( selectObject )
			{
		
				selection.x = MIN(x, origin.x);
				selection.y = MIN(y, origin.y);
				selection.width = std::abs(x - origin.x);
				selection.height = std::abs(y - origin.y);

				selection &= Rect(0, 0, 560, 480);
				
				//try to initialize with mouse click
				trackWindowFromDepthLeft = selection;
				trackWindowFromDepthRight = selection;
			 
				
			}

			switch( event )
			{
			case CV_EVENT_LBUTTONDOWN:
				origin = Point(x,y);
				selection = Rect(x,y,0,0);
				selectObject = true;
				selectionLeft = selection;
			 
				//try to initialize with mouse click
				trackWindowFromDepthLeft = selection;
				
				depthDistanceLeft = 0;  //very important. if you miss depth somewhere, try to reinitialize it.

				break;
			case CV_EVENT_LBUTTONUP:
				selectObject = false;
				if( selection.width > 0 && selection.height > 0 ){
					trackObjectLeft = -1;
					trackObjectRight = -1;
					selectionLeft = selection;
					//try to initialize with mouse click
					trackWindowFromDepthLeft = selection;
					depthDistanceLeft = 0;		//very important. if you miss depth somewhere, try to reinitialize it.
				}
				break;

			case CV_EVENT_RBUTTONDOWN:
				origin = Point(x,y);
				selection = Rect(x,y,0,0);
				selectObject = true;
				selectionRight = selection;
				//try to initialize with mouse click
				trackWindowFromDepthRight = selection;
				 depthDistanceRight = 0;		//very important. if you miss depth somewhere, try to reinitialize it.
				break;
			case CV_EVENT_RBUTTONUP:
				selectObject = false;
				if( selection.width > 0 && selection.height > 0 ){
					trackObjectLeft = -1;
					trackObjectRight = -1;
					selectionRight = selection;
					//try to initialize with mouse click
					trackWindowFromDepthRight = selection;
					depthDistanceRight = 0;		//very important. if you miss depth somewhere, try to reinitialize it.
				}
				break;
			}

	  
	
			 
}
void MainMethodClass::createTrackbars(){
	 
    namedWindow( "CamShift Tracker", 0 );
	namedWindow( "Trackbars", 0 );
	
    setMouseCallback( "CamShift Tracker", MainMethodClass::onMouseStatic, this );
	setMouseCallback( "Trackbars", MainMethodClass::onMouseStatic, this );
	 
	cv::createTrackbar( "G_HMin1", "Trackbars", &staticHueMin1, 256, 0);
	cv::createTrackbar( "G_HMax1", "Trackbars", &staticHueMax1, 256, 0 );
	cv::createTrackbar( "G_HMin2", "Trackbars", &staticHueMin2, 256, 0 );
	cv::createTrackbar( "G_HMax2", "Trackbars", &staticHueMax2, 256, 0 );
  
    cv::createTrackbar( "Smin", "Trackbars", &satMin, 256, 0 );
	cv::createTrackbar( "Smax", "Trackbars", &satMax, 256, 0 );
	cv::createTrackbar( "Vmax", "Trackbars", &valMax, 256, 0 );
	cv::createTrackbar( "Vmin", "Trackbars", &valMin, 256, 0 );
	
	cv::createTrackbar( "Fingers", "Trackbars", &findFingersFlag, 1, 0 );
	cv::createTrackbar( "Dynamic-Thr", "Trackbars", &dynamicThresholding, 1, 0 );		//flag for dynamic thresholding
	
	createTrackbar("gesture","Trackbars",&gesture,1);
	createTrackbar("flickering","Trackbars",&flickering,1);
}
 
void MainMethodClass::pyrDownManually(IplImage* srcMat,IplImage* returnedIplImage){
	  
	int height = srcMat->height/4;
	int width = srcMat->width/4;
	 for(int y=0,y1 = 0; y<height; y++,y1++){
		for(int x=0,x1=0; x<width; x++,x1++){
	
			((unsigned short*)(returnedIplImage->imageData +
				returnedIplImage->widthStep*y1))[x1] = ((unsigned short*)(srcMat->imageData + srcMat->widthStep*(y*4)))[x*4];
			 

		}
	}

	 
 } 
 
//THis method loops through depth image and searches for the pixels values greater than 2000(depth), if condition satisfies then it sets those pixls to 0.
// At the same time it also searches for the edge pixls with neighbourhood difference of specified depthThreshold.
//It also tries to minimize the flickering noise in depth data.
void MainMethodClass::removeBackGroundBasedOnDepthAndCreateEdgeImage(IplImage* srcMat,int depth,IplImage* depthBackGrndSubMask,IplImage* edgeDepthImage){
	int depthThreshold = 50;  // very imp.. make sure this threshold matches the Region Growing threshold for edge growing..
	 
	int height = srcMat->height - 1;
	int width = srcMat->width - 1;
	cvSet(tempImageEdgeImage, cvScalar(0)); 
	unsigned short value,preValue;
	for(int y=1; y<height; y++){
		for(int x=1; x<width; x++){
			 value  = ((unsigned short *)&(srcMat->imageData[srcMat->widthStep * y]))[x];

			 if(value != 0){
				if( value >  depth) {
					((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] = 0;
				}
				else{
					/*preValue = ((unsigned short *)&(prevdeflickerIplImage->imageData[prevdeflickerIplImage->widthStep * y]))[x];
					value = abs(value + preValue)/2;*/
					((uchar*)(depthBackGrndSubMask->imageData +
									depthBackGrndSubMask->widthStep*y))[x] = 255;

					if( abs(value -  ((unsigned short*)(srcMat->imageData +
									srcMat->widthStep*(y-1)))[x-1]) > depthThreshold) {
					//the value is edge pixel..
					((uchar*)(tempImageEdgeImage->imageData +
									tempImageEdgeImage->widthStep*y))[x] = 255;

					((unsigned short*)(edgeDepthImage->imageData +
									edgeDepthImage->widthStep*(y)))[x] = value;
					
					//edgePoints.push_back(Point(x,y));
					continue;
					} else if(( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*(y-1)))[x]) > depthThreshold)) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*(y-1)))[x+1]) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*y))[x-1]) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*y))[x+1]) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*(y+1)))[x-1]) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*(y+1)))[x]) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					} else if( abs(value -  ((unsigned short*)(srcMat->imageData +
										srcMat->widthStep*(y+1)))[x]+1) > depthThreshold) {
						//the value is edge pixel..
						((uchar*)(tempImageEdgeImage->imageData +
										tempImageEdgeImage->widthStep*y))[x] = 255;
						((unsigned short*)(edgeDepthImage->imageData +
										edgeDepthImage->widthStep*(y)))[x] = value;
						//edgePoints.push_back(Point(x,y));
						continue;
				
					}
				}

				}// if value != 0

		} 
	}

	Mat edeMat;
	cvarrToMat(tempImageEdgeImage).copyTo(edeMat);
	erode(edeMat,edeMat,1);
	dilate(edeMat,edeMat,1);
	imshow("edeMat",edeMat);
}

void MainMethodClass::copyIplImageToOtherIplImage(IplImage* srcImage,IplImage* destImage){
	int height = srcImage->height;
	int width = srcImage->width;
	//cvSet(tempImageEdgeImage, cvScalar(0)); 
	unsigned short value;
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			 value  = ((unsigned short *)&(srcImage->imageData[srcImage->widthStep * y]))[x];
			 ((unsigned short*)(destImage->imageData +
										destImage->widthStep*(y)))[x] = value;
		}
	}
}
 
Mat MainMethodClass::deflicker(Mat Mat1,int strengthcutoff){  //deflicker - compares each pixel of the frame to a previously stored frame, and throttle small changes in pixels (flicker)

    if (prevdeflicker.rows){//check if we stored a previous frame of this name.//if not, theres nothing we can do. clone and exit
         int i,j;
        unsigned short* p;
        unsigned short* prevp;
        for( i = 0; i < Mat1.rows; ++i)
        {
            p = Mat1.ptr<unsigned short>(i);
            prevp = prevdeflicker.ptr<unsigned short>(i);
            for ( j = 0; j < Mat1.cols; ++j){  

                Scalar previntensity = prevp[j];
                Scalar intensity = p[j];
                int strength = abs(intensity.val[0] - previntensity.val[0]);

                if(strength < strengthcutoff){ //the strength of the stimulus must be greater than a certain point, else we do not want to allow the change
                    //value 25 works good for medium+ light.  anything higher creates too much blur around moving objects. 
                    //in low light however this makes it worse, since low light seems to increase contrasts in flicker - some flickers go from 0 to 255 and back.  :(
                    //I need to write a way to track large group movements vs small pixels, and only filter out the small pixel stuff.  maybe blur first?

                    if(intensity.val[0] > previntensity.val[0]){  // use the previous frames value.  Change it by +1 - slow enough to not be noticable flicker
                        p[j] = previntensity.val[0] + 1;  
                    }else{
                        p[j] = previntensity.val[0] - 1;
                    }
                }

            }

        }//end for
    }

    prevdeflicker = Mat1.clone();//clone the current one as the old one.
    return Mat1;
}


void MainMethodClass::writeMatToFile(Mat& srcMat,String filename,String MatName){

	cv::FileStorage file;
	file.open(filename+".text", cv::FileStorage::WRITE);

	 
		 
		file<<MatName<<srcMat; 
		  
	file.release();
}
 
void MainMethodClass::initializeVariables(cv::Mat& colorImage,cv::Mat& depthImage){
	colorImage.copyTo(colorMatOriginalSize);
	//depthImage.copyTo(depthMatOriginalSize);
	 
	//pyr down color image..
	pyrDown(colorMatOriginalSize,colorMatPyrDown);
	pyrDown(colorMatPyrDown,colorMatPyrDown);

	IplImage copy;
	copy = depthImage;
	depthIplImageOriginalSize = &copy;

	//pyrDown IplDepthImage manually, because if we use opencv method it adds noise in it..
	cvSet(depthIplImagePyrDown, cvScalar(0));
	pyrDownManually(depthIplImageOriginalSize,depthIplImagePyrDown);
	depthMatPyrDown = cvarrToMat(depthIplImagePyrDown);

	//find depth edge image and remove backGround..
	cvSet(edgeIplDepthImage,Scalar(0));
	cvSet(depthIplBackGrndSubMask,Scalar(0));

	removeBackGroundBasedOnDepthAndCreateEdgeImage(depthIplImagePyrDown, 2000,depthIplBackGrndSubMask,edgeIplDepthImage);
	//multiply with color so it masks the background
	Mat depthMatBackGrndSubMask = cv::cvarrToMat(depthIplBackGrndSubMask);	 
	cvtColor(depthMatBackGrndSubMask,depthMatBackGrndSubMask,CV_GRAY2BGR);
	colorMatPyrDown = colorMatPyrDown & depthMatBackGrndSubMask;


	const float scaleFactor = 0.05f;
	depthMatPyrDown.convertTo( show, CV_8U, scaleFactor );
	show.copyTo(depthImageForHandFingers);
}

void MainMethodClass::skinThresholding(){
	cvtColor(colorMatPyrDown, hsv, CV_BGR2HSV);
			//cvtColor(image, YCrCbMat, CV_BGR2YCrCb);
			inRange(hsv, Scalar(staticHueMin1, satMin, valMin),
				Scalar(staticHueMax1, satMax,valMax), mask1);
			/*inRange(hsv, Scalar(staticHueMin2, satMin, valMin),
				Scalar(staticHueMax2, satMax,valMax), mask2);
			inRange(YCrCbMat,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),
				cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),mask5);*/
			// this is dynamic thresholding but its not working nicely with kinect,
			// with webcame it works well
			//pyrUp(mask5,mask5);
			//imshow("YCrCb Mask",mask5);
			if(dynamicThresholding == 1){
				inRange(hsv, Scalar(dynamicHueMin1,satMin, valMin),
					Scalar(dynamicHueMax1, satMax,valMax), mask3);
				inRange(hsv, Scalar(dynamicHueMin2, satMin, valMin),
					Scalar(dynamicHueMax2, satMax,valMax), mask4);
				
				mask1 = mask1 |  mask3 | mask4 /*mask2 | mask5 */;
			}
	
			dilate(mask1,mask1,Mat(),Point(-1,-1),1); 
			erode(mask1,mask1,Mat(),Point(-1,-1),1);
			//medianBlur(backprojThreshold,backprojThreshold,7);
			//blur( backprojThreshold, backprojThreshold, Size( 3, 3 ), Point(-1,-1) );
			GaussianBlur( mask1, mask1, Size( 3, 3 ), 0, 0 );
				 
			threshold(mask1,mask1,30,255,THRESH_BINARY);
			
			int ch[] = {0, 0};
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);		
}
 
void MainMethodClass::initializeTrackingWindowLeft(){
	if(selectionLeft.area() == 0 || counterForLeftTWNull >= 50){		//if TW is null or counter is greater than 50
			 //initialize it for the first time, then assign this rect to selection left and right..
			 Rect initialTrackWindowLeft = Rect(20,50,20,30);
			  
			 Mat initialMatLeftMask1,initialMatRightMask1;
			 mask1(initialTrackWindowLeft).copyTo(initialMatLeftMask1);
		 
			 //find countours and its size...
			 vector<vector<Point>> coutoursLeft;
			 findContours(initialMatLeftMask1,coutoursLeft,0,1);
			  
			 int iTest = findBiggestContour(coutoursLeft);
				//	 cout<<"iTest : findContours : "<<iTest<<endl;
				Point center1; //very very imp
				double cArea = 0;
				Rect rect;
				if(iTest != -1){
					vector<Point> contourTest = coutoursLeft[iTest];
					  rect = boundingRect( Mat(contourTest) );
				}

				if(abs(rect.width - initialTrackWindowLeft.width) < 5 && abs(rect.height - initialTrackWindowLeft.height) < 5){
					selectionLeft = initialTrackWindowLeft;
					depthDistanceLeft = 0;
				 }

			 Rect drawLeft = Rect(initialTrackWindowLeft.tl().x*4,initialTrackWindowLeft.tl().y*4,initialTrackWindowLeft.width*4,initialTrackWindowLeft.height*4);
			 rectangle(colorMatOriginalSize,drawLeft,Scalar(0,0,255),2);
		 }

}

void MainMethodClass::initializeTrackingWindowRight(){
	if(selectionRight.area() == 0 || counterForRightTWNull >= 50){		//if TW is null or counter is greater than 50
			 //TODO: initialize it for the first time, then assign this rect to selection left and right..
			 Rect initialtrackWindowRight = Rect(120,50,20,30);

			 Mat initialMatRightMask1;
			 mask1(initialtrackWindowRight).copyTo(initialMatRightMask1);
		 
			 //find countours and its size...
			 vector<vector<Point>> coutoursRight;
			 findContours(initialMatRightMask1,coutoursRight,0,1);

			 int iTest = findBiggestContour(coutoursRight);
				//	 cout<<"iTest : findContours : "<<iTest<<endl;
				Point center1; //very very imp
				double cArea = 0;
				Rect rect;
				if(iTest != -1){
					vector<Point> contourTest = coutoursRight[iTest];
					 //cArea = contourArea(Mat(contourTest));
					 rect = boundingRect( Mat(contourTest) );
				}

				if(abs(rect.width - initialtrackWindowRight.width) < 5 && abs(rect.height - initialtrackWindowRight.height) < 5){
					selectionRight = initialtrackWindowRight;
					depthDistanceRight = 0;
				}

			 Rect drawRight = Rect(initialtrackWindowRight.tl().x*4,initialtrackWindowRight.tl().y*4,initialtrackWindowRight.width*4,initialtrackWindowRight.height*4);
			 rectangle(colorMatOriginalSize,drawRight,Scalar(255,0,0),2);
		 }

}
void MainMethodClass::run(cv::Mat& colorImage,cv::Mat& depthImage,OpenCVKinect& openCVKinect,int frameRate,
		Point3i& leftHandCoordinates,Point3i& rightHandCoordinates){
		//initialize color and depth Mat, IplImages 
		initializeVariables(colorImage,depthImage);
		skinThresholding();
		 
		initializeTrackingWindowLeft();
		initializeTrackingWindowRight();
		//do operation for left hand...
		if(selectionLeft.height > 0 && selectionLeft.width > 0 && 
			selectionLeft.tl().x > 0 && selectionLeft.tl().y > 0){
			trackLeftHand(leftHandCoordinates);
		} 
		if(selectionRight.height > 0 && selectionRight.width > 0 && 
			selectionRight.tl().x > 0 && selectionRight.tl().y > 0){
			trackRightHand(rightHandCoordinates);
		} 
  
		stringstream text;
		text<< frameRate;
		 
		putText(colorMatOriginalSize,text.str(),Point(460,460),
			CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,2,Scalar(0,255,0),3,3);
		imshow( "CamShift Tracker", colorMatPyrDown );

		pyrUp(show,show);
		imshow("Depth Image",show);
		imshow("Color Image",colorMatOriginalSize);
        
		mask1.copyTo(mask1PreviousFrame);
} 
void MainMethodClass::trackLeftHand(Point3i& handCoordinates){

	calculateBackProjForLeft();

	bool distanceLeftOutOfRangeFlag = false; 
	distanceLeftOutOfRangeFlag = createDepthMaskBeforeCamShiftLeft();	 
	
	//do manual threshold with cuurent meadian depth because this does not work well..
	if(trackWindowLeft.height > 0 && trackWindowLeft.width > 0 &&
		trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0) {		
		trackBox = CamShift(backprojLeft, trackWindowLeft,
		TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));		
	}												
	
	// if the trackBox has area greater than 50, then do some more operations...
	if(trackWindowLeft.height > 0 && trackWindowLeft.width > 0 &&
		trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0) {	
		 
		if(distanceLeftOutOfRangeFlag == true){
			counterForLeftTWNull++;		//increament counter if TW is out of range.. for reinitialization
		} else {	 
			//if TW is predicted correctly by camshift then reinitialize counter to 0
			counterForLeftTWNull = 0;		//reinitialize counter to 0 for camshift != null
		}
		//works like a charm...
		vector<Point> returnedPoints;
		 
		//this method takes depth image and depth edge image and centre point, returns two points XL,XH or YL,YH as a vector of point
		playWIthShowLeft(edgeIplDepthImage,depthIplImagePyrDown,returnedPoints,centerPointLeft,show);
		//grow region..
		if(returnedPoints.size() == 2){
			//findEdgeDepthPixelWithMeadianDepthRange(edgeDepthImage,returnedPoints);
			//initialize Region Growth object
			//start point can be XL or YL
			Point startPoint = returnedPoints[1];
			//stop point can be XH or YH
			Point stopPoint = returnedPoints[0];
			if(startPoint.x != 0 && startPoint.y != 0 && stopPoint.x != 0 && stopPoint.y != 0 ){
				 
				//initialRG = clock();
				GrowRegion rg(edgeIplDepthImage);
				vector<Point> borderPointsRGreturned;
				 
				rg.run(edgeIplDepthImage,startPoint,100,stopPoint,borderPointsRGreturned,returnedBorderMatLeft);		
				//finalRG =  clock() - initialRG; 
				//cout<<"Final RG Left : " <<(double)finalRG<<endl;
				// sort border points according to trackwindow size.
				if(borderPointsRGreturned.size() != 0){
					vector<Point> sortedPoints;
					//this method will sort the correct point 
					//on x axis either side and return the final points which includes the final close loop
					String flag;
					sortVectorPointsLeft(borderPointsRGreturned,sortedPoints,startPoint,stopPoint,flag,show);
					if(trackWindowFromDepthLeft.width > 0 && trackWindowFromDepthLeft.height > 0 && 
						trackWindowFromDepthLeft.tl().x > 0 && trackWindowFromDepthLeft.tl().y > 0){
						trackWindowLeft = trackWindowFromDepthLeft;
					}
					    
						Point3i point;
						float x,y,z;
						//openCVKinect.calculateWorldCoOrdinates(centerPointLeft.x*4,centerPointLeft.y*4,depthDistanceLeft,x,y,z);
						point.x = centerPointLeft.x;
						point.y = centerPointLeft.y;
						point.z = depthDistanceLeft;

						//Assign current hand coordinate value to return handCoordinates
						handCoordinates = Point3i(point.x*4,point.y*4,point.z);

						//if(flagForGesture == false){
							trajectory50Points.push_back(point);
						//}
						if(trajectory50Points.size() >= 30 && flagForGesture == false){
							flagForGesture = gestureRecognizerModified.checkForGesture(trajectory50Points);
							trajectory50Points.clear();
						}
						  
						if(flagForGesture || gesture == 1){
							trajectory.push_back(point);
							putText(colorMatOriginalSize,"Gesture: Start",Point(150,350),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(0,0,255),2,1);
							
							for (int i = 0; i < trajectory.size()-1; i++) 
							line(colorMatOriginalSize, Point(trajectory[i].x*4,trajectory[i].y*4),Point(trajectory[i+1].x*4,trajectory[i+1].y*4), 
								Scalar(0,0,255), 3, 8, 0);

							// check if the gesture stoped?
							if(trajectory50Points.size() >= 30){
								bool tempFlag = false;
								tempFlag = gestureRecognizerModified.checkForGesture(trajectory50Points);
								
								if(tempFlag){
									flagForGesture = false;
								}
								trajectory50Points.clear();
							}  
						} else {
							if(trajectory.size() != 0){
								putText(colorMatOriginalSize,"Gesture: Stop",Point(150,350),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(0,0,255),2,1);
								//for each point.. copy it to trajectory..
								individualTrajectory.clear();
								originalTrajectory.clear();
						 
								Point3i point;
								point = trajectory[0];
								for(int cn = 0; cn < trajectory.size() ; cn++){
									originalTrajectory.push_back(trajectory[cn]);		//this vec will held original points for shape descriptors
									if(abs(point.x - trajectory[cn].x) >= 5 || abs(point.y - trajectory[cn].y) >= 5){
										individualTrajectory.push_back(trajectory[cn]);
										point = trajectory[cn];
								 
									}
								}
								indexOfGestures++;
								//individualTrajectory = trajectory;
								trajectory.clear();
						 
							}
					   
							//Adrian's suggestion.
							if(originalTrajectory.size() != 0 && individualTrajectory.size() != 0){
								putText(colorMatOriginalSize,"Gesture: Stop",Point(150,350),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(0,0,255),2,1);
								understandGesture(originalTrajectory,individualTrajectory);
					 
								//clear this vectors so we call gesture understand method only once.
								originalTrajectory.clear();
								individualTrajectory.clear();
							}//originalTrajectory.size() != 0
						}
					
					gestureRecognizerModified.drawTrajectoryOnMat(colorMatOriginalSize);
					putText(colorMatOriginalSize,gestureText,Point(50,50),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(0,0,255),2,1);
					 
				} //borderPointsRGreturned.size()
			} 
 
			rectangle( colorMatPyrDown, trackWindowLeft, Scalar(0,0,255), 3, 1);
			Rect rectLarge(trackWindowLeft.tl().x*4,trackWindowLeft.tl().y*4,trackWindowLeft.width*4,trackWindowLeft.height*4);
			rectangle( colorMatOriginalSize, rectLarge, Scalar(0,0,255), 3, 1);
			 
		} 
	
		//this is for next frame where the Rect it belongs for further processing...
		//if trackWindowLeft is greater than some range then do not copy it to selection 
		//keep the old value of selection as it is this will serve the purpose of last 
		//frame unchanged and it shd restrict the size of trackWindowLeft as well 
		if(trackWindowLeft.width < 30 && trackWindowLeft.height < 40 && trackWindowLeft.height > 0 && trackWindowLeft.width > 0 &&
			trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0){
			selectionLeft = trackWindowLeft;	 
		}
		 
		//Find Fingers...
		if(findFingersFlag == 1){
			Mat LeftHand;
			LeftHand = Mat::zeros(depthMatPyrDown.rows,depthMatPyrDown.cols,depthMatPyrDown.type());
			Rect trackWindowLeftExpand;
			expandTrackWindow(trackWindowLeft,trackWindowLeftExpand,15,depthMatPyrDown.cols,depthMatPyrDown.rows);
			//Mat LeftHandEdgeImage = cv::cvarrToMat(edgeIplDepthImage);
			depthMatPyrDown(trackWindowLeftExpand).copyTo(LeftHand(trackWindowLeftExpand));
			//threshold(LeftHand,LeftHand,20,255,THRESH_TOZERO);

			//do manual threshold with cuurent meadian depth because this does not work well..
			
			Mat leftHandUchar;
			
			leftHandUchar = Mat::zeros(depthMatPyrDown.rows,depthMatPyrDown.cols,CV_8U);
			//manualThreshold(LeftHand,20,leftHandUchar);
			thresholdCvMatBasedOnDepthValue(LeftHand,leftHandUchar);
			/*pyrUp(LeftHand,LeftHand);
			pyrUp(LeftHand,LeftHand);*/
			bool isHand = false;
			Point returnPoint;
			//imshow("LeftHand",LeftHand);
			findHand(leftHandUchar,returnPoint,isHand,"left");
		}
	//cout<<"Watch 3 : Manual TW Generation+Gesture After"<<endl;
	} else {	// if(trackWindowLeft.height > 0 && trackWindowLeft.width > 0 && trackBox.boundingRect().area() > 50)
		putText(colorMatOriginalSize,"Left TW = null",Point(50,50),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),1,1);
		rectangle(colorMatPyrDown,trackWindowFromDepthLeft,Scalar(255,255,255),3);
		Rect rectLarge(trackWindowFromDepthLeft.tl().x*4,trackWindowFromDepthLeft.tl().y*4,trackWindowFromDepthLeft.width*4,trackWindowFromDepthLeft.height*4);
		rectangle( colorMatOriginalSize, rectLarge, Scalar(255,255,255), 3, 1);

		counterForLeftTWNull++;		//increament counter for camshift TW == null
		 
	}

	//Kalman tracking
	//use of centerPoint as kalman seed point will give much more smoothed trajectory compare to just finding trackWindow rect center point..  
	//cout<<"Watch 5 : After All cal "<<endl;
	//TODO: if distance left is out of range then dont use kalman...
	//if(!distanceLeftOutOfRangeFlag){ //imp for hand left median distance range 
		// find contours in with ROI(trackWindow) 
		if(trackWindowLeft.height > 0 && trackWindowLeft.width > 0 &&
		trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0){
			vector<vector<Point>> smallCountersPointsTest;
			Mat depthMaskForRGTest;
			depthMaskForRGTest = Mat::zeros(show.size(),show.type());
			show(trackWindowLeft ).copyTo(depthMaskForRGTest(trackWindowLeft ));
			findContours(depthMaskForRGTest,smallCountersPointsTest,0,1);
		 
			//find biggest contour index.
			int iTest = findBiggestContour(smallCountersPointsTest);
			//	 cout<<"iTest : findContours : "<<iTest<<endl;
			Point center1; //very very imp
			if(iTest != -1){
				vector<Point> contourTest = smallCountersPointsTest[iTest];
				Scalar centerTemp1 = mean(Mat(contourTest));
				center1 = Point(centerTemp1.val[0], centerTemp1.val[1]);
			}	
			kalman.runLeft(kfLeft,center1,trackWindowLeft,colorMatOriginalSize,kalmanReturnedWindowLeft,returnedKalmanPointLeft);
			putText(colorMatOriginalSize,"BLACK: Prediction",Point(50,400),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_BLACK_AS_PREDICTION,2,1);
			putText(colorMatOriginalSize,"YELLOW: Measurement",Point(50,450),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_YELLOW_AS_MEASURMENT,2,1);
		} else {
			//Input for kalman
			Point center1;
			center1.x = abs(kalmanReturnedWindowLeft.tl().x - kalmanReturnedWindowLeft.br().x)/2 + kalmanReturnedWindowLeft.tl().x;
			center1.y = abs(kalmanReturnedWindowLeft.tl().y - kalmanReturnedWindowLeft.br().y)/2 + kalmanReturnedWindowLeft.tl().y;
			
			//kalman method
			kalman.runLeft(kfLeft,center1,kalmanReturnedWindowLeft,colorMatOriginalSize,kalmanReturnedWindowLeft,returnedKalmanPointLeft);
			
			//display kalman results
			Rect boundingBoxLarge(kalmanReturnedWindowLeft.tl().x*4,kalmanReturnedWindowLeft.tl().y*4,kalmanReturnedWindowLeft.width*4,kalmanReturnedWindowLeft.height*4);
			//Rect predRectLarge(predRect.tl().x*4,predRect.tl().y*4,predRect.width*4,predRect.height*4);
			cv::rectangle(colorMatOriginalSize, boundingBoxLarge, COLOR_YELLOW_AS_MEASURMENT, 3);
			cv::rectangle(colorMatOriginalSize, boundingBoxLarge, COLOR_BLACK_AS_PREDICTION, 3);
			putText(colorMatOriginalSize,"BLACK: Prediction",Point(50,400),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_BLACK_AS_PREDICTION,2,1);
			putText(colorMatOriginalSize,"YELLOW: Measurement",Point(50,450),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_YELLOW_AS_MEASURMENT,2,1);
		}  
	//cout<<"Watch 6 : Kalman After "<<endl;
} //end of method
void MainMethodClass::trackRightHand(Point3i& handCoordinates){

	calculateBackProjForRight();
	//imshow("backprojRight before",backprojRight);
	bool distanceRightOutOfRangeFlag = false; 
	distanceRightOutOfRangeFlag = createDepthMaskBeforeCamShiftRight();	 
	//imshow("backprojLeft",backprojLeft);
	
	if(trackWindowRight.height > 0 && trackWindowRight.width > 0 &&
		trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0){
		trackBoxRight = CamShift(backprojRight, trackWindowRight,
			TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));		
	}												
	 
	// if the trackBox has area greater than 50, then do some more operations...
	if(trackWindowRight.height > 0 && trackWindowRight.width > 0 && 
		trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0) {	

		if(distanceRightOutOfRangeFlag == true){
			counterForRightTWNull++;		//increament counter if TW is out of range.. for reinitialization
		} else {	 
			//if TW is predicted correctly by camshift then reinitialize counter to 0
			counterForRightTWNull = 0;		//reinitialize counter to 0 for camshift != null
		}

		//works like a charm...
		vector<Point> returnedPoints;
		Point centerPoint;
		//this method takes depth image and depth edge image and centre point, returns two points XL,XH or YL,YH as a vector of point
		
		playWIthShowRight(edgeIplDepthImage,depthIplImagePyrDown,returnedPoints,centerPointRight,show);
		//grow region..
		if(returnedPoints.size() == 2){
			//findEdgeDepthPixelWithMeadianDepthRange(edgeDepthImage,returnedPoints);
			//initialize Region Growth object
			//start point can be XL or YL
			Point startPoint = returnedPoints[0];
			//stop point can be XH or YH
			Point stopPoint = returnedPoints[1];
			if(startPoint.x != 0 && startPoint.y != 0 && stopPoint.x != 0 && stopPoint.y != 0 ){
				//cout<<"b: RG: "<<endl;
				//initialRG = clock();
				GrowRegion rg(edgeIplDepthImage);
				vector<Point> borderPointsRGreturned;
				
				rg.run(edgeIplDepthImage,startPoint,100,stopPoint,borderPointsRGreturned,returnedBorderMatRight);		
				//finalRG =  clock() - initialRG; 
				//cout<<"Final RG Right : " <<(double)finalRG<<endl;
				// sort border points according to trackwindow size.
				if(borderPointsRGreturned.size() != 0){
					vector<Point> sortedPoints;
					//this method will sort the correct point 
					//on x axis either side and return the final points which includes the final close loop
					String flag;
					sortVectorPointsRight(borderPointsRGreturned,sortedPoints,startPoint,stopPoint,flag,show);
					if(trackWindowFromDepthRight.width > 0 && trackWindowFromDepthRight.height > 0 &&
						trackWindowFromDepthRight.tl().x > 0 && trackWindowFromDepthRight.tl().y > 0){
						trackWindowRight = trackWindowFromDepthRight;
					}
					Point3i point;
					 
					//openCVKinect.calculateWorldCoOrdinates(centerPointLeft.x*4,centerPointLeft.y*4,depthDistanceLeft,x,y,z);
					point.x = centerPointRight.x;
					point.y = centerPointRight.y;
					point.z = depthDistanceRight;

					//Assign current hand coordinate value to return handCoordinates
					handCoordinates = Point3i(point.x*4,point.y*4,point.z);
					 
				}
			}
 
			rectangle( colorMatPyrDown, trackWindowRight, Scalar(0,255,0), 3, 1);
			Rect rectLarge(trackWindowRight.tl().x*4,trackWindowRight.tl().y*4,trackWindowRight.width*4,trackWindowRight.height*4);
			rectangle( colorMatOriginalSize, rectLarge, Scalar(0,255,0), 3, 1);
		}
	
		//this is for next frame where the Rect it belongs for further processing...
		//if trackWindowLeft is greater than some range then do not copy it to selection 
		//keep the old value of selection as it is this will serve the purpose of last 
		//frame unchanged and it shd restrict the size of trackWindowLeft as well 
		if(trackWindowRight.width < 30 && trackWindowRight.height < 40 && trackWindowRight.height > 5 && trackWindowRight.width > 5 && 
			trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0){
			selectionRight = trackWindowRight;	 
		} 
	} else {	// if(trackWindowRight.height > 0 && trackWindowRight.width > 0 && trackWindowRight.boundingRect().area() > 50)
		putText(colorMatOriginalSize,"Right TW = null",Point(50,50),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,255),0.1,1);
		rectangle(colorMatPyrDown,trackWindowFromDepthRight,Scalar(255,255,255),3);
		Rect rectLarge(trackWindowFromDepthRight.tl().x*4,trackWindowFromDepthRight.tl().y*4,trackWindowFromDepthRight.width*4,trackWindowFromDepthRight.height*4);
		rectangle(colorMatOriginalSize,rectLarge,Scalar(255,255,255),3);

		counterForRightTWNull++;		//increament counter for camshift TW == null
	}

	//Kalman tracking
	//use of centerPoint as kalman seed point will give much more smoothed trajectory compare to just finding trackWindow rect center point..  
	// find contours in with ROI(trackWindow) 
	if(trackWindowRight.height > 0 && trackWindowRight.width > 0 &&
		trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0){
			vector<vector<Point>> smallCountersPointsTest;
			Mat depthMaskForRGTest;
			depthMaskForRGTest = Mat::zeros(show.size(),show.type());
			show(trackWindowRight).copyTo(depthMaskForRGTest(trackWindowRight));
			findContours(depthMaskForRGTest,smallCountersPointsTest,0,1);
		 
			//find biggest contour index.
			int iTest = findBiggestContour(smallCountersPointsTest);
			//	 cout<<"iTest : findContours : "<<iTest<<endl;
			Point center1; //very very imp
			if(iTest != -1){
				vector<Point> contourTest = smallCountersPointsTest[iTest];
				Scalar centerTemp1 = mean(Mat(contourTest));
				center1 = Point(centerTemp1.val[0], centerTemp1.val[1]);
			}	
			kalman.runRight(kfRight,center1,trackWindowRight,colorMatOriginalSize,kalmanReturnedWindowRight,returnedKalmanPointRight);
			putText(colorMatOriginalSize,"BLACK: Prediction",Point(50,400),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_BLACK_AS_PREDICTION,2,1);
			putText(colorMatOriginalSize,"YELLOW: Measurement",Point(50,450),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_YELLOW_AS_MEASURMENT,2,1);
		} else {
			//Input for kalman
			Point center1;
			center1.x = abs(kalmanReturnedWindowRight.tl().x - kalmanReturnedWindowRight.br().x)/2 + kalmanReturnedWindowRight.tl().x;
			center1.y = abs(kalmanReturnedWindowRight.tl().y - kalmanReturnedWindowRight.br().y)/2 + kalmanReturnedWindowRight.tl().y;
			
			//kalman method
			kalman.runRight(kfRight,center1,kalmanReturnedWindowRight,colorMatOriginalSize,kalmanReturnedWindowRight,returnedKalmanPointRight);
			
			//display kalman results
			Rect boundingBoxLarge(kalmanReturnedWindowRight.tl().x*4,kalmanReturnedWindowRight.tl().y*4,kalmanReturnedWindowRight.width*4,kalmanReturnedWindowRight.height*4);
			//Rect predRectLarge(predRect.tl().x*4,predRect.tl().y*4,predRect.width*4,predRect.height*4);
			cv::rectangle(colorMatOriginalSize, boundingBoxLarge, COLOR_YELLOW_AS_MEASURMENT, 3);
			cv::rectangle(colorMatOriginalSize, boundingBoxLarge, COLOR_BLACK_AS_PREDICTION, 3);
			putText(colorMatOriginalSize,"BLACK: Prediction",Point(50,400),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_BLACK_AS_PREDICTION,2,1);
			putText(colorMatOriginalSize,"YELLOW: Measurement",Point(50,450),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,COLOR_YELLOW_AS_MEASURMENT,2,1);
		}  
} //end of method

//This method will go through sorted Vector poins and find the top/left or bottom/right extrema point
void MainMethodClass::findTheTopPointOfThePalm(vector<Point>&sortedPoints,Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,String flag){
	
	if(sortedPoints.size() == 0){
		topPointPalm.x = 0;
		topPointPalm.y = 0;
		
		leftPointPalm.x = 0;
		leftPointPalm.y = 0;
		
		rightPointPalm.x = 0;
		rightPointPalm.y = 0;
		
		return;
	}
	//if flag == top, these conditions are set: Y==Common & Y < C & we need lowest Y as top point as well as lowest X as Xmin and Highest X as Xmax
	if(flag == "top"){
		//work around Y== constant axis
		topPointPalm.y = sortedPoints[0].y;	// find lowest Y
		topPointPalm.x = sortedPoints[0].x;	// very imp..

		leftPointPalm.x = sortedPoints[0].x; // for top position we will have lowest x as left point
		//very imp.. if you dont assign y at the same time it will be 0 if the first x value is the smallest because if condition will never exicuted.
		leftPointPalm.x = sortedPoints[0].x;  
		

		rightPointPalm.x = sortedPoints[0].x; // for top position we will have highest x as right point
		rightPointPalm.y = sortedPoints[0].y; // very imp..

		for(int i = 0;i < sortedPoints.size();i++){
			if(sortedPoints[i].y < topPointPalm.y){
				topPointPalm.x = sortedPoints[i].x;
				topPointPalm.y = sortedPoints[i].y;
			}
			if(sortedPoints[i].x < leftPointPalm.x){
				leftPointPalm.x = sortedPoints[i].x;
				leftPointPalm.y = sortedPoints[i].y;
			}
			
			if(sortedPoints[i].x > rightPointPalm.x){
				rightPointPalm.x = sortedPoints[i].x;
				rightPointPalm.y = sortedPoints[i].y;
			}
		}
	} else if(flag == "bottom"){ //if flag == bottom, these conditions are set: Y==Common & Y > C & we need highest Y as top point as well as lowest X as Xmin and Highest X as Xmax
		//work around Y == constant axis
		topPointPalm.y = sortedPoints[0].y;	// find highest Y
		topPointPalm.x = sortedPoints[0].x;	// very imp..
		
		leftPointPalm.x = sortedPoints[0].x; // for top position we will have lowest x as left point
		leftPointPalm.y = sortedPoints[0].y; // very imp..

		rightPointPalm.x = sortedPoints[0].x; // for top position we will have highest x as right point
		rightPointPalm.y = sortedPoints[0].y; // very imp..
		for(int i = 0;i < sortedPoints.size();i++){
			if(sortedPoints[i].y > topPointPalm.y){
				topPointPalm.x = sortedPoints[i].x;
				topPointPalm.y = sortedPoints[i].y;
			}
			if(sortedPoints[i].x < leftPointPalm.x){
				leftPointPalm.x = sortedPoints[i].x;
				leftPointPalm.y = sortedPoints[i].y;
			}
			
			if(sortedPoints[i].x > rightPointPalm.x){
				rightPointPalm.x = sortedPoints[i].x;
				rightPointPalm.y = sortedPoints[i].y;
			}
		}
	} else if(flag == "left"){ //if flag == left, these conditions are set: X==Common & X < C & we need lowest X as top point as well as lowest Y as Ymin and Highest Y as Ymax
		//work around X == constant axis
		topPointPalm.x = sortedPoints[0].x;	// find lowest X
		topPointPalm.y = sortedPoints[0].y;	// very imp..

		leftPointPalm.y = sortedPoints[0].y; // for top position we will have lowest y as left point
		leftPointPalm.x = sortedPoints[0].x; // very imp..

		rightPointPalm.y = sortedPoints[0].y; // for top position we will have highest y as right point
		rightPointPalm.x = sortedPoints[0].x; // very imp..

		for(int i = 0;i < sortedPoints.size();i++){
			if(sortedPoints[i].x < topPointPalm.x){
				topPointPalm.x = sortedPoints[i].x;
				topPointPalm.y = sortedPoints[i].y;
			}
			if(sortedPoints[i].y < leftPointPalm.y){
				leftPointPalm.x = sortedPoints[i].x;
				leftPointPalm.y = sortedPoints[i].y;
			}
			
			if(sortedPoints[i].y > rightPointPalm.y){
				rightPointPalm.x = sortedPoints[i].x;
				rightPointPalm.y = sortedPoints[i].y;
			}
		}
	} else if(flag == "right"){//if flag == right, these conditions are set: X==Common & X > C & we need highest X as top point as well as lowest Y as Ymin and Highest Y as Ymax
		//work around X == constant axis
		topPointPalm.x = sortedPoints[0].x;	// find highest X
		topPointPalm.y = sortedPoints[0].y;	// very imp..

		leftPointPalm.y = sortedPoints[0].y; // for top position we will have lowest y as left point
		leftPointPalm.x = sortedPoints[0].x; // very imp..

		rightPointPalm.y = sortedPoints[0].y; // for top position we will have highest y as right point
		rightPointPalm.x = sortedPoints[0].x; // very imp..

		for(int i = 0;i < sortedPoints.size();i++){
			if(sortedPoints[i].x > topPointPalm.x){
				topPointPalm.x = sortedPoints[i].x;
				topPointPalm.y = sortedPoints[i].y;
			}
			if(sortedPoints[i].y < leftPointPalm.y){
				leftPointPalm.x = sortedPoints[i].x;
				leftPointPalm.y = sortedPoints[i].y;
			}
			
			if(sortedPoints[i].y > rightPointPalm.y){
				rightPointPalm.x = sortedPoints[i].x;
				rightPointPalm.y = sortedPoints[i].y;
			}
		}
	}	
}

//this method will sort the correct point 
//on x axis either side and return the final points which includes the final close loop
void MainMethodClass::sortVectorPointsLeft(vector<Point>& srcVecPoints,vector<Point>& sortedPoints,Point startPoint,Point stopPoint,String& flag,cv::Mat& show){
	vector<Point> sortVec1,sortVec2;

	//add star point to both vecotrs..
	/*sortVec1.push_back(startPoint);
	sortVec2.push_back(startPoint);*/
	Point startPointTemp = startPoint;
	//check on the trackWindowSize, and then sort acconrdingly around the x or y axis.
	if(trackWindowLeft.height >= trackWindowLeft.width){
		// this would be horizontal position, ie XL and XH, hence try to filter on common X axis..
		//go through backwards..
		vector<Point> tempSortVect;
		int topCount = 0;
		int bottomCount = 0;
		Point tempPoint = stopPoint;
		for(int j = srcVecPoints.size() - 1; j > 0 ; j--){
			//find the closest point of stoppoint 
			if(abs(srcVecPoints[j].x - tempPoint.x) <= 1 && abs(srcVecPoints[j].y - tempPoint.y) <= 1){
				//check if that closest point has y == top or bottom, or third condition, in that case continue..
				tempSortVect.push_back(srcVecPoints[j]); 
				tempPoint = srcVecPoints[j];
				srcVecPoints.erase(srcVecPoints.begin()+j);		//very imp..if you remove the current seedpoint then it will not appear again in the loop!
			} 
		}

		for(int ix = 0 ; ix < tempSortVect.size() ; ix++){
			if(tempSortVect[ix].y < stopPoint.y){
				topCount++;
			} else if(tempSortVect[ix].y > stopPoint.y){
				bottomCount++;
			}

			circle(show,tempSortVect[ix],1,Scalar(255,255,255),1);
		}
		 
		if((topCount > bottomCount)){
			flag = "top";
			sortedPoints = tempSortVect;
			flagForDirectionLeft = flag;
			
			/*for(int iix = 0 ; iix < sortVec1.size(); iix++){
				circle(show,sortVec1[iix],1,Scalar(255,255,255),1);
			}*/
		} else if(topCount < bottomCount){ //here the previous frame flag can not be top.. 
			flag = "bottom";
			sortedPoints = tempSortVect;
			flagForDirectionLeft = flag;
			/*for(int iix = 0 ; iix < sortVec2.size(); iix++){
				circle(show,sortVec2[iix],1,Scalar(255,255,255),1);
			}*/
		}

		 
		
		 
	} else if(trackWindowLeft.height < trackWindowLeft.width){
		// this would be vertical position, ie YL and YH, hence try to filter on common Y axis..
		// try to sort vector points on YL.x/YH.x right or left..
		//go through backwards..
		vector<Point> tempSortVect;
		int leftCount = 0;
		int rightCount = 0;
		Point tempPoint = stopPoint;
		for(int j = srcVecPoints.size() - 1; j > 0 ; j--){
			//find the closest point of stoppoint 
			if(abs(srcVecPoints[j].x - tempPoint.x) <= 1 && abs(srcVecPoints[j].y - tempPoint.y) <= 1){
				//check if that closest point has y == top or bottom, or third condition, in that case continue..
				tempSortVect.push_back(srcVecPoints[j]); 
				tempPoint = srcVecPoints[j];
				srcVecPoints.erase(srcVecPoints.begin()+j);		//very imp..if you remove the current seedpoint then it will not appear again in the loop!
			} 
		}

		for(int ix = 0 ; ix < tempSortVect.size() ; ix++){
			if(tempSortVect[ix].x < stopPoint.x){
				leftCount++;
			} else if(tempSortVect[ix].x > stopPoint.x){
				rightCount++;
			}

			circle(show,tempSortVect[ix],1,Scalar(255,255,255),1);
		}  	
		 
		if((leftCount > rightCount)){
			flag = "left";
			sortedPoints = tempSortVect;
			flagForDirectionLeft = flag;
			 /*for(int iix = 0 ; iix < sortVec1.size(); iix++){
				circle(show,sortVec1[iix],1,Scalar(255,255,255),1);
			}*/
		} else if(leftCount < rightCount){ //here the previous frame flag can not be top.. 
			flag = "right";
			sortedPoints = tempSortVect;
			flagForDirectionLeft = flag;
			/*for(int iix = 0 ; iix < sortVec2.size(); iix++){
				circle(show,sortVec2[iix],1,Scalar(255,255,255),1);
			} */
		}

	}

	Point topPointPalm,leftPointPalm,rightPointPalm;
	findTheTopPointOfThePalm(sortedPoints,topPointPalm,leftPointPalm,rightPointPalm,flag);
	circle(show,topPointPalm,2,Scalar(255,255,0),1);
	putText(show,"TP",Point(topPointPalm.x,topPointPalm.y),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
	
	drawRectOnShowLeft(topPointPalm,leftPointPalm,rightPointPalm,show,flag);	
	putText(show,"Flag Left: " + flag,Point(10,10),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
	 
} //EOM

//this method will sort the correct point 
//on x axis either side and return the final points which includes the final close loop
void MainMethodClass::sortVectorPointsRight(vector<Point>& srcVecPoints,vector<Point>& sortedPoints,Point startPoint,Point stopPoint,String& flag,cv::Mat& show){
	vector<Point> sortVec1,sortVec2;

	//add star point to both vecotrs..
	Point startPointTemp = startPoint;
	//check on the trackWindowSize, and then sort acconrdingly around the x or y axis.
	if(trackWindowRight.height >= trackWindowRight.width){
		// this would be horizontal position, ie XL and XH, hence try to filter on common X axis..
		// try to sort vector points on XL.y/XH.y above or below..
		  
		//go through backwards..
		vector<Point> tempSortVect;
		int topCount = 0;
		int bottomCount = 0;
		Point tempPoint = stopPoint;
		for(int j = srcVecPoints.size() - 1; j > 0 ; j--){
			//find the closest point of stoppoint 
			if(abs(srcVecPoints[j].x - tempPoint.x) <= 1 && abs(srcVecPoints[j].y - tempPoint.y) <= 1){
				//check if that closest point has y == top or bottom, or third condition, in that case continue..
				tempSortVect.push_back(srcVecPoints[j]); 
				tempPoint = srcVecPoints[j];
				srcVecPoints.erase(srcVecPoints.begin()+j);		//very imp..if you remove the current seedpoint then it will not appear again in the loop!
			} 
		}

		for(int ix = 0 ; ix < tempSortVect.size() ; ix++){
			if(tempSortVect[ix].y < stopPoint.y){
				topCount++;
			} else if(tempSortVect[ix].y > stopPoint.y){
				bottomCount++;
			}

			circle(show,tempSortVect[ix],1,Scalar(255,255,255),1);
		}
		
		if((topCount > bottomCount)){
			flag = "top";
			sortedPoints = tempSortVect;
			flagForDirectionRight = flag;
			 
		} else if(topCount < bottomCount){ //here the previous frame flag can not be top.. 
			flag = "bottom";
			sortedPoints = tempSortVect;
			flagForDirectionRight = flag;
			 
		}
 
	} else if(trackWindowRight.height < trackWindowRight.width){
		// this would be vertical position, ie YL and YH, hence try to filter on common Y axis..
		// try to sort vector points on YL.x/YH.x right or Right..
		 
		//go through backwards..
		vector<Point> tempSortVect;
		int leftCount = 0;
		int rightCount = 0;
		Point tempPoint = stopPoint;
		for(int j = srcVecPoints.size() - 1; j > 0 ; j--){
			//find the closest point of stoppoint 
			if(abs(srcVecPoints[j].x - tempPoint.x) <= 1 && abs(srcVecPoints[j].y - tempPoint.y) <= 1){
				//check if that closest point has y == top or bottom, or third condition, in that case continue..
				tempSortVect.push_back(srcVecPoints[j]); 
				tempPoint = srcVecPoints[j];
				srcVecPoints.erase(srcVecPoints.begin()+j);		//very imp..if you remove the current seedpoint then it will not appear again in the loop!
			} 
		}

		for(int ix = 0 ; ix < tempSortVect.size() ; ix++){
			if(tempSortVect[ix].x < stopPoint.x){
				leftCount++;
			} else if(tempSortVect[ix].x > stopPoint.x){
				rightCount++;
			}

			circle(show,tempSortVect[ix],1,Scalar(255,255,255),1);
		}
		
		if((leftCount > rightCount)){
			flag = "left";
			sortedPoints = tempSortVect;
			flagForDirectionRight = flag;
			 
		} else if(leftCount < rightCount){ //here the previous frame flag can not be top.. 
			flag = "right";
			sortedPoints = tempSortVect;
			flagForDirectionRight = flag;
			 
		}

	}

	Point topPointPalm,leftPointPalm,rightPointPalm;
	findTheTopPointOfThePalm(sortedPoints,topPointPalm,leftPointPalm,rightPointPalm,flag);
	circle(show,topPointPalm,2,Scalar(255,255,0),1);
	putText(show,"TP",Point(topPointPalm.x,topPointPalm.y),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
	
	drawRectOnShowRight(topPointPalm,leftPointPalm,rightPointPalm,show,flag);
	
	putText(show,"Flag Right : " + flag,Point(10,30),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
} //EOM

void MainMethodClass::drawRectOnShowLeft(Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,Mat& show,String& flag){

	//check points are in range or not..
	int showCols = show.cols;
	int showRows = show.rows;
	checkPointInRange(topPointPalm,showCols,showRows);
	checkPointInRange(leftPointPalm,showCols,showRows);
	checkPointInRange(rightPointPalm,showCols,showRows);
		Rect newRectFromThePoints;
		
			
		if(flag == "top"){
			float width = abs(rightPointPalm.x - leftPointPalm.x);
			if(width <= 10){
				width = 10;
			}

			if(width >=20){
				width = 20;
			}
		
			float height = width*widthHeightRationFactor;
			//Point topRectPoint(leftPointPalm.x,topPointPalm.y);
			if((leftPointPalm.x + width ) > showCols){
				width = abs(leftPointPalm.x - showCols);
			}
			if((topPointPalm.y + height ) > showRows){
				height = abs(topPointPalm.y - showRows);
			}
			newRectFromThePoints  = Rect(leftPointPalm.x,topPointPalm.y,width,height);
			 
		} else if(flag == "bottom"){//TODO: try to find batter way...........
			
			float width = abs(rightPointPalm.x - leftPointPalm.x);
			if(width <= 10){
				width = 10;
			}
			if(width >=20){
				width = 20;
			}
			if((rightPointPalm.x ) > showCols){
				width = abs(leftPointPalm.x - showCols);
			}
			if((topPointPalm.y ) > showRows){
				topPointPalm.y = topPointPalm.y;
			}
			float height = width*widthHeightRationFactor;
			Point topRectPoint(rightPointPalm.x-width,topPointPalm.y-height);
			newRectFromThePoints  = Rect(topRectPoint.x,topRectPoint.y,width,height);
		 
		}  else if(flag == "left"){
		
			float height = abs(rightPointPalm.y - leftPointPalm.y);
			if(height <= 10){
				height = 10;
			}
			if(height >= 20){
				height = 20;
			}
			float width = height*widthHeightRationFactor;
			Point topRectPoint(topPointPalm.x,leftPointPalm.y);

			if((topPointPalm.x + width ) > showCols){
				width = abs(topPointPalm.x - showCols);
			}
			if((leftPointPalm.y + height ) > showRows){
				height = abs(leftPointPalm.y - showRows);
			}
			newRectFromThePoints  =Rect(topPointPalm.x,leftPointPalm.y,width,height);
			 
		} else if(flag == "right"){
			float height = abs(rightPointPalm.y - leftPointPalm.y);
			if(height <= 10){
				height = 10;
			}
			if(height >= 20){
				height = 20;
			}
		 
			if((rightPointPalm.y ) > showRows){
				height = abs(leftPointPalm.y - showRows);
			}
			if((topPointPalm.x ) > showCols){
				topPointPalm.x = topPointPalm.x;
			}
			float width = height*widthHeightRationFactor;
			Point topRectPoint(topPointPalm.x-width,rightPointPalm.y-height);
			 
 
			newRectFromThePoints  = Rect(topRectPoint.x,topRectPoint.y,width,height);
		}
	 
		if(newRectFromThePoints.width > 0 && newRectFromThePoints.height > 0){
			trackWindowFromDepthLeft = newRectFromThePoints;
			rectangle(show,newRectFromThePoints,Scalar(255,255,255),1);
		}
	  

}

void MainMethodClass::checkPointInRange(Point& point,int maxColsRange,int maxRowSize){
	if(point.x <= 1){
		point.x = 0;
	} 
	if(point.y <= 1){
		point.y = 0;
	}
	if(point.x >= maxColsRange){
		point.x = maxColsRange;
	}
		
	if(point.y >= maxColsRange){
		point.y = maxColsRange;
	}
}

void MainMethodClass::drawRectOnShowRight(Point& topPointPalm,Point& leftPointPalm,Point& rightPointPalm,Mat& show,String& flag){

	//check points are in range or not..
	int showCols = show.cols;
	int showRows = show.rows;
	checkPointInRange(topPointPalm,showCols,showRows);
	checkPointInRange(leftPointPalm,showCols,showRows);
	checkPointInRange(rightPointPalm,showCols,showRows);
		Rect newRectFromThePoints;
		
			
		if(flag == "top"){
			float width = abs(rightPointPalm.x - leftPointPalm.x);
			if(width <= 10){
				width = 10;
			}

			if(width >=20){
				width = 20;
			}
		
			float height = width*widthHeightRationFactor;
			//Point topRectPoint(leftPointPalm.x,topPointPalm.y);
			if((leftPointPalm.x + width ) > showCols){
				width = abs(leftPointPalm.x - showCols);
			}
			if((topPointPalm.y + height ) > showRows){
				height = abs(topPointPalm.y - showRows);
			}
			newRectFromThePoints  = Rect(leftPointPalm.x,topPointPalm.y,width,height);
			 
		} else if(flag == "bottom"){//TODO: try to find batter way...........
			
			float width = abs(rightPointPalm.x - leftPointPalm.x);
			if(width <= 10){
				width = 10;
			}
			if(width >=20){
				width = 20;
			}
			if((rightPointPalm.x ) > showCols){
				width = abs(leftPointPalm.x - showCols);
			}
			if((topPointPalm.y ) > showRows){
				topPointPalm.y = topPointPalm.y;
			}
			float height = width*widthHeightRationFactor;
			Point topRectPoint(rightPointPalm.x-width,topPointPalm.y-height);
			newRectFromThePoints  = Rect(topRectPoint.x,topRectPoint.y,width,height);
		 
		}  else if(flag == "left"){
		
			float height = abs(rightPointPalm.y - leftPointPalm.y);
			if(height <= 10){
				height = 10;
			}
			if(height >= 20){
				height = 20;
			}
			float width = height*widthHeightRationFactor;
			Point topRectPoint(topPointPalm.x,leftPointPalm.y);

			if((topPointPalm.x + width ) > showCols){
				width = abs(topPointPalm.x - showCols);
			}
			if((leftPointPalm.y + height ) > showRows){
				height = abs(leftPointPalm.y - showRows);
			}
			newRectFromThePoints  =Rect(topPointPalm.x,leftPointPalm.y,width,height);
			 
		} else if(flag == "right"){
			float height = abs(rightPointPalm.y - leftPointPalm.y);
			if(height <= 10){
				height = 10;
			}
			if(height >= 20){
				height = 20;
			}
		 
			if((rightPointPalm.y ) > showRows){
				height = abs(leftPointPalm.y - showRows);
			}
			if((topPointPalm.x ) > showCols){
				topPointPalm.x = topPointPalm.x;
			}
			float width = height*widthHeightRationFactor;
			Point topRectPoint(topPointPalm.x-width,rightPointPalm.y-height);
			 
 
			newRectFromThePoints  = Rect(topRectPoint.x,topRectPoint.y,width,height);
		}
	 
		if(newRectFromThePoints.width > 0 && newRectFromThePoints.height > 0){
			trackWindowFromDepthRight = newRectFromThePoints;
			rectangle(show,newRectFromThePoints,Scalar(255,255,255),1);
		}
		
}

//Para: returnedPoints : return a vector of Poins which include XL,XH or YL,YH combination. 
//Para: centerPoint : returned center point of depth image along wiht XL,XH or YL,YH
//Para: input edge ipl image
//Para: input depth image image
void MainMethodClass::playWIthShowLeft(IplImage* edgeDepthImage,IplImage* depthIplImage,vector<Point>& returnedPoints,Point& centerPoint,Mat& depthShow){
			 
			int thresholdForDepthDifference = 50;
			  
			//find center point of this biggest contour
			//if(firstTimeFlag == true){
				// find contours in with ROI(trackWindow) 
				vector<vector<Point>> smallCountersPointsTest;
				Mat depthMaskForRGTest;
				depthMaskForRGTest = Mat::zeros(depthShow.size(),depthShow.type());
				depthShow(trackWindowLeft ).copyTo(depthMaskForRGTest(trackWindowLeft ));
				findContours(depthMaskForRGTest,smallCountersPointsTest,0,1);
				//	cout<<"findContours : "<<endl;
				//find biggest contour index.
				int iTest = findBiggestContour(smallCountersPointsTest);
				//	 cout<<"iTest : findContours : "<<iTest<<endl;
				Point center1; //very very imp
				if(iTest != -1){
					vector<Point> contourTest = smallCountersPointsTest[iTest];
					Scalar centerTemp1 = mean(Mat(contourTest));
					center1 = Point(centerTemp1.val[0], centerTemp1.val[1]);
				}
			  
				//write this Points on Image
				circle(depthShow,center1,2,Scalar(255,255,0),1);
				putText(depthShow,"T",center1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
				
				//return point
				centerPoint = center1;
				  
				findcorrospondindPointsInDepthImageLeft(returnedPoints,center1);
				
} //end method

//Para: returnedPoints : return a vector of Poins which include XL,XH or YL,YH combination. 
//Para: centerPoint : returned center point of depth image along wiht XL,XH or YL,YH
//Para: input edge ipl image
//Para: input depth image image
void MainMethodClass::playWIthShowRight(IplImage* edgeDepthImage,IplImage* depthIplImage,vector<Point>& returnedPoints,Point& centerPoint,Mat& depthShow){
			 
			int thresholdForDepthDifference = 50;
			  
			//find center point of this biggest contour
			//if(firstTimeFlag == true){
				// find contours in with ROI(trackWindow) 
				vector<vector<Point>> smallCountersPointsTest;
				Mat depthMaskForRGTest;
				depthMaskForRGTest = Mat::zeros(depthShow.size(),depthShow.type());
				depthShow(trackWindowRight).copyTo(depthMaskForRGTest(trackWindowRight));
				findContours(depthMaskForRGTest,smallCountersPointsTest,0,1);
				//	cout<<"findContours : "<<endl;
				//find biggest contour index.
				int iTest = findBiggestContour(smallCountersPointsTest);
				//	 cout<<"iTest : findContours : "<<iTest<<endl;
				Point center1;
				if(iTest != -1){
					vector<Point> contourTest = smallCountersPointsTest[iTest];
					Scalar centerTemp1 = mean(Mat(contourTest));
					center1 = Point(centerTemp1.val[0], centerTemp1.val[1]);
				}
			  
				//write this Points on Image
				circle(depthShow,centerPoint,2,Scalar(255,255,0),1);
				putText(depthShow,"T",center1,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);
				
				//return point
				centerPoint = center1;
				  
				findcorrospondindPointsInDepthImageRight(returnedPoints,center1);
		 
} //end method

void MainMethodClass::findcorrospondindPointsInDepthImageLeft(vector<Point>& returnedPoints,Point& centerPoint){
	Point XL,XH,YL,YH;
				int thresholdForDepthDifference = 50;
				//if trackWindow.height >= trackWindow.width
				if(trackWindowLeft.height >= trackWindowLeft.width){
					//check for XLow point in same row as center point
					unsigned short XLDepth  = 0; 
					for(int cn = centerPoint.x ; cn > 0 ; cn-- ){
						XLDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * centerPoint.y]))[cn];
						if(abs(XLDepth - depthDistanceLeft) < thresholdForDepthDifference){
							XL.x = cn;
							XL.y = centerPoint.y;
						} else {
							break;
						}
					}
			 
					//check for XHigh point in same row as center point
					unsigned short XHDepth  = 0; 
					for(int cn1 = centerPoint.x ; cn1 < depthIplImagePyrDown->width ; cn1++ ){
						XHDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * centerPoint.y]))[cn1];
						if(abs(XHDepth - depthDistanceLeft) < thresholdForDepthDifference){
							XH.x = cn1;
							XH.y = centerPoint.y;
						} else {
							break;
						}
					}

					//check if the final XL and XH points has same depth as global meadian or not
					unsigned short edgeXLDepth = 0;
					edgeXLDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * XL.y]))[XL.x];	 
					if(abs(edgeXLDepth - depthDistanceLeft) < thresholdForDepthDifference){
						returnedPoints.push_back(XL);
					}
					unsigned short edgeXHDepth  = 0;
					edgeXHDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * XH.y]))[XH.x];
					if(abs(edgeXHDepth - depthDistanceLeft) < thresholdForDepthDifference){
						returnedPoints.push_back(XH);
					}
					
				} else {		//if trackWindow.width > trackWindow.height
					
					//check for YLow point in same column as center point
					unsigned short YLDepth  = 0; 
					for(int cn = centerPoint.y ; cn > 0 ; cn-- ){
						YLDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * cn]))[centerPoint.x];
						if(abs(YLDepth - depthDistanceLeft) < thresholdForDepthDifference){
							YL.x = centerPoint.x;
							YL.y = cn;
						} else {
							break;
						}
					}
			 
					//check for YHigh point in same row as center point
					unsigned short YHDepth  = 0; 
					for(int cn1 = centerPoint.y ; cn1 < depthIplImagePyrDown->height ; cn1++ ){
						YHDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * cn1]))[centerPoint.x];
						if(abs(YHDepth - depthDistanceLeft) < thresholdForDepthDifference){
							YH.x = centerPoint.x;
							YH.y = cn1;
						 
						} else {
							break;
						}
					}
					
					//check if the final YL and YH points has same depth as global meadian or not
					unsigned short edgeYLDepth  = 0;
					edgeYLDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * YL.y]))[YL.x];	 
					if(abs(edgeYLDepth - depthDistanceLeft) < thresholdForDepthDifference){
						returnedPoints.push_back(YL);
					}
					unsigned short edgeYHDepth  = 0;
					edgeYHDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * YH.y]))[YH.x];

					if(abs(edgeYHDepth - depthDistanceLeft) < thresholdForDepthDifference){
						returnedPoints.push_back(YH);
					}
					//cout<<"YH : "<<edgeYHDepth<< " , YL : "<< edgeYLDepth<<endl;
				} //end of else
}

void MainMethodClass::findcorrospondindPointsInDepthImageRight(vector<Point>& returnedPoints,Point& centerPoint){
	Point XL,XH,YL,YH;
				int thresholdForDepthDifference = 50;
				//if trackWindow.height >= trackWindow.width
				if(trackWindowRight.height >= trackWindowRight.width){
					//check for XLow point in same row as center point
					unsigned short XLDepth  = 0; 
					for(int cn = centerPoint.x ; cn > 0 ; cn-- ){
						XLDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * centerPoint.y]))[cn];
						if(abs(XLDepth - depthDistanceRight) < thresholdForDepthDifference){
							XL.x = cn;
							XL.y = centerPoint.y;
						} else {
							break;
						}
					}
			 
					//check for XHigh point in same row as center point
					unsigned short XHDepth  = 0; 
					for(int cn1 = centerPoint.x ; cn1 < depthIplImagePyrDown->width ; cn1++ ){
						XHDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * centerPoint.y]))[cn1];
						if(abs(XHDepth - depthDistanceRight) < thresholdForDepthDifference){
							XH.x = cn1;
							XH.y = centerPoint.y;
						} else {
							break;
						}
					}

					//write this Points on Image
					/*putText(depthShow,"XL",Point(XL.x-20,XL.y),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);	
					circle(depthShow,XL,2,Scalar(255,255,0),1);

					putText(depthShow,"XH",Point(XH.x+10,XH.y),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,255),0.1,1);	
					circle(depthShow,XH,2,Scalar(255,255,255),1);*/

					//check if the final XL and XH points has same depth as global meadian or not
					unsigned short edgeXLDepth = 0;
					edgeXLDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * XL.y]))[XL.x];	 
					if(abs(edgeXLDepth - depthDistanceRight) < thresholdForDepthDifference){
						returnedPoints.push_back(XL);
					}
					unsigned short edgeXHDepth  = 0;
					edgeXHDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * XH.y]))[XH.x];
					if(abs(edgeXHDepth - depthDistanceRight) < thresholdForDepthDifference){
						returnedPoints.push_back(XH);
					}
					
					 
					
				} else {		//if trackWindow.width > trackWindow.height
					
					//check for YLow point in same column as center point
					unsigned short YLDepth  = 0; 
					for(int cn = centerPoint.y ; cn > 0 ; cn-- ){
						YLDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * cn]))[centerPoint.x];
						if(abs(YLDepth - depthDistanceRight) < thresholdForDepthDifference){
							YL.x = centerPoint.x;
							YL.y = cn;
						} else {
							break;
						}
					}
			 
					//check for YHigh point in same row as center point
					unsigned short YHDepth  = 0; 
					for(int cn1 = centerPoint.y ; cn1 < depthIplImagePyrDown->height ; cn1++ ){
						YHDepth  = ((unsigned short *)&(depthIplImagePyrDown->imageData[depthIplImagePyrDown->widthStep * cn1]))[centerPoint.x];
						if(abs(YHDepth - depthDistanceRight) < thresholdForDepthDifference){
							YH.x = centerPoint.x;
							YH.y = cn1;
						 
						} else {
							break;
						}
					}
					//write this Points on Image
					/*putText(depthShow,"YL",Point(YL.x,YL.y-10),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,0),0.1,1);	
					circle(depthShow,YL,2,Scalar(255,255,0),1);

					putText(depthShow,"YH",Point(YH.x,YH.y+10),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,255),0.1,1);	
					circle(depthShow,YH,2,Scalar(255,255,255),1);*/
					//check if the final YL and YH points has same depth as global meadian or not
					
					unsigned short edgeYLDepth  = 0;
					edgeYLDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * YL.y]))[YL.x];	 
					if(abs(edgeYLDepth - depthDistanceRight) < thresholdForDepthDifference){
						returnedPoints.push_back(YL);
					}
					unsigned short edgeYHDepth  = 0;
					edgeYHDepth = ((unsigned short *)&(edgeIplDepthImage->imageData[edgeIplDepthImage->widthStep * YH.y]))[YH.x];

					if(abs(edgeYHDepth - depthDistanceRight) < thresholdForDepthDifference){
						returnedPoints.push_back(YH);
					}
					//cout<<"YH : "<<edgeYHDepth<< " , YL : "<< edgeYLDepth<<endl;
				} //end of else
}

void MainMethodClass::calculateBackProjForLeft(){
	
	//Assign se´lection window to track-window and use track-window now onwards..
	trackWindowLeft = selectionLeft;

	////calculate histogram
	if(dynamicThresholding == 1){
		float hranges[] = {0,180};
		const float* phranges = hranges;
                   
		// Left selection window
		Mat roiLeft(hue, selectionLeft), maskroiLeft(mask1, selectionLeft);
		calcHist(&roiLeft, 1, 0, maskroiLeft, histLeft, 1, &hsize, &phranges);
		normalize(histLeft, histLeft, 0, 255, CV_MINMAX);
	 //                   
	
		calcBackProject(&hue, 1, 0, histLeft, backprojLeft, &phranges);
		backprojLeft &= mask1;

		HistThresholding(histLeft,dynamicHueMin1,dynamicHueMin2,dynamicHueMax1,dynamicHueMax2);
		displayHistogram(histLeft,"left");
	//imshow( "BP:1", backprojLeft );
	} else {	//if dynamic thresholding is not enable then use ROI cropped-mask as backProjection Image 
		backprojLeft = Mat::zeros(mask1.rows,mask1.cols,mask1.type());
		Rect newTrackwidnowLeft;
		expandTrackWindow(trackWindowLeft,newTrackwidnowLeft,15,mask1.cols,mask1.rows);
		if(newTrackwidnowLeft.tl().x >= 0 && newTrackwidnowLeft.tl().y >= 0){
			mask1(newTrackwidnowLeft).copyTo(backprojLeft(newTrackwidnowLeft));
		} else if(trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0){
			mask1(trackWindowLeft).copyTo(backprojLeft(trackWindowLeft));
		}
	} 
	//imshow( "BP:2", backprojLeft );
	if( trackWindowLeft.area() <= 50 )
	{
		/*int cols = backprojLeft.cols, rows = backprojLeft.rows, r = (MIN(cols, rows) + 5)/6;
		trackWindowLeft = Rect(abs(trackWindowLeft.x - r), abs(trackWindowLeft.y - r),
			abs(trackWindowLeft.x + r), abs(trackWindowLeft.y + r)) &
				Rect(0, 0, cols, rows);*/
		trackObjectLeft = 0;
		histLeft = 0;
		histimg = 0;
		//return;
	} 

	 
}

void MainMethodClass::calculateBackProjForRight(){
	//Assign se´lection window to track-window and use track-window now onwards..
	trackWindowRight = selectionRight;

	////calculate histogram
	if(dynamicThresholding == 1){
		float hranges[] = {0,180};
		const float* phranges = hranges;
		// Right selection window
		Mat roiRight(hue, selectionRight), maskroiRight(mask1, selectionRight);
		calcHist(&roiRight, 1, 0, maskroiRight, histRight, 1, &hsize, &phranges);
		normalize(histRight, histRight, 0, 255, CV_MINMAX);
	
		calcBackProject(&hue, 1, 0, histRight, backprojRight, &phranges);

		int dynamicHueMin1Right,dynamicHueMin2Right,dynamicHueMax1Right,dynamicHueMax2Right;
		HistThresholding(histLeft,dynamicHueMin1Right,dynamicHueMin2Right,dynamicHueMax1Right,dynamicHueMax2Right);

		//if left and right both hands are initialized and we need to set limit with only one hand
		if(dynamicHueMin1Right < dynamicHueMin1){
			dynamicHueMin1 = dynamicHueMin1Right;
		}

		if(dynamicHueMin2Right < dynamicHueMin2){
			dynamicHueMin2 = dynamicHueMin2Right;
		}

		if(dynamicHueMax1Right > dynamicHueMax1){
			dynamicHueMax1 = dynamicHueMax1Right;
		}

		if(dynamicHueMax2Right > dynamicHueMax2){
			dynamicHueMax2 = dynamicHueMax2Right;
		}

		displayHistogram(histRight,"right");
	} else {	//if dynamic thresholding is not enable then use ROI cropped-mask as backProjection Image 
		backprojRight = Mat::zeros(mask1.rows,mask1.cols,mask1.type());

		Rect newTrackwidnowRight;
		expandTrackWindow(trackWindowRight,newTrackwidnowRight,15,mask1.cols,mask1.rows);

		//in order to get rid of possibility of getting -ve coordinates with extension of the window
		if(newTrackwidnowRight.tl().x > 0 && newTrackwidnowRight.tl().y > 0){		
			mask1(newTrackwidnowRight).copyTo(backprojRight(newTrackwidnowRight));
		} else if(trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0){	
			mask1(trackWindowRight).copyTo(backprojRight(trackWindowRight));
		}
	}
	////imshow( "backproj befor", backproj );
	//
	
	
	if( trackWindowRight.area() <= 50 )
	{
		/*int cols = backprojRight.cols, rows = backprojRight.rows, r = (MIN(cols, rows) + 5)/6;
		trackWindowRight = Rect(abs(trackWindowRight.x - r), abs(trackWindowRight.y - r),
			abs(trackWindowRight.x + r), abs(trackWindowRight.y + r)) &
				Rect(0, 0, cols, rows);*/
		trackObjectRight = 0;
		histRight = 0;
		histimg = 0;
		//return;
	} 	
}

bool MainMethodClass::createDepthMaskBeforeCamShiftLeft(){
	bool returnFlag;
	//it is the depth image thresholding mask based on meadian depth..
	cvSet(tempImageLeft,Scalar(0));
	
	cvSet(edgeMaskImageLeft,Scalar(0));
	//do operation for left hand... 
	if(trackWindowLeft.height > 0 && trackWindowLeft.width > 0 && 
		trackWindowLeft.tl().x >= 0 && trackWindowLeft.tl().y >= 0){
		
		//depthDistanceLeft is a global variable which can be utilized in a good smart way..
		returnFlag = findMeadianOftheTrakWindowPixelsInDepthImageLeft(depthMatPyrDown,depthDistanceLeft);
		//cout<<"distance left meadian" <<depthDistanceLeft<<endl; 
		thresholdIplImageBasedOnDepthValue(depthIplImagePyrDown,depthDistanceLeft ,tempImageLeft);
		cvErode(tempImageLeft,tempImageLeft,0,1);
		cvDilate(tempImageLeft,tempImageLeft,0,1);
		 
		 

		// for left hand
		Mat depthMask = cv::cvarrToMat(tempImageLeft);
	 
		Rect modifiedTrackWindowLeft;
		expandTrackWindow(trackWindowLeft,modifiedTrackWindowLeft,10,depthMask.cols,depthMask.rows);

		Mat tempBPL = Mat::zeros(backprojLeft.size(),backprojLeft.type());
		Mat tempDepthMask = Mat::zeros(depthMask.size(),depthMask.type());

		//in order to get rid of possibility of getting -ve coordinates with extension of the window
		if(modifiedTrackWindowLeft.tl().x >= 0 && modifiedTrackWindowLeft.tl().y >= 0){		
			backprojLeft(modifiedTrackWindowLeft).copyTo(tempBPL(modifiedTrackWindowLeft)); 
			depthMask(modifiedTrackWindowLeft).copyTo(tempDepthMask(modifiedTrackWindowLeft)); 		 
		} else {
			backprojLeft(trackWindowLeft).copyTo(tempBPL(trackWindowLeft)); 
			depthMask(trackWindowLeft).copyTo(tempDepthMask(trackWindowLeft)); 		
		}
		bool result = isCvMatEmpty(tempDepthMask);

		/*imshow("depthmask",depthMask);*/
		//imshow("tempDepthMask Left",tempDepthMask);
		if(!result){
			//imshow("DepthMask",depthMask);
			backprojLeft = tempBPL & tempDepthMask;
		} else {
			backprojLeft = tempBPL ;
		}
		vector<vector<Point>> smallCountersPointsTest1;
			Mat depthMaskForRGTest1;
			depthMaskForRGTest1 = Mat::zeros(backprojLeft.size(),backprojLeft.type());
			backprojLeft(trackWindowLeft ).copyTo(depthMaskForRGTest1(trackWindowLeft ));
			findContours(depthMaskForRGTest1,smallCountersPointsTest1,0,1);
			
			//find biggest contour index.
			int iTest = findBiggestContour(smallCountersPointsTest1);
			//	 cout<<"iTest : findContours : "<<iTest<<endl;
			Point center1; //very very imp
			if(iTest != -1){
				drawContours(depthMaskForRGTest1,smallCountersPointsTest1,iTest,Scalar(255,255,255),3);
				
				/*vector<Point> contourTest = smallCountersPointsTest1[iTest];
				Scalar centerTemp1 = mean(Mat(contourTest));
				center1 = Point(centerTemp1.val[0], centerTemp1.val[1]);*/
			}

			//imshow("depthMaskForRGTest1",depthMaskForRGTest1);
	}

	return returnFlag;
  
}//EOM

bool MainMethodClass::createDepthMaskBeforeCamShiftRight(){
	
	bool returnFlag;
	//it is the depth image thresholding mask based on meadian depth..
	cvSet(tempImageRight,Scalar(0));

	//do operation for Right hand... 
	if(trackWindowRight.height > 0 && trackWindowRight.width > 0 &&
		trackWindowRight.tl().x >= 0 && trackWindowRight.tl().x >= 0){
		
		//depthDistanceRight is a global variable which can be utilized in a good smart way..
		returnFlag = findMeadianOftheTrakWindowPixelsInDepthImageRight(depthMatPyrDown,depthDistanceRight);
		//cout<<"distance Right meadian" <<depthDistanceRight<<endl; 
		thresholdIplImageBasedOnDepthValue(depthIplImagePyrDown,depthDistanceRight,tempImageRight);
		cvErode(tempImageRight,tempImageRight,0,1);
		cvDilate(tempImageRight,tempImageRight,0,1);
		 
		// for Right hand
		Mat depthMask = cv::cvarrToMat(tempImageRight);
	 
		Rect modifiedtrackWindowRight;
		expandTrackWindow(trackWindowRight,modifiedtrackWindowRight,10,depthMask.cols,depthMask.rows);


		Mat tempBPR = Mat::zeros(backprojRight.size(),backprojRight.type());
		Mat tempDepthMask = Mat::zeros(depthMask.size(),depthMask.type());

		//in order to get rid of possibility of getting -ve coordinates with extension of the window
		if(modifiedtrackWindowRight.tl().x >= 0 && modifiedtrackWindowRight.tl().x >= 0){
			backprojRight(modifiedtrackWindowRight).copyTo(tempBPR(modifiedtrackWindowRight)); 
			depthMask(modifiedtrackWindowRight).copyTo(tempDepthMask(modifiedtrackWindowRight)); 		 
		} else {
			backprojRight(trackWindowRight).copyTo(tempBPR(trackWindowRight)); 
			depthMask(trackWindowRight).copyTo(tempDepthMask(trackWindowRight)); 		 
		}
		
		
		bool result = isCvMatEmpty(tempDepthMask);

		/*imshow("depthmask",depthMask);*/
		//imshow("tempDepthMaskRight",tempDepthMask);
		if(!result){
			//imshow("DepthMask",depthMask);
			backprojRight = tempBPR & tempDepthMask;
		} else {
			backprojRight = tempBPR ;
		} 
	
	 
	}
  
	return returnFlag;
}//EOM

void MainMethodClass::understandGesture(vector<Point3i>& originalTrajectory,vector<Point3i>& srcVector5x5Scattered){
	//copy trajectory to GestureReco class
						 
	Rect returnRectAroundGesture;
	vector<int> angleReturnedVec;
	vector<int> returnReducedVector;
	Rect returnRectAroundGestur;	 
	
						 
	 gestureFlag = gestureRecognizerModified.understandTheGesture(originalTrajectory,srcVector5x5Scattered,colorMatOriginalSize,
		returnRectAroundGestur,indexOfGestures,gestureText);
						 
	//putText(colorMatPyrDown,gestureText,Point(10,10),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,Scalar(255,255,255),0.1,1);

	

	//rectangle(colorMatPyrDown,returnRectAroundGesture,Scalar(255,255,0),1);

	Rect rectLarge(returnRectAroundGesture.tl().x*4,returnRectAroundGesture.tl().y*4,returnRectAroundGesture.width*4,returnRectAroundGesture.height*4);
	rectangle(colorMatOriginalSize,rectLarge,Scalar(200,200,200),3);
						 
}

//Para: input srcIplImage, a mat with cropped diementions eg, Mat(limitwindow) is already cropped before calling this method..
//Returns : bool == true if srcMat is empty 
bool MainMethodClass::isCvMatEmpty(Mat& srcMat){
	 
	int noOfNonZeroPixels = 0;
	int rowcnt = srcMat.rows;
	int colcnt = srcMat.cols;
	for(int i = 0 ; i < rowcnt; i++){
		for(int j = 0 ; j < colcnt; j++){
		
			//if the first pixel it encounters is not 0, then return false, hence the matrix is not empty
			if(srcMat.at<uchar>(i,j) == 255){
				noOfNonZeroPixels++;
				if(noOfNonZeroPixels > 20)
					return false;
			}
		}
	}

		//if mat is full of 0 then return true, ie matrix is empty
		return true;
}
 

// TODO: Scope of improvement, try to pass one of color mask image too, and do checking 
//if the color pixel is skin pixel then and then only include it fromm depth image to calculate final depth in median vector
// but for this we need depth + color total synchronization which is not the case right now. may be in kinect2 we can do that...
bool MainMethodClass::findMeadianOftheTrakWindowPixelsInDepthImageLeft(cv::Mat& depthImage,short& distance){
	// initialize it to zero, so for next frame it does not take the same value as previous frame.

	short tempDistance = 0;
	 
	cv::Mat tempDepthWindowMat;
	depthImage(trackWindowLeft).copyTo(tempDepthWindowMat);

	long sumOfPixels = 0;
	int noOfPixels = 0;
	 
	Mat colorMask;
	backprojLeft(trackWindowLeft).copyTo(colorMask);

	vector<short> depthValVector;
	unsigned short value;
	//copy all pixels in a vector for further processing
	for(int i = 0; i < tempDepthWindowMat.rows; i++){
		for(int j = 0; j < tempDepthWindowMat.cols; j++){
			value = tempDepthWindowMat.at<unsigned short>(i,j);
			if(value != 0 && value > 500 && colorMask.at<uchar>(i,j) != 0){
				// if its direct depth image then use below
				depthValVector.push_back(value);
			}
			// if its converted image of depthMat then use below
			//depthValVector.push_back(tempDepthWindowMat.at<uchar>(i,j));
			  
		}
	}

	 
	size_t size = depthValVector.size();

	if(size > 0){
		sort(depthValVector.begin(), depthValVector.end());

		if (size  % 2 == 0)
			{
				tempDistance = (depthValVector[size / 2 - 1] + depthValVector[size / 2]) / 2;
			}
		else 
		{
			tempDistance = depthValVector[size / 2];
		}
	}

	if(tempDistance <= 500){
			return true;
	}

	//Most important piece of code, where you are not allowed to move further than +/-100 mm..
	//Here, this can be seen as depth tracking, where the current depth has to be in range with 
	// previous depth..
	if(depthDistanceLeft != 0){
		if(abs(depthDistanceLeft - tempDistance) < 100){
			depthDistanceLeft = tempDistance;
		} else {
			 
			rectangle(colorMatPyrDown,trackWindowFromDepthLeft,Scalar(0,255,255),3);
			putText(colorMatOriginalSize,"L-Dpt-Median",Point(50,50),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),1,1);
			
			Rect rectLarge(trackWindowFromDepthLeft.tl().x*4,trackWindowFromDepthLeft.tl().y*4,trackWindowFromDepthLeft.width*4,trackWindowFromDepthLeft.height*4);
			rectangle(colorMatOriginalSize,rectLarge,Scalar(0,255,255),3);

			
			
			//cout<<" not in range Left Depth Meadian ..........................."<<endl;
			return true;	//imp for hand left median distance range 
	
		}
	} else {
		depthDistanceLeft = tempDistance;
	}
	 
	//cout<<"distance Left : "<<depthDistanceLeft<<endl;
	return false; //false means the depth is in range..
}
 
 // TODO: Scope of improvement, try to pass one of color mask image too, and do checking 
//if the color pixel is skin pixel then and then only include it fromm depth image to calculate final depth in median vector
// but for this we need depth + color total synchronization which is not the case right now. may be in kinect2 we can do that...
bool MainMethodClass::findMeadianOftheTrakWindowPixelsInDepthImageRight(cv::Mat& depthImage,short& distance){
	// initialize it to zero, so for next frame it does not take the same value as previous frame.

	short tempDistance = 0;
	 
	cv::Mat tempDepthWindowMat;
	depthImage(trackWindowRight).copyTo(tempDepthWindowMat);

	long sumOfPixels = 0;
	int noOfPixels = 0;
	 
	Mat colorMask;
	backprojRight(trackWindowRight).copyTo(colorMask);

	vector<short> depthValVector;
	unsigned short value;
	//copy all pixels in a vector for further processing
	for(int i = 0; i < tempDepthWindowMat.rows; i++){
		for(int j = 0; j < tempDepthWindowMat.cols; j++){
			value = tempDepthWindowMat.at<unsigned short>(i,j);
			if(value != 0 && value > 500 && colorMask.at<uchar>(i,j) != 0){
				// if its direct depth image then use below
				depthValVector.push_back(value);
			}
			 
		}
	}

	 
	size_t size = depthValVector.size();

	if(size > 0){
		sort(depthValVector.begin(), depthValVector.end());

		if (size  % 2 == 0)
			{
				tempDistance = (depthValVector[size / 2 - 1] + depthValVector[size / 2]) / 2;
			}
		else 
		{
			tempDistance = depthValVector[size / 2];
		}
	}

	if(tempDistance <= 500){
			return true;
	}

	//Most important piece of code, where you are not allowed to move further than +/-100 mm..
	//Here, this can be seen as depth tracking, where the current depth has to be in range with 
	// previous depth..
	if(depthDistanceRight != 0){
		if(abs(depthDistanceRight - tempDistance) < 100){
			depthDistanceRight = tempDistance;
		} else {
			rectangle(colorMatPyrDown,trackWindowFromDepthRight,Scalar(0,255,255),3);
			putText(colorMatOriginalSize,"R-Dpt-Median",Point(50,100),CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,1,Scalar(255,255,255),1,1);
			Rect rectLarge(trackWindowFromDepthRight.tl().x*4,trackWindowFromDepthRight.tl().y*4,trackWindowFromDepthRight.width*4,trackWindowFromDepthRight.height*4);
			rectangle(colorMatOriginalSize,rectLarge,Scalar(0,255,255),3);
			 
			//cout<<" not in range Right Depth Meadian ..........................."<<endl;
			return true;
		}
	} else {
		depthDistanceRight = tempDistance;
	}
	 
	//cout<<"distance Right : "<<depthDistanceRight<<endl;
	return false; //false means the depth is in range..
}

void MainMethodClass::preProcessingBeforeCamshiftRight(Rect& trackWindow){
	newBackGrngThrImage.create(trackWindow.size(),backprojRight.type());
	newBackGrngThrImage.setTo(Scalar(0));
	backprojRight(trackWindow).copyTo(newBackGrngThrImage);
	//backprojLeft(newTrackWindow).copyTo(newBackGrngThrImage(newTrackWindow));

				 
	dilate(newBackGrngThrImage,newBackGrngThrImage,Mat(),Point(-1,-1),1); 
	erode(newBackGrngThrImage,newBackGrngThrImage,Mat(),Point(-1,-1),1);
	//medianBlur(backprojThreshold,backprojThreshold,7);
	//blur( backprojThreshold, backprojThreshold, Size( 3, 3 ), Point(-1,-1) );
	GaussianBlur( newBackGrngThrImage, newBackGrngThrImage, Size( 3, 3 ), 0, 0 );
	threshold(newBackGrngThrImage,newBackGrngThrImage,30,255,THRESH_BINARY);
			 
				 
	processBackProjThrMat = Mat::zeros(backprojRight.rows,backprojRight.cols,backprojRight.type());
	newBackGrngThrImage.copyTo(processBackProjThrMat(trackWindow));
	//imshow("processBackProjThrMatRight",processBackProjThrMat);

}

void MainMethodClass::preProcessingBeforeCamshiftLeft(Rect& trackWindow){
				newBackGrngThrImage.create(trackWindow.size(),backprojLeft.type());
				newBackGrngThrImage.setTo(Scalar(0));
				backprojLeft(trackWindow).copyTo(newBackGrngThrImage);
				
				dilate(newBackGrngThrImage,newBackGrngThrImage,Mat(),Point(-1,-1),1); 
				erode(newBackGrngThrImage,newBackGrngThrImage,Mat(),Point(-1,-1),1);
				//medianBlur(backprojThreshold,backprojThreshold,7);
				//blur( backprojThreshold, backprojThreshold, Size( 3, 3 ), Point(-1,-1) );
				GaussianBlur( newBackGrngThrImage, newBackGrngThrImage, Size( 3, 3 ), 0, 0 );
				 
				threshold(newBackGrngThrImage,newBackGrngThrImage,30,255,THRESH_BINARY);
				 
				processBackProjThrMat = Mat::zeros(backprojLeft.rows,backprojLeft.cols,backprojLeft.type());
				newBackGrngThrImage.copyTo(processBackProjThrMat(trackWindow));
				 

}



void MainMethodClass::expandTrackWindow(Rect& inputTrackWindow, Rect& returnTrackWindow, int margin,int MaxColVal,int MaxRowVal){
	Point newBr = inputTrackWindow.br();
	newBr.x = newBr.x + margin;
	if(newBr.x > MaxColVal){
		newBr.x = MaxColVal;
	}
	newBr.y = newBr.y + margin;
	if(newBr.y > MaxRowVal){
		newBr.y = MaxRowVal;
	}

	Point newTl = inputTrackWindow.tl();
	newTl.x = newTl.x - margin;
	if(newTl.x < 0){
		newTl.x = 0;
	}

	newTl.y = newTl.y - margin;
	if(newTl.y < 0){
		newTl.y = 0;
	}

	 
	returnTrackWindow = Rect(newTl.x,newTl.y,newBr.x - newTl.x,newBr.y - newTl.y);

}

void MainMethodClass::findHand(cv::Mat& backprojThreshold, Point& returnPoint,bool& isHand,String flag){
	 
	pyrUp(backprojThreshold,backprojThreshold);
	 
	cv::Mat tempShowMat;

	//tempShowMat = backprojThreshold;
	backprojThreshold.copyTo(tempShowMat);
	 

	cvtColor(tempShowMat,tempShowMat,CV_GRAY2BGR);
	 
	//find countors
	std::vector< std::vector<Point> > contours;
    findContours(backprojThreshold, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	//for each countor
	if (contours.size() > 0) {
               
		//for (int i = 0; i < contours.size(); i++) {
		int i = findBiggestContour(contours);

                    vector<Point> contour = contours[i];
                    Mat contourMat = Mat(contour);
                    double cArea = contourArea(contourMat);

                    if(cArea > 700 ) // likely the hand
                    { 
                        Scalar centerTemp = mean(contourMat);
                        Point centerPoint = Point(centerTemp.val[0], centerTemp.val[1]);
						bRect=boundingRect(contourMat);	
						 
						 
						//if bounding rect has certain diementions..
						if(bRect.width < 180 && bRect.height < 220){
							// approximate the contour by a simple curve
							vector<Point> approxCurve;
							approxPolyDP(contourMat, approxCurve, 10, true);

							vector< vector<Point> > debugContourV;
							debugContourV.push_back(approxCurve);
							drawContours(tempShowMat, debugContourV, 0, COLOR_DARK_GREEN, 3);

							vector<int> hull;
							convexHull(Mat(approxCurve), hull, false, false);

							// draw the hull points
							/*for(int j = 0; j < hull.size(); j++)
							{
								int index = hull[j];
								circle(tempShowMat, approxCurve[index], 3, COLOR_YELLOW, 2);
							}*/

							// find convexity defects
							vector<ConvexityDefect> convexDefects;
							findConvexityDefects(approxCurve, hull, convexDefects);
							//printf("Number of defects Befor: %d.\n", (int) convexDefects.size());
							eleminateDefects(convexDefects);
							//printf("Number of defects After : %d.\n", (int) convexDefects.size());

							int fingerNum = 0;

							for (int zz = 0; zz < convexDefects.size(); zz++)
								{
									Point startPoint = convexDefects[zz].start;

									Point depthPoint = convexDefects[zz].depth_point;

									Point endPoint = convexDefects[zz].end;
								 
									//TODO: implement smart way to find fingure no..... and open palm and closed palm...

									////Custom heuristic based on some experiment, double check it before use
									if ((startPoint.y < bRect.br().y || depthPoint.y < bRect.br().y)  
										&& (startPoint.x < bRect.br().x || depthPoint.x < bRect.br().x) 
										&& (startPoint.y > bRect.tl().y || depthPoint.y > bRect.tl().y)  
										&& (startPoint.x > bRect.tl().x || depthPoint.x > bRect.tl().x))
									{
								
									circle(tempShowMat,startPoint,2,COLOR_RED,2);
									circle(tempShowMat,depthPoint,2,COLOR_YELLOW,2);
									circle(tempShowMat,endPoint,2,COLOR_RED,2);
									fingerNum++;
									}

									//currentFrame.Draw(startCircle, new Bgr(Color.Red), 2);
									//currentFrame.Draw(depthCircle, new Bgr(Color.Yellow), 5);
									//currentFrame.Draw(endCircle, new Bgr(Color.DarkBlue), 4);
								}
						
							/*for(int j = 0; j < convexDefects.size(); j++)
							{
								circle(tempShowMat, convexDefects[j].depth_point, 3, COLOR_BLUE, 2);

							}*/
                        
							// assemble point set of convex hull
							vector<Point> hullPoints;
							for(int k = 0; k < hull.size(); k++)
							{
								int curveIndex = hull[k];
								Point p = approxCurve[curveIndex];
								hullPoints.push_back(p);
							}

							// area of hull and curve
							double hullArea  = contourArea(Mat(hullPoints));
							double curveArea = contourArea(Mat(approxCurve));
							double handRatio = curveArea/hullArea;

							fingerNum++;
							if(fingerNum > 5){
								fingerNum = 5;
							}
						
						
						
						
							// hand is grasping
							if(handRatio > GRASPING_THRESH){
								circle(tempShowMat, centerPoint, 5, COLOR_LIGHT_GREEN, 5);
								fingerNum = 0; // if the countors suggests that it is closed palm, then no of fingers are 0.
							
							}
							else{
								circle(tempShowMat, centerPoint, 5, COLOR_RED, 5);
							}
							if(convexDefects.size() > 5 || convexDefects.size() <= 0){
								isHand = false;

							}else {
								isHand = true;
							}
							if(fingerNum > 0){
								isHand = true;
							}
							stringstream ss;
							ss << fingerNum;

							string str = ss.str();
							stringstream ss1;
							ss1<< isHand;
							str = str+" , Hand "+ flag +" = "+ss1.str();

							if(flag.compare("right") == 0){
								putText(tempShowMat,str,Point(10, 50),1, 1.2f,Scalar(200,200,200),2);
							} else{
								putText(tempShowMat,str,Point(10, 100),1, 1.2f,Scalar(200,200,200),2);
							}
							cout<<"finger number : "<<fingerNum<<endl;

							returnPoint = centerPoint;
						}
				}// bRect width and height if condition... 
                //} // contour conditional
            } // hands loop
    
	/*cv::Mat rightMat;
	cv::Mat leftMat;*/
	if(flag.compare("left") == 0){
		//tempShowMat.copyTo(leftMat);
		imshow("fingerMatLeft",tempShowMat);
		 
	}else if(flag.compare("right") == 0){
		imshow("fingerMatRight",tempShowMat);
	}
 
}
// Thanks to Jose Manuel Cabrera for part of this C++ wrapper function
void MainMethodClass::findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<ConvexityDefect>& convexDefects)
{
    if(hull.size() > 0 && contour.size() > 0)
    {
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect *defectArray = 0;

        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq),sizeof(CvPoint), strDefects );

        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
        for(int i = 0; i < (int)contour.size(); i++) {
            CvPoint cp = {contour[i].x,  contour[i].y};
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int) hull.size();
        //int hullK[count];
        int* hullK = (int*) malloc(count*sizeof(int));
        for(int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
        //printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y);

        //We store defects points in the convexDefects parameter.
        for(int i = 0; i<defects->total; i++){
            ConvexityDefect def;
            def.start       = Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end         = Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth       = defectArray[i].depth;
            convexDefects.push_back(def);
        }

    // release memory
    cvReleaseMemStorage(&contourStr);
    cvReleaseMemStorage(&strDefects);
    cvReleaseMemStorage(&storage);

    }
}
void MainMethodClass::thresholdIplImageBasedOnDepthValue(IplImage* srcMat,int depth,IplImage* tempImage){
	
	int depthThreshold = 30;
	
	for(int y=0; y<srcMat->height; y++){
		for(int x=0; x<srcMat->width; x++){
			if(abs(((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] -  depth) > depthThreshold){
				//((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] = 0;
			}else{
			
				((uchar*)(tempImage->imageData +
								tempImage->widthStep*y))[x] = 255;
			}

		}
	}
	//cvShowImage("tempImage",tempImage);
}


void MainMethodClass::thresholdCvMatBasedOnDepthValue(Mat& srcMat,Mat& tempImage){
	
	int depthThreshold = 30;
	int rowcnt = srcMat.rows;
	int colcnt = srcMat.cols;
	for(int i = 0 ; i < rowcnt; i++){
		for(int j = 0 ; j < colcnt; j++){
			 
			if(abs(srcMat.at<unsigned short>(i,j) -  depthDistanceLeft) > depthThreshold){
				//((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] = 0;
			}else{
				tempImage.at<uchar>(i,j) = 255;
			}

		}
	}
	//cvShowImage("tempImage",tempImage);
}

/*** Some Additional methods.........
	These methods are some what experimental
**/
void MainMethodClass::displayHistogram(Mat& srcHistMat,String hand){
	histimg = Scalar::all(0);
    int binW = histimg.cols / hsize;
    Mat buf(1, hsize, CV_8UC3);
    for( int i = 0; i < hsize; i++ )
		buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
        cvtColor(buf, buf, CV_HSV2BGR);
                        
        for( int i = 0; i < hsize; i++ )
		{
			int val = saturate_cast<int>(srcHistMat.at<float>(i)*histimg.rows/255);
			rectangle( histimg, Point(i*binW,histimg.rows),
				Point((i+1)*binW,histimg.rows - val),
					Scalar(buf.at<Vec3b>(i)), -1, 8 );
		}
		if(hand == "left"){
			imshow( "Histogram-Left", histimg );
			//imwrite("histimg"+str+".jpg",histimg);
		} else{
			imshow( "Histogram-Right", histimg );
		}
}


void MainMethodClass::myDrawContours(Mat& srcMat,HandDetection& hg){
	
	 
	drawContours(srcMat,hg.hullP,hg.cIdx,cv::Scalar(200,0,0),2, 8, vector<Vec4i>(), 0, Point());

	rectangle(srcMat,hg.bRect.tl(),hg.bRect.br(),Scalar(0,0,200));
	vector<Vec4i>::iterator d=hg.defects[hg.cIdx].begin();
	int fontFace = FONT_HERSHEY_PLAIN;
		
	while( d!=hg.defects[hg.cIdx].end() ) {
   	    Vec4i& v=(*d);
	    int startidx=v[0]; Point ptStart(hg.contours[hg.cIdx][startidx] );
   		int endidx=v[1]; Point ptEnd(hg.contours[hg.cIdx][endidx] );
  	    int faridx=v[2]; Point ptFar(hg.contours[hg.cIdx][faridx] );
	    float depth = v[3] / 256;
   /*	
		line( m->src, ptStart, ptFar, Scalar(0,255,0), 1 );
	    line( m->src, ptEnd, ptFar, Scalar(0,255,0), 1 );
   		circle( m->src, ptFar,   4, Scalar(0,255,0), 2 );
   		circle( m->src, ptEnd,   4, Scalar(0,0,255), 2 );
   		circle( m->src, ptStart,   4, Scalar(255,0,0), 2 );
*/
   		circle( srcMat, ptFar,   9, Scalar(0,205,0), 5 );	
	    d++;

   	 }
//	imwrite("./images/contour_defects_before_eliminate.jpg",result);

}

int MainMethodClass::findBiggestContour(vector<vector<Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

void MainMethodClass::checkHandInImage(cv::Mat srcMat, bool& isHand,HandDetection& hg){

	Mat aBw;
	 
	//dilate(newBackGrngThrImage,newBackGrngThrImage,Mat(),Point(-1,-1),1); 
	srcMat.copyTo(aBw);
	//dilate(aBw,aBw,Mat(),Point(-1,-1),2); 
	cvtColor(aBw,aBw,CV_GRAY2BGR);
	findContours(srcMat,hg.contours,CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	hg.initVectors(); 
	hg.cIdx=findBiggestContour(hg.contours);
	if(hg.cIdx!=-1){
//		approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
		hg.bRect=boundingRect(Mat(hg.contours[hg.cIdx]));		
		convexHull(Mat(hg.contours[hg.cIdx]),hg.hullP[hg.cIdx],false,true);
		convexHull(Mat(hg.contours[hg.cIdx]),hg.hullI[hg.cIdx],false,false);
		approxPolyDP( Mat(hg.hullP[hg.cIdx]), hg.hullP[hg.cIdx], 18, true );
		if(hg.contours[hg.cIdx].size()>3 ){
			convexityDefects(hg.contours[hg.cIdx],hg.hullI[hg.cIdx],hg.defects[hg.cIdx]);
			hg.eleminateDefects();
		}
		bool isHand= false;
		 
			hg.getFingerTips(aBw);
			hg.drawFingerTips(aBw);
			isHand=hg.detectIfHand();
			hg.printGestureInfo(aBw);
			myDrawContours(aBw,hg);
		 
	}
	imshow("img1",aBw);

}

void MainMethodClass::makeContours(cv::Mat& srcMat, HandDetection& hg){
	Mat aBw;
	//pyrUp(srcMat,srcMat);
	srcMat.copyTo(aBw);
	findContours(aBw,hg.contours,CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	hg.initVectors(); 
	hg.cIdx=findBiggestContour(hg.contours);
	if(hg.cIdx!=-1){
		approxPolyDP( Mat(hg.contours[hg.cIdx]), hg.contours[hg.cIdx], 11, true );
		hg.bRect=boundingRect(Mat(hg.contours[hg.cIdx]));		
		convexHull(Mat(hg.contours[hg.cIdx]),hg.hullP[hg.cIdx],false,true);
		convexHull(Mat(hg.contours[hg.cIdx]),hg.hullI[hg.cIdx],false,false);
		approxPolyDP( Mat(hg.hullP[hg.cIdx]), hg.hullP[hg.cIdx], 18, true );
		if(hg.contours[hg.cIdx].size()>3 ){
			convexityDefects(hg.contours[hg.cIdx],hg.hullI[hg.cIdx],hg.defects[hg.cIdx]);
			hg.eleminateDefects();
		}
		bool isHand=hg.detectIfHand();
		hg.printGestureInfo(aBw);
		if(isHand){	
			hg.getFingerTips(aBw);
			hg.drawFingerTips(aBw);
			myDrawContours(aBw,hg);
		}
	}

	hg.getFingerNumber(aBw);
	imshow("img1",aBw);
}
           
void MainMethodClass::eleminateDefects(vector<ConvexityDefect>& convexDefects){
	int tolerance =  bRect_height/5;
	float angleTol=95;
	vector<ConvexityDefect> newDefects;
	int startidx, endidx, faridx;
	//vector<ConvexityDefect>::iterator d=convexDefects.begin();
	for(int i = 0 ; i < convexDefects.size() ; i++){
	
		 
   			//ConvexityDefect& v= d;
			Point ptStart(convexDefects[i].start );
			Point ptEnd(convexDefects[i].end );
			Point ptFar(convexDefects[i].depth_point );
			if(distanceP2P(ptStart, ptFar) > tolerance  &&  distanceP2P(ptEnd, ptFar) > tolerance &&  getAngle(ptStart, ptFar, ptEnd  ) < angleTol ){
				if( ptEnd.y > (bRect.y + bRect.height -bRect.height/4 ) ){
				}else if( ptStart.y > (bRect.y + bRect.height -bRect.height/4 ) ){
				}else {
					newDefects.push_back(convexDefects[i]);	
				}
			}	
			 
	}
	 
	//nrOfDefects=newDefects.size();
	convexDefects.swap(newDefects);
	//removeRedundantEndPoints(convexDefects,contour);
}
 
float MainMethodClass::distanceP2P(Point a, Point b){
	float d= sqrt(fabs((double) pow((double)(a.x-b.x),2) + (double)pow((double)(a.y-b.y),2) )) ;  
	return d;
}

float MainMethodClass::getAngle(Point s, Point f, Point e){
	float l1 = distanceP2P(f,s);
	float l2 = distanceP2P(f,e);
	float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
	float angle = acos(dot/(l1*l2));
	angle=angle*180/PI;
	return angle;
}

/** This method counts the pixel in the window and 
 *  if they are smaller than some range,
 *  It will change the global hue and sat thresholds for better performance **/
void MainMethodClass::processTheImage(cv::Mat& backprojThreshold){
if( trackWindowLeft.height <= 50 && trackWindowLeft.width <= 50) 
                {
					cv::Mat tempMat;
					backprojThreshold(trackWindowLeft).copyTo(tempMat);
					
					// if no of non zero pixels are less than equal to 200(MINNOOFPIXELS), 
					// set globalHueMin and Max thrshlds to full range;
					if(countNoOfPixelsInTrackWindow(tempMat) <= MINNOOFPIXELS ){
						cout<<"WINDOW HAS LESS THAN 200 PIXELS"<<endl;
						staticHueMax1 = 180;
						satMin = 20;
					} 
				} else if ( trackWindowLeft.height >= 200 || trackWindowLeft.width >= 200)  {
						//VERY IMP: make sure, if the tacjwindow size is tto big then reinitialize all the variables and make the window size smaller
						staticHueMax1 = 11;	
						satMin = 58;
						dynamicHueMin1 = 2;
						dynamicHueMax1 = 15;
						dynamicHueMin2 = 145;
						dynamicHueMax2 = 180;

						int X,Y;
						X = abs(trackWindowLeft.br().x - trackWindowLeft.tl().x)/2;
						Y = abs(trackWindowLeft.br().y - trackWindowLeft.tl().y)/2;	
						Point SeedPoint;
						SeedPoint.x = X + trackWindowLeft.tl().x;
						SeedPoint.y = Y + trackWindowLeft.tl().y;

						//this condition checks that if window size is bigger than 
						//limit then make it smaller with its center point
						if(SeedPoint.x > 10 && SeedPoint.y > 10){
							trackWindowLeft = Rect(SeedPoint.x -10, SeedPoint.y -10 , 20 , 20 );
						}
						

				} else {
						//VERY IMP: make sure, if above condition is not true set them to normal
						staticHueMax1 = 11;	
						satMin = 58;
				}
}

//this method will count no of non zeros pixels inside rect of given srcMat image
int MainMethodClass::countNoOfPixelsInTrackWindow(cv::Mat& srcMat){
	
	int noOfPixelsRet = 0;

	for(int i = 0; i < srcMat.rows; i++)
		for(int j = 0; j < srcMat.cols; j++){
		
			if((int)srcMat.at<uchar>(i,j) != 0){
			
				noOfPixelsRet++;

				// break the for loop to save computation time
				if(noOfPixelsRet > MINNOOFPIXELS){
					break;
				}
			}
			
		}

		return noOfPixelsRet;

}


void MainMethodClass::HistThresholding(cv::Mat& srcHistMat,int& minRange1, int& maxRange1,int& minRange2, int& maxRange2){

	// some constants, hist height is 255, one bean size is 16, total no of beans are 16
	
	//TODO: find all first two hihest peaks 
	// call method to do so
	float index1,index2;
	//index1 is first biggest and index2 i second biggest
	findFirstTwoBiggestPeakInHist(srcHistMat,index1,index2);

	 

	//TODO: for each peak, set range(lower & upper)
	// bean size is 16
	int firstLowerIndex = index1*11.25 + 0 ;
	int firstHigherIndex = index1*11.25 + 11.25 ;


	int secondLowerIndex = index2*11.25 + 0 ;
	int secondHigherIndex = index2*11.25 + 11.25;

	 
	if(firstLowerIndex < secondLowerIndex){
		minRange1 = firstLowerIndex;
		minRange2 = secondLowerIndex;
	} else{
		minRange1 = secondLowerIndex;
		minRange2 = firstLowerIndex;
	}

	if(firstHigherIndex < secondHigherIndex){
		maxRange1 = firstHigherIndex;
		maxRange2 = secondHigherIndex;
	} else{
		maxRange1 = secondHigherIndex;
		maxRange2 = firstHigherIndex;
	}


	//to avoid white color
	if(minRange1 == 0){
		minRange1  = 2;
	}

	// to avoid yellowish colr
	/*if(maxRange1 >= 12){
		maxRange1  = 12;
	}*/

	//to avoid red color
	if(maxRange2 >= 180){
		maxRange2  = 175;
	}
	//return two lower and upper ranges..

}


// this method will return two biggest indexes in the hist mat
void MainMethodClass::findFirstTwoBiggestPeakInHist(Mat& srcHistMat, float& index1, float& index2 ){
	
	vector<float> max,second_max,third_max;
	if(srcHistMat.at<float>(0) > srcHistMat.at<float>(1)) {
		
		// this would be the value which is the second biggest max
		second_max.push_back(srcHistMat.at<float>(1));
		
		// this would be the index which contains the second biggest max
		second_max.push_back(1);

		// this would be the value which is the biggest max
		max.push_back(srcHistMat.at<float>(0));
		
		// this would be the index which contains the biggest max
		max.push_back(0);
		 
	} else {
		second_max.push_back(srcHistMat.at<float>(0));
		second_max.push_back(0);

		max.push_back(srcHistMat.at<float>(1));
		max.push_back(1);
		
	}

	for(int i = 2; i< srcHistMat.rows; i++){
	
		// use >= n not just > as max and second_max can hav same value. Ex:{1,2,3,3}   
		if(srcHistMat.at<float>(i) >= max[0]){  
			//copy value
			second_max[0] = max[0];
			//copy index
			second_max[1] = max[1];

			//vopy value
			max[0] = srcHistMat.at<float>(i);          
			//copy index
			max[1] = i;          
		}
		else if(srcHistMat.at<float>(i) > second_max[0]){
			
			second_max[0] = srcHistMat.at<float>(i);
			second_max[1] = i;
		}
	}

	index1 = max[1];
	index2 = second_max[1];

	 
}

//this method does manual thresholding bze cv::threshold does not work with CV_16UC1 type..
void MainMethodClass::manualThreshold(Mat& srcMat, int thresholdVal,Mat& returnNearMask,Mat& returnFarMask){

 
	returnNearMask = Mat::zeros(srcMat.rows, srcMat.cols, CV_8UC1);
	  
	returnFarMask = Mat::zeros(srcMat.rows, srcMat.cols, CV_8UC1);

	int nearLimit = thresholdVal - 20;
	int farLimit = thresholdVal + 20;
	for(int i = 0; i < srcMat.rows; i++){
	
		for(int j = 0; j < srcMat.cols; j++){

			if(abs(nearLimit - (int)srcMat.at<unsigned short>(i,j)) < 20){
			
				returnNearMask.at<uchar>(i,j) = 255 ;
			} else if(abs(farLimit - (int)srcMat.at<unsigned short>(i,j)) < 20){
					returnFarMask.at<uchar>(i,j) = 255 ;
			}
		}
	}
	 
}


//this method does manual thresholding bze cv::threshold does not work with CV_16UC1 type..
void MainMethodClass::manualThreshold(Mat& srcMat, int thresholdVal,Mat& dstMat){

   
	int rows = srcMat.rows;
	int cols = srcMat.cols;
	for(int i = 0; i < rows; i++){
	
		for(int j = 0; j < cols; j++){

			if(srcMat.at<unsigned short>(i,j) > 20){
			
				dstMat.at<uchar>(i,j) = 255 ;
			}  
		}
	}
	 
}

void MainMethodClass::averageDepthImageValue(Mat srcMat,Rect roi, float& avgDepthValue){

	// initialize it to zero, so for next frame it does not take the same value as previous frame.
	avgDepthValue = 0;
	cv::Mat tempDepthWindowMat;
	srcMat(roi).copyTo(tempDepthWindowMat);

	long sumOfPixels = 0;
	int noOfPixels = 0;
	 
	for(int i = 0; i < tempDepthWindowMat.rows; i++){
		for(int j = 0; j < tempDepthWindowMat.cols; j++){
		
			// ig its direct depth image then use below
			long tempPixelValue = (long)tempDepthWindowMat.at<unsigned short>(i,j);

			// if its converted image of depthMat then use below
			//long tempPixelValue = (long)tempDepthWindowMat.at<uchar>(i,j);
			if(tempPixelValue != 0){
				sumOfPixels = sumOfPixels + tempPixelValue ;
				noOfPixels++;
			}	
		}
	}

	if(noOfPixels != 0){
		avgDepthValue = sumOfPixels/noOfPixels;
	}
}

class WatershedSegmenter{
private:
    cv::Mat markers;
public:
    void setMarkers(cv::Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};


void MainMethodClass::mainWaterShade(Mat& srcMat,Mat& srcColorMat)
{
     
    cv::Mat binary;// = cv::imread(argv[2], 0);
    //cv::cvtColor(srcMat, binary, CV_BGR2GRAY);
	//medianBlur(srcMat,srcMat,3);
	srcMat.copyTo(binary);
	cv::threshold(binary, binary, 100, 255, THRESH_BINARY);
	 
	medianBlur(binary,binary,(2*medianBlurCnt+1));
	imshow("srcMat", binary);
    // Eliminate noise and smaller objects
    cv::Mat fg;
    cv::erode(binary,fg,cv::Mat(),cv::Point(-1,-1),2);
    imshow("fg", fg);

    // Identify image pixels without objects
    cv::Mat bg;
    cv::dilate(binary,bg,cv::Mat(),cv::Point(-1,-1),2);
    cv::threshold(bg,bg,1, 128,cv::THRESH_BINARY_INV);
    imshow("bg", bg);

    // Create markers image
    cv::Mat markers(binary.size(),CV_8U,cv::Scalar(0));
    markers= fg+bg;
    imshow("markers", markers);

    // Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);

    cv::Mat result = segmenter.process(srcColorMat);
    result.convertTo(result,CV_8U);
    imshow("final_result", result);	 
}