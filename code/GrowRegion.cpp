#include "GrowRegion.hpp"

GrowRegion::GrowRegion(IplImage* srcImage){

	cloneLookUpDepthImageDifferently = cvCreateImage(cvSize(srcImage->width,srcImage->height), IPL_DEPTH_8U,1);
	cvSet(cloneLookUpDepthImageDifferently, cvScalar(0));
	borderImage = cvCreateImage(cvSize(srcImage->width,srcImage->height), srcImage->depth,srcImage->nChannels);
	cvSet(borderImage, cvScalar(0));
	
	srcMat = cvCreateImage(cvSize(srcImage->width,srcImage->height), srcImage->depth,srcImage->nChannels);
	cvSet(srcMat, cvScalar(0));

	//create storage for contours
    //storage = cvCreateMemStorage(0);
	ROWS = srcImage->height;
	COLS = srcImage->width;
	XMIN = COLS;
	XMAX = 0;
	YMIN = ROWS; 
	YMAX = 0;
	  
}

GrowRegion::~GrowRegion(){
	
	cvReleaseImage(&borderImage);
	cvReleaseImage(&cloneLookUpDepthImageDifferently);
	cvReleaseImage(&srcMat);
    
}



void GrowRegion::thresholdIplImageBasedOnDepthValue(IplImage* srcMat,unsigned short depth){
	
	
	for(int y=0; y<srcMat->height; y++){
		for(int x=0; x<srcMat->width; x++){
			if(abs(((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] -  depth) > 50){
				((unsigned short*)(srcMat->imageData + srcMat->widthStep*y))[x] = 0;
			}/*else{
			
				((uchar*)(tempImage->imageData +
								tempImage->widthStep*y))[x] = 255;
			}*/

		}
	}
	//cvShowImage("tempImage",tempImage);
}


//Para: srcMat is Ipl depth image for RG,
//Para: seedPoint is the starting point for RG algo.,
//Para: distance is the depth value at which Region is growing, so that hand size can be limited based on depth(not working)
//Return Para : returnPoint is the central point of the grown region which is kind of return value of the algo.
//Para: trackWindow is the window to limit region growth(not working)
//Para: flag is to indicate which hans is it, ie Left hand or Right hand.
void GrowRegion::run(IplImage* srcMatInput, Point SeedPointInput,int distance,Point& stopPointInput,vector<Point>& borderPoints,Mat& returnedBorderMat){

 
	//cout<<"run"<<endl;
	if(distance <= 0){
			return ;
	}
	
	//initialize global class variables.. very imp
	 globalCounter = 0;
	 startPoint = SeedPointInput;
	 stopPoint = stopPointInput;
	 srcMat = cvCloneImage(srcMatInput);
	  

	// for region growth limit
	Point FirstSP = SeedPointInput;
	vector<Point> seedPoints;
	seedPoints.push_back(SeedPointInput); 
	getNewSeedPoints(seedPoints);
  
	borderPoints = countor;
	  
}

void GrowRegion::getNewSeedPoints(vector<Point> SeedPoints){
	
	globalCounter ++;
	 
	//cout<<"getNewSeedPoints : "<<globalCounter<<endl;
	int depthThreshold = 50;
	// increase globalCounter, so that we can count how many times this method has been recursively called..
	 
	vector<Point> recursiveNeighbourPoints;
	//recursiveNeighbourPoints.clear();

	if(SeedPoints.size() != 0){
		for(int j = 0; j < SeedPoints.size(); j++){
			//find 4 neightbourhood pixels
			vector<Point> neighbourPoints;
			getNeighbourhoodPixels(SeedPoints[j],neighbourPoints);
		
			int neighbourhoodEdgePixelCounter = 0;
			for(int i = 0 ; i < neighbourPoints.size() ; i++){
				 
				// this condition makes sense and prevents an error in CV::MAT case, but it should not matter in IPLImage
				if(neighbourPoints[i].y < ROWS && neighbourPoints[i].y > 0 && neighbourPoints[i].x < COLS && neighbourPoints[i].x > 0){
					// check if these new neighbour pixels are already defined in clone loop-up image
					if(((uchar*)(cloneLookUpDepthImageDifferently->imageData +
							cloneLookUpDepthImageDifferently->widthStep*neighbourPoints[i].y))[neighbourPoints[i].x] != 255){
						
						// add this pt to selected pt, if it setisfies the condition..
						if(abs(((unsigned short*)(srcMat->imageData + srcMat->widthStep*SeedPoints[j].y))[SeedPoints[j].x]
						-  ((unsigned short*)(srcMat->imageData + srcMat->widthStep*neighbourPoints[i].y))[neighbourPoints[i].x]) < depthThreshold){
						
						
							((uchar*)(cloneLookUpDepthImageDifferently->imageData +
								cloneLookUpDepthImageDifferently->widthStep*neighbourPoints[i].y))[neighbourPoints[i].x] = 255;
								 
							recursiveNeighbourPoints.push_back(neighbourPoints[i]);
							//cout<<"inside if second condition"<<endl;
						} else {
						 // if the neighbour pixel does not satisfies the condition for neighbourhood then the current pixel is boundray pixel
							/*((uchar*)(borderImage->imageData +
								borderImage->widthStep*SeedPoints[j].y))[SeedPoints[j].x] = 255;*/
							neighbourhoodEdgePixelCounter++;
							 
						}
					} // if image look up is == 255
				} // if neighbourhood pixel is in the range of source matrix rows and cols wise..
	
			} //for each neighbourhood pixel
			
			if(neighbourhoodEdgePixelCounter != 0){
				/*((unsigned short*)(borderImage->imageData +
								borderImage->widthStep*SeedPoints[j].y))[SeedPoints[j].x] = 
								((unsigned short*)(srcMat->imageData + srcMat->widthStep*SeedPoints[j].y))[SeedPoints[j].x];*/
				countor.push_back(SeedPoints[j]);
			}
			if(stopPoint.x == SeedPoints[j].x && stopPoint.y == SeedPoints[j].y){
				return;
			}
		 
		}// for each recursiv seed point 

	//	cvShowImage("imagedepthwithregiongrow",cloneLookUpDepthImageDifferently);
	//	waitKey(0);
		getNewSeedPoints(recursiveNeighbourPoints);
	}
	 
}


void GrowRegion::DrawTextOnIPLImage(IplImage* srcImage,Point& position,String text){

	CvFont font;
	double hScale=1.0;
	double vScale=1.0;
	int    lineWidth=1;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
	  
	cvPutText (srcImage, text.c_str(),position, &font, cvScalar(255,255,255));

}

void GrowRegion::getNeighbourhoodPixels(Point SeedPoint, vector<Point>& neighbourPixels){

	// find the 4 - neighbourhood vector, 
		// check for each pixel that it does not get out of Image range(560x480)
		if(SeedPoint.x > 0 || SeedPoint.x < COLS || SeedPoint.y > 0 || SeedPoint.y < ROWS ){
			
			// 4 neighbourhood points
			/*neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y));	
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y-1));
			
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y));
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y+1));*/

			//skipping a point
			/*neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y));	
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y-1));
			
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y));
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y+1));*/
			

			// 8 neighbourhood points
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y-1));	
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y-1));
			
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y-1));
			
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y));
			
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y));
			
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y+1));
			
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y+1));
			
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y+1));

			// 16 extra pixels two layers...
			/*
			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y-2));
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y-2));
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y-2));
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y-2));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y-2));

			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y-1));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y-1));

			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y));

			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y+1));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y+1));

			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y+2));
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y+2));
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y+2));
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y+2));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y+2));
			/*
			// additional 25 points...
			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y-3));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y-3));

			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y-2));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y-2));

			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y-1));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y-1));
			 
			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y));

			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y+1));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y+1));

			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y+2));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y+2));

			neighbourPixels.push_back(Point(SeedPoint.x-3,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x-2,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x-1,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x+1,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x+2,SeedPoint.y+3));
			neighbourPixels.push_back(Point(SeedPoint.x+3,SeedPoint.y+3));
			*/
			
			//try to calculate area of the hand region and stop the recursive mthod..
			// for some iterarions dont need to count area..
			// count countors and then areas
		}

}

 
 