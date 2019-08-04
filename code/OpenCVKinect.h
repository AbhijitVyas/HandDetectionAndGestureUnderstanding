// *******************************************************************************
//	OpenCVKinect: Provides method to access Kinect Color and Depth Stream        *
//				  in OpenCV Mat format.                                           *
//                                                                                *
//				  Pre-requisites: OpenCV_2.x, OpenNI_2.x, KinectSDK_1.8           *
//                                                                                *
//   Copyright (C) 2013  Muhammad Asad                                            *
//                       Webpage: http://seevisionc.blogspot.co.uk/p/about.html   *
//						 Contact: masad.801@gmail.com                             *
//                                                                                *
//   This program is free software: you can redistribute it and/or modify         *
//   it under the terms of the GNU General Public License as published by         *
//   the Free Software Foundation, either version 3 of the License, or            *
//   (at your option) any later version.                                          *
//                                                                                *
//   This program is distributed in the hope that it will be useful,              *
//   but WITHOUT ANY WARRANTY; without even the implied warranty of               *
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
//   GNU General Public License for more details.                                 *
//                                                                                *
//   You should have received a copy of the GNU General Public License            *
//   along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
//                                                                                *
// *******************************************************************************

#pragma once

#include <OpenNI.h>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>
 
#define C_DEPTH_STREAM 0
#define C_COLOR_STREAM 1

#define C_NUM_STREAMS 2

#define C_MODE_COLOR 0x01
#define C_MODE_DEPTH 0x02
#define C_MODE_ALIGNED 0x04

#define C_STREAM_TIMEOUT 2000

using namespace cv;
using namespace std;
using namespace openni;

class OpenCVKinect
{
	openni::Status m_status;
	openni::Device m_device;
	openni::VideoStream m_depth, m_color, **m_streams;
	openni::Recorder recorder;
	 
	openni::VideoFrameRef m_depthFrame, m_colorFrame;
	int m_currentStream;
	uint64_t m_depthTimeStamp, m_colorTimeStamp;
	cv::Mat m_depthImage, m_colorImage;
	bool m_alignedStreamStatus, m_colorStreamStatus, m_depthStreamStatus;

	int threshNear;
	int threshFar;
	int dilateAmt;
	int erodeAmt;
	int blurAmt;
	int blurPre;
	vector<vector<Point> > contours;
	 
public:
	OpenCVKinect(void);
	bool setMode(int inMode);
	bool init();
	cv::vector<cv::Mat> getData();
	void registerDepthAndImage();
	void calculateWorldCoOrdinates(float x,float y,float depth,float& xW,float& yW,float& zW);
	cv::Mat getDepth();
	cv::Mat getColor();
	~OpenCVKinect(void);
	int selectedOption();
	    
};