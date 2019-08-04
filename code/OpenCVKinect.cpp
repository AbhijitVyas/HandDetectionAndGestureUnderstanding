 
#include "OpenCVKinect.h"

OpenCVKinect::OpenCVKinect(void)
{
	m_depthTimeStamp = 0;
	m_colorTimeStamp = 0;

	m_alignedStreamStatus = false;
	m_colorStreamStatus = false;
	m_depthStreamStatus = false;
}

bool OpenCVKinect::setMode(int inMode)
{
	if((inMode&C_MODE_COLOR) == C_MODE_COLOR && (inMode&C_MODE_DEPTH) == C_MODE_DEPTH )
	{
		std::cout << "COLOR + DEPTH" << std::endl;
		m_colorStreamStatus = true;
		m_depthStreamStatus = true;

		if((inMode&C_MODE_ALIGNED) == C_MODE_ALIGNED)
		{
			std::cout << "+ ALIGNED" << std::endl;
			m_alignedStreamStatus = true;
		}
		return true;
	} 
	else
	{
		std::cout << "NOTHING" << std::endl;
		return false;

	}
	return false;
}

/**
* init method initializes kinect device url and sets color and depth image configuration modes
*
*/

bool OpenCVKinect::init()

{
	setMode(C_MODE_DEPTH | C_MODE_COLOR | C_MODE_ALIGNED);
	m_status = openni::STATUS_OK;
	//const char* deviceURI = "C:/Users/Abhijit/Desktop/OniFiles/circle.oni"; 
	const char* deviceURI = openni::ANY_DEVICE;	
	m_status = openni::OpenNI::initialize();

	std::cout << "After initialization: " << std::endl; 
	std::cout << openni::OpenNI::getExtendedError() << std::endl;

	// open the device
	m_status = m_device.open(deviceURI);
	if(m_status != openni::STATUS_OK)
	{
		std::cout << "OpenCVKinect: Device open failseed: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	if(m_depthStreamStatus)
	{
		// create a depth object
		m_status = m_depth.create(m_device, openni::SENSOR_DEPTH);
		if(m_status == openni::STATUS_OK)
		{
			m_status = m_depth.start();
			if(m_status != openni::STATUS_OK)
			{
				std::cout << "OpenCVKinect: Couldn't start depth stream: " << std::endl;
				std::cout << openni::OpenNI::getExtendedError() << std::endl;
				m_depth.destroy();
				return false;
			}
		}
		else
		{
			std::cout << "OpenCVKinect: Couldn't find depth stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			return false;
		}
	}

	if(m_colorStreamStatus)
	{
		// create a color object
		m_status = m_color.create(m_device, openni::SENSOR_COLOR);
		if(m_status == openni::STATUS_OK)
		{
			// if the status is okay then start reading color images
			m_status = m_color.start();
			if(m_status != openni::STATUS_OK)
			{
				std::cout << "OpenCVKinect: Couldn't start color stream: " << std::endl;
				std::cout << openni::OpenNI::getExtendedError() << std::endl;
				m_color.destroy();
				return false;
			}
		}
		else
		{
			std::cout << "OpenCVKinect: Couldn't find color stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			return false;
		}

	}

	if(m_alignedStreamStatus)
	{
		m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	}


	if(!m_depth.isValid() && !m_color.isValid())
	{
		std::cout << "OpenCVKinect: No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	this->m_streams = new openni::VideoStream*[C_NUM_STREAMS];
	m_streams[C_DEPTH_STREAM] = &m_depth;
	m_streams[C_COLOR_STREAM] = &m_color;
	return true;
}

/**
* getData method initilizes different color and depth martices based on the settings it gets from global variables
*/
cv::vector<cv::Mat> OpenCVKinect::getData()
{
	cv::vector<cv::Mat> returnVec;
	cv::Mat bufferImage;
	bool depthCaptured = false, colorCaptured = false;
	Mat xCoOrdinateMat,yCoOrdinateMat,zCoOrdinateMat;
	xCoOrdinateMat.create(480, 640, CV_16UC1);
	yCoOrdinateMat.create(480, 640, CV_16UC1);
	zCoOrdinateMat.create(480, 640, CV_16UC1);

	if(this->m_colorStreamStatus & this->m_depthStreamStatus)
	{
		while( !depthCaptured || !colorCaptured || m_depthTimeStamp != m_colorTimeStamp)
		{
			m_status = openni::OpenNI::waitForAnyStream(m_streams, C_NUM_STREAMS, &m_currentStream, C_STREAM_TIMEOUT);
			if(m_status != openni::STATUS_OK)
			{
				std::cout << "OpenCVKinect: Unable to wait for streams. Exiting" << std::endl;
				exit(EXIT_FAILURE);
			}

			switch(m_currentStream)
			{
			case C_DEPTH_STREAM:
				m_depth.readFrame(&m_depthFrame);
				m_depthImage.create(m_depthFrame.getHeight(), m_depthFrame.getWidth(), CV_16UC1);
				m_depthImage.data = (uchar*)m_depthFrame.getData();
				this->m_depthTimeStamp = m_depthFrame.getTimestamp() >> 16;
//				std::cout << "Depth Timestamp: " << this->m_depthTimeStamp << std::endl;
				depthCaptured = true;

				
				break;
			case C_COLOR_STREAM:
				m_color.readFrame(&m_colorFrame);
				m_colorImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
				bufferImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
				bufferImage.data = (uchar*)m_colorFrame.getData();
				this->m_colorTimeStamp = m_colorFrame.getTimestamp() >> 16;
//				std::cout << "Color Timestamp: " << m_colorTimeStamp << std::endl;
				colorCaptured = true;
				cv::cvtColor(bufferImage, m_colorImage, CV_BGR2RGB);
				break;
			default:
				break;
			}

		}
	} else {
		m_colorImage = cv::Mat::zeros(10, 10, CV_8UC1);
		m_depthImage = cv::Mat::zeros(10, 10, CV_8UC1);

	} 
	returnVec.push_back(m_depthImage);
	returnVec.push_back(m_colorImage);
	bufferImage.release();
	return returnVec;
}

/*
* For calculating world coordinates.	 
*/
void OpenCVKinect::calculateWorldCoOrdinates(float x,float y,float depth,float& xW,float& yW,float& zW){
	float px, py, pz;
	openni::CoordinateConverter::convertDepthToWorld(m_depth,x,y,depth,&px,&py,&pz);
	 
	xW = px;
	yW = py;
	zW = pz;
}

void OpenCVKinect::registerDepthAndImage()
{
	m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
}

cv::Mat OpenCVKinect::getColor()
{
	return m_colorImage;
}

cv::Mat OpenCVKinect::getDepth()
{
	return m_depthImage;
}

OpenCVKinect::~OpenCVKinect(void)
{
	this->m_depthFrame.release();
	this->m_colorFrame.release();
	this->m_depth.stop();
	this->m_color.stop();
	openni::OpenNI::shutdown();
	this->m_device.close();

	recorder.stop();
	recorder.destroy();
}