/* 
 * File:   HandPoseExtractor.cpp
 * Author: caos
 * 
 * Created on January 29, 2015, 5:50 PM
 */

#include "HandPoseExtractor.h"

using namespace cv;
using namespace std;

/* Constants */
const char HandPoseExtractor::palmDetectorCascade[] = "./res/palm.xml";
const float HandPoseExtractor::hueRange[] = {0, 180};
const float HandPoseExtractor::saturationRange[] = {0, 255};
const int HandPoseExtractor::histROIsz[] = { 16, 32 };
const float* HandPoseExtractor::ranges[] = { hueRange, saturationRange}; 
const int HandPoseExtractor::ch[] = {0, 1};

HandPoseExtractor::HandPoseExtractor(const Mat& frame, const Rect& oTrackWindow) 
{
    Mat roi(frame, oTrackWindow);
    Mat roiHSV;
    Mat roiMask;
    histROI = Mat();
   
    /* SKIN */
    minHSV = Scalar(0, 58, 0);
    maxHSV = Scalar(50, 174, 255);
    
    cvtColor(roi, roiHSV, COLOR_BGR2HSV);

    inRange(roiHSV, minHSV, maxHSV, roiMask);
    calcHist(&roiHSV, 1, ch, roiMask, histROI, 2, histROIsz, ranges);
    
    normalize(histROI, histROI, 0, 255, NORM_MINMAX);

    trackWindow = Rect(oTrackWindow);
    origTrackWindow = Rect(oTrackWindow);
}


RotatedRect HandPoseExtractor::getHandPosition(const Mat& frame)
{
    Mat hsv;
    Mat backproj;
    Mat mask;
    
    
    /* calculate hsv for the frame */
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, minHSV, maxHSV, mask);
    
    calcBackProject(&hsv, 1, ch, histROI, backproj, ranges, 1, true); 
    backproj &= mask; 
    
    /* Remove some noise - mainly background */
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3,3));
    
    /* open */
    erode(backproj, backproj, kernel);
    dilate(backproj, backproj, kernel);
    
    /* close */
    dilate(backproj, backproj, kernel);
    erode(backproj, backproj, kernel);
    
    /* threshold - filter out things with too low probability */
    int thresh = 0.5*255; 
    threshold(backproj, backproj, thresh, 255, THRESH_TOZERO);
    
    RotatedRect currBox = CamShift(backproj, trackWindow, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 15, 3 ));
    if(trackWindow.area() <= 1)
        trackWindow = origTrackWindow;
      
    Rect boundingBox = getBoundingBox(frame, currBox);    
    imshow("BP", backproj);
    moveWindow("BP", 900, 50);
    /* Enforce the estimation with Kalman filter */
    int stateSize = 7;
    int measSize = 5;
    int contrSize = 0;
    int type = CV_32F;
    
    Mat x(stateSize, 1, type) ; // State
    Mat z(measSize, 1, type); // Measures
    
    KalmanFilter kf(stateSize, measSize, contrSize);
    
    
    return currBox;
   
}

Rect HandPoseExtractor::getBoundingBox(const Mat& frame, const RotatedRect& rect)
{
    Rect br = rect.boundingRect();
    
    br.x = MAX(0, br.x - 50);
    br.y = MAX(0, br.y - 50);
    
    br.height = MIN(frame.rows - br.y, br.height + 100);
    br.width = MIN(frame.cols - br.x, br.width + 100);
    
    br.height = MAX(0, br.height);
    br.width = MAX(0, br.width);
    
    return br;
}

HandPoseExtractor::~HandPoseExtractor() 
{

}

