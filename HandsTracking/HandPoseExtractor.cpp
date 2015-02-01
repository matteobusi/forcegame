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
const int HandPoseExtractor::histROIsz = 16;
const float* HandPoseExtractor::ranges = hueRange; 
const Scalar HandPoseExtractor::minHSV = Scalar(0, 107, 66);
const Scalar HandPoseExtractor::maxHSV = Scalar(180, 256, 256);

HandPoseExtractor::HandPoseExtractor(const Mat& roi, const Rect& initialTW) 
{
    /* Setup for roi */
    Mat hsvROI;
    Mat hueROI;
    Mat maskROI;
    histROI = Mat();
    
    imshow("test", roi);
    
    /* Convert to HSV */
    cvtColor(roi, hsvROI, COLOR_BGR2HSV);
    inRange(hsvROI, minHSV, maxHSV, maskROI);
    
    /* Extract Hue channel */
    hueROI.create(hsvROI.size(), hsvROI.depth());
    int ch[] = {0, 0};
    mixChannels(&hsvROI, 1, &hueROI, 1, ch, 1); 
    
    /* Calculate the histogram :) */ 
    calcHist(&hueROI, 1, 0, maskROI, histROI, 1, &histROIsz, &ranges);
    normalize(histROI, histROI, 0, 255, CV_MINMAX);
    
    trackWindow = Rect(initialTW);
    origTrackWindow = Rect(initialTW);
    
    /* Setup everything for haar-like feature detection */
    palmClassifier = CascadeClassifier();
    if(!palmClassifier.load(palmDetectorCascade))
    {
        cerr << "Error: cannot find haar-cascade file." << endl;
        exit(-1);    
    }
}

RotatedRect HandPoseExtractor::getROICurrent(const Mat& frame)
{
    Mat hsv;
    Mat hue;
    Mat backproj;
    Mat mask;
    
    /* calculate hsv for the frame */
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, minHSV, maxHSV, mask);
    
    /* Extract hue :) */
    hue.create(hsv.size(), hsv.depth());
    int ch[] = {0, 0};
    mixChannels(&hue, 1, &hue, 1, ch, 1);
    calcBackProject(&hue, 1, 0, histROI, backproj, &ranges); 
    backproj &= mask; 
    
    RotatedRect currBox = CamShift(backproj, trackWindow, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 3 ));
    
    if(trackWindow.area() <= 1)n
        trackWindow = origTrackWindow;
    
    Rect br = getBoundingBox(frame, currBox);
    
    Mat currentROI = Mat(frame, br);
    cvtColor(currentROI, currentROI, COLOR_BGR2GRAY);
    adaptiveThreshold(currentROI, currentROI, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,11,2);

    
    imshow("thresh", currentROI);
    
    return currBox;
}

Rect HandPoseExtractor::getBoundingBox(const Mat& frame, const RotatedRect& rect)
{
    Rect br = rect.boundingRect();
    
    br.x = MAX(0, br.x - 10);
    br.y = MAX(0, br.y - 10);
    
    br.height = MIN(frame.rows - br.y, br.height + 10);
    br.width = MIN(frame.cols - br.x, br.width + 10);
    
    br.height = MAX(0, br.height);
    br.width = MAX(0, br.width);
    
    return br;
}

vector<Rect> HandPoseExtractor::getHandPositionHaar(const Mat& frame)
{
    vector<Rect> palms;
    Mat grayFrame;
    
    imshow("HAAR", frame);
    
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    equalizeHist(grayFrame, grayFrame);
    
    palmClassifier.detectMultiScale(grayFrame, palms);
    
    return palms;
}

HandPoseExtractor::~HandPoseExtractor() 
{

}

