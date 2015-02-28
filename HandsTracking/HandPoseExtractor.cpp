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
    minHSV = Scalar(0, 48, 80);
    maxHSV = Scalar(20, 255, 255);
    
    cvtColor(roi, roiHSV, COLOR_BGR2HSV);

    inRange(roiHSV, minHSV, maxHSV, roiMask);
    calcHist(&roiHSV, 1, ch, roiMask, histROI, 2, histROIsz, ranges);
    
    normalize(histROI, histROI, 0, 255, NORM_MINMAX);

    trackWindow = Rect(oTrackWindow);
    origTrackWindow = Rect(oTrackWindow);
    
    /* We're using the filter to predict the hand size, position, velocity and angle - so state  */
    // x_k = F_k * x_(k-1) + w_k
    x = Mat_<float>(stateSize, 1) ; // State    
    
    // z_k = H_k * x_k + v_k
    z = Mat(measSize, 1, type); // Measures
    Mat H = Mat::zeros(measSize, stateSize, type); // Measurement matrix
    
    /* Initialize all structures */
    x.at<float>(0) = oTrackWindow.x + oTrackWindow.width/2;
    x.at<float>(1) = oTrackWindow.y + oTrackWindow.height/2;       
    x.at<float>(2) = 0;
    x.at<float>(3) = 0;
    x.at<float>(4) = oTrackWindow.width;
    x.at<float>(5) = oTrackWindow.height;
    x.at<float>(6) = 0;
    x.at<float>(7) = 0;
       
    H.at<float>(0, 0) = 1;
    H.at<float>(1, 1) = 1;
    H.at<float>(2, 4) = 1;
    H.at<float>(3, 5) = 1;
    H.at<float>(4, 6) = 1;
    
    kf = KalmanFilter(stateSize, measSize, contrSize);
    
    setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = H;
    
    setIdentity(kf.processNoiseCov, 1e-2);
    kf.processNoiseCov.at<float>(2,2) = 5;
    kf.processNoiseCov.at<float>(3,3) = 5;
    
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPre, Scalar::all(10));

    /* Initial status */
    kf.statePost = x;
    
    countNotFound = 0;
    found = false;
    
    /* Background subtractor */
    bgSub = new BackgroundSubtractorMOG2();
        
    namedWindow("BP");
    moveWindow("BP", 1700, 50);
    
    
    namedWindow("CH");
    moveWindow("CH", 2300, 50);    
}


RotatedRect HandPoseExtractor::getHandPosition(const Mat& frame)
{
    Mat hsv;
    Mat backproj;
    Mat mask;
    Mat fgMask;
    
    if(learningCount>=0)
    {
        learningCount--;
        bgSub->operator()(frame, fgMask, 0.0005f);
    }
    else
        bgSub->operator()(frame, fgMask, 0.0f);
    
    Mat bakFrame = frame.clone();
    Mat newFrame;
    
    bakFrame.copyTo(newFrame, fgMask);
    imshow("fg", newFrame);

    /* calculate hsv for the frame */
    cvtColor(newFrame, hsv, COLOR_BGR2HSV);
    inRange(hsv, minHSV, maxHSV, mask);
    
    calcBackProject(&hsv, 1, ch, histROI, backproj, ranges, 1, true); 
    
    /* Remove some noise - mainly background */
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(5, 5));
    
    /* close */
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);
    
    GaussianBlur(mask, mask, Size(3,3), 0);
    backproj &= mask; 
        
    float area;
    
    /* open */
    erode(backproj, backproj, kernel);
    dilate(backproj, backproj, kernel);
        
    GaussianBlur(backproj, backproj, Size(3,3), 0);
    
    RotatedRect currBox = CamShift(backproj, trackWindow, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    
    /* Find contours and pick the bigger. then find convex hull and fill it. */
    Rect bb = getBoundingBox(backproj, currBox);
    Mat backup = Mat(backproj, bb).clone();
    vector< vector<Point> > contours;
    findContours(backup, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    /* Pick the biggest */
    int contourNum = contours.size();
    float maxArea=0, maxI=-1;
    for(int i=0; i < contourNum; i++)
    {
        float tmpArea = contourArea(contours.at(i));
        if(tmpArea > maxArea)
        {
            maxI = i;
            maxArea = tmpArea;
        }
    }
    
    if(maxI >= 0) /* Found something */
    {
        vector<int> convexHull;
        vector<Vec4i> convDefects;
        cv::convexHull(Mat(contours[maxI]), convexHull);
        cv::convexityDefects(Mat(contours[maxI]), convexHull, convDefects);

        int numPts = contours[maxI].size();
        for(int i=0; i < numPts; i++)
        {
            contours[maxI][i].x += bb.x;
            contours[maxI][i].y += bb.y;
        }
        
        int convDefectSz = convDefects.size();
        for(int i=0; i < convDefectSz; i++)
        {
            Point s = contours[maxI][convDefects[i][0]];
            Point e = contours[maxI][convDefects[i][1]];
            Point d = contours[maxI][convDefects[i][2]];
            
            double dsd = dist(s,d);
            double ded = dist(e, d);
            double dse = dist(s, e);
            double dist = MIN(dsd, ded);
            double angle = acos((-dse*dse + dsd*dsd + ded*ded)/(2*dsd*ded));
            
            if(bb.height >= 2.5*dist && angle < 1.4)
            {
                circle(bakFrame, s, 2, Scalar(0, 255, 255), 2);
                circle(bakFrame, e, 2, Scalar(0, 255, 255), 2);
                circle(bakFrame, d, 2, Scalar(0, 255, 0), 2);
            }
        }
        //int conDefectsCount = 
        //drawContours(bakFrame, container, 0, Scalar(255), 3);
    }
    
    
    area = trackWindow.area();
    imshow("BP", backproj);
    imshow("CH", bakFrame);
    
    if(found)
    {
        double prevTicks = ticks;
        ticks = (double)getTickCount();
        double dt = (ticks-prevTicks)/getTickFrequency();
        kf.transitionMatrix.at<float>(0, 2) = dt;
        kf.transitionMatrix.at<float>(1, 3) = dt;
        kf.transitionMatrix.at<float>(6, 7) = dt;
        
        x = kf.predict();
    }
    
    if(area <= 1)
    {
        trackWindow = origTrackWindow;
        if(countNotFound>=10)
            found=false;
        else
            kf.statePost = x;
        countNotFound++;
    }
    else
    {    
        countNotFound = 0;
        
        if(!found)
        {            
            setIdentity(kf.errorCovPre, Scalar::all(10));
            x.at<float>(0) = currBox.center.x; 
            x.at<float>(1) = currBox.center.y;
            x.at<float>(2) = 0;
            x.at<float>(3) = 0;
            x.at<float>(4) = currBox.size.width;
            x.at<float>(5) = currBox.size.height;
            x.at<float>(6) = currBox.angle;
            x.at<float>(7) = 0;
            
            found = true;
        }
        else
        {
            z.at<float>(0) = currBox.center.x; 
            z.at<float>(1) = currBox.center.y;
            z.at<float>(2) = currBox.size.width;
            z.at<float>(3) = currBox.size.height;
            z.at<float>(4) = currBox.angle;
           
            kf.correct(z);
        }
        /* Enforce the estimation with Kalman filter */
        cout << "Prediction: " << x << endl;
        RotatedRect predictedBox;
        predictedBox.center = Point(x.at<float>(0), x.at<float>(1));
        predictedBox.size = Size2f(x.at<float>(4), x.at<float>(5));
        predictedBox.angle = x.at<float>(6);
        
        return predictedBox;
    }
   
    return RotatedRect();
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

double HandPoseExtractor::dist(cv::Point p1, cv::Point p2)
{
    double d1=p1.x - p2.x;
    double d2=p1.y - p2.y;
    return sqrt(d1*d1 + d2*d2);
}


HandPoseExtractor::~HandPoseExtractor() 
{
   // delete bgSub;
}

