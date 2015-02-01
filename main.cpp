/* 
 * File:   main.cpp
 * Author: caos
 *
 * Created on January 29, 2015, 5:41 PM
 */

#include <iostream>
#include <opencv2/opencv.hpp>

#include "HandsTracking/HandPoseExtractor.h"


#define CAL_WND "calibration"
#define ROI_WND "roi"
#define OUT_WND "output"

#define ESC_KEY 27
#define SPACE_KEY 32

using namespace std;
using namespace cv;

/* Define ROI initial points */
const Point p1ROI(90, 120);
const Point p2ROI(200, 300);

const int camera = 1;

int main(int argc, char** argv) 
{
    VideoCapture capture(camera);
    
    char key = '\0';
    Mat frame, roi;
    
    if(!capture.isOpened())
    {
        cerr << "Cannot open the cam" << endl;
        exit(-1);            
    }
    
    namedWindow(CAL_WND);
    moveWindow(CAL_WND, 450, 50);
    
    /* Let's take the sample */
    while(key != ESC_KEY && key != SPACE_KEY)
    {
        bool successCam = capture.read(frame);
        // flip me :)
        flip(frame, frame, 1);
        
        Mat bak = frame.clone();
                
        if(!successCam)
        {
            cerr << "Error: cannot read webcam frame" << endl;
            exit(-1);
        }
        
        rectangle(bak, p1ROI, p2ROI, Scalar(0, 255, 255), 2);
        imshow(CAL_WND, bak);
        key = waitKey(1);
    }
    
    if(key==ESC_KEY)
        exit(0);
    
    
    roi = Mat(frame, Rect(p1ROI, p2ROI)).clone();
    HandPoseExtractor hand(roi, Rect(p1ROI, p2ROI)); 
                
    namedWindow(ROI_WND);
    moveWindow(ROI_WND, 10, 50);
    imshow(ROI_WND, roi);
   
    destroyWindow(CAL_WND);
    
    namedWindow(OUT_WND);
    moveWindow(OUT_WND, 450, 50);
            
    while(key != ESC_KEY)
    {
        bool successCam = capture.read(frame);
        // flip me :)
        flip(frame, frame, 1);
        
        if(!successCam)
        {
            cerr << "Error: cannot read webcam frame" << endl;
            exit(-1);
        }
        
        RotatedRect r = hand.getROICurrent(frame);
        Rect roiHandRect = hand.getBoundingBox(frame, r);        
        rectangle(frame, roiHandRect, Scalar(255, 255, 0), 2);  
        
        Point2f pts[4];
        r.points(pts);
        
        for (int i = 0; i < 4; i++)
            line(frame, pts[i], pts[(i+1)%4], Scalar(0,255,0));
               
        
        imshow(OUT_WND, frame);
        key = waitKey(10);
    }
    
    return 0;
}

