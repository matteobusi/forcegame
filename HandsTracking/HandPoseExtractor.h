/* 
 * File:   HandPoseExtractor.h
 * Author: caos
 *
 * Created on January 29, 2015, 5:50 PM
 */

#ifndef HANDPOSEEXTRACTOR_H
#define	HANDPOSEEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <utility>
#include <string>

class HandPoseExtractor
{
public:
    HandPoseExtractor(const cv::Mat& frame, const cv::Rect& oTrackWindow);
    HandPoseExtractor(); 
    
    cv::RotatedRect getHandPosition(const cv::Mat& frame);
    cv::Rect getBoundingBox(const cv::Mat& frame, const cv::RotatedRect& rect);
    
    virtual ~HandPoseExtractor();
private:
    /* Constants */ 
    static const int histROIsz[];
    static const float hueRange[], saturationRange[];
    static const float* ranges[]; 
    static const int ch[];
    
    cv::Scalar minHSV, maxHSV;  

    cv::Rect trackWindow, origTrackWindow;
    cv::Mat histROI;
    
    
    static const int stateSize = 8;
    static const int measSize = 5;
    static const int contrSize = 0;
    static const int type = CV_32F;
    cv::Mat_<float> x; 
    cv::Mat_<float> z;
    
    cv::KalmanFilter kf;   
    
    int countNotFound;
    bool found;
    double ticks;
    
    int learningCount = 500;
    
    cv::Ptr<cv::BackgroundSubtractor> bgSub;
    
    double dist(cv::Point p1, cv::Point p2);
};

#endif	/* HANDPOSEEXTRACTOR_H */

