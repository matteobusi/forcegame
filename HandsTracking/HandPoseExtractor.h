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
    static const char palmDetectorCascade[];    
    static const int histROIsz[];
    static const float hueRange[], saturationRange[];
    static const float* ranges[]; 
    static const int ch[];
    
    cv::Scalar minHSV, maxHSV;  

    cv::Rect trackWindow, origTrackWindow;
    cv::Mat histROI;
};

#endif	/* HANDPOSEEXTRACTOR_H */

