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

class HandPoseExtractor {
public:
    HandPoseExtractor(const cv::Mat& roi, const cv::Rect& initialTW);
    HandPoseExtractor(); 
    
    cv::RotatedRect getROICurrent(const cv::Mat& frame);
    std::vector<cv::Rect> getHandPositionHaar(const cv::Mat& frame);
    
    cv::Rect getBoundingBox(const cv::Mat& frame, const cv::RotatedRect& rect);
    
    virtual ~HandPoseExtractor();
private:
    /* Constants */
    static const char palmDetectorCascade[];    
    static const int histROIsz;
    static const float hueRange[];
    static const float* ranges;
    static const cv::Scalar minHSV, maxHSV;

    cv::Rect trackWindow, origTrackWindow;
    cv::Mat histROI;
    cv::CascadeClassifier palmClassifier;
};

#endif	/* HANDPOSEEXTRACTOR_H */

