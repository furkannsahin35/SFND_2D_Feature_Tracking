/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"      
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char* argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 3;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results


    double detector_time = 0;
    double descriptor_time = 0;
    bool writedetectorsFlag = 0;
    bool writedescriptorsFlag = 0;

    string detectorType_t = "FAST";// SHITOMASI , HARRIS ,FAST BRISK, ORB, AKAZE, SIFT 
    string descriptorType_t = "BRIEF"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

    std::ofstream features_file;
    std::ofstream features_file2;

    if (writedetectorsFlag)
    {
        
        features_file.open("../detector_results/detectorType_SIFT.csv"); // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        features_file << "Detector Type " << "," << "Image Index" << "," << "Total Keypoints" << "," << "Vehicle Keypoints" << "," << "Process Time" << endl;;
    }
    if (writedescriptorsFlag)
    {
        features_file2.open("../descriptor_results/descriptorType_SIFT_SIFT.csv"); // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        features_file2 << "Detector Type/ Descriptor Type  " <<"," << "Matches" <<  "," << "Process Time" << "," << "Total Time" << endl;;
    }

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
 
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        if (dataBuffer.size() == dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
            cout << "#0 : ERASE BUFFER DONE" << endl;
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER DONE" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = detectorType_t; // SHITOMASI , HARRIS ,FAST BRISK, ORB, AKAZE, SIFT 
   
        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            double t = (double)cv::getTickCount();
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            detector_time = 1000 * t / 1.0;
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            double t = (double)cv::getTickCount();
            detKeypointsHarris(keypoints, imgGray, bVis);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            detector_time = 1000 * t / 1.0;
        }
        else if ((detectorType.compare("FAST") == 0) || (detectorType.compare("BRISK") == 0) ||
                  (detectorType.compare("ORB") == 0)  || (detectorType.compare("AKAZE") == 0) ||
                  (detectorType.compare("SIFT") == 0))
        {
           double t = (double)cv::getTickCount();
           detKeypointsModern(keypoints, imgGray, detectorType, bVis);
           t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
           detector_time = 1000 * t / 1.0;
        }
        else
        {
            throw invalid_argument(detectorType + "is not valid. Only -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT ");
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> vehicleKeyPoints;
            for (auto key_Points : keypoints)
            {
                if (vehicleRect.contains(key_Points.pt))
                {
                    vehicleKeyPoints.push_back(key_Points);
                }
            }
            if (writedetectorsFlag)
            {
                features_file << /*"Detector Type = " <<*/ detectorType;
                features_file <</*"Image Index = "*/"," << imgIndex;
                features_file <</*"Total Keypoints = "*/"," << keypoints.size();
                features_file <</* "Vehicle Keypoints = "*/"," << vehicleKeyPoints.size();
                features_file <</* "Process Time = "*/"," << detector_time << endl;
            }
            cout << "NUMBER OF TOTAL KEYPOINTS = " << keypoints.size() << endl;
            cout << "NUMBER OF VEHICLE KEYPOINTS = " << vehicleKeyPoints.size() << endl;
            cout << "Process Time  = " << detector_time << endl;
            keypoints = vehicleKeyPoints;
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = descriptorType_t; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG  !!! DES_HOG FOR  SIFT
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            double t = (double)cv::getTickCount();

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            descriptor_time = 1000 * t / 1.0;
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            features_file2<< detectorType_t << "/" << descriptorType_t << "," << matches.size() << "," << descriptor_time <<","<< detector_time+ descriptor_time<< endl;
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            //bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            //bVis = false;
        }

    } // eof loop over all images
    features_file.close();
    features_file2.close(); 
    return 0;
}
