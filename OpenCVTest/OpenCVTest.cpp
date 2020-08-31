// OpenCVTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// Reference https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
// https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
// https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
// https://github.com/kanika2018/Object-Recognition-using-SIFT/blob/master/object_recog_SIFT.py
// https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html
#include <opencv2/features2d.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

using namespace std;
using namespace cv;

bool comparator(DMatch a, DMatch b)
{
	return a.distance < b.distance;
}

int main() {
	// Mat photo = imread("C:/Users/kien.nm173206/Desktop/SIFT/20200823_234943.jpg");
	// Mat childphoto = imread("C:/Users/kien.nm173206/Desktop/SIFT/20200823_234957.jpg");

	Mat photo = imread("C:/Users/kien.nm173206/Desktop/SIFT/IMG_0364.JPG");
	Mat childphoto = imread("C:/Users/kien.nm173206/Desktop/SIFT/IMG_0364_CUT.png");


	InputArray pt = cv::_InputArray::_InputArray(photo);
	InputArray cpt = cv::_InputArray::_InputArray(childphoto);

	InputArray mask = cv::_InputArray::_InputArray();
	InputArray maskchild = cv::_InputArray::_InputArray();

	vector<KeyPoint> keypoint, keypointchild;
	/*
	OutputArray descriptor = cv::_OutputArray::_OutputArray();
	OutputArray descriptorchild = cv::_OutputArray::_OutputArray();
	*/

	Mat descriptor, descriptorchild;
	/*
	sift = cv.SIFT_create()
	kp = sift.detect(gray,None)
	img=cv.drawKeypoints(gray,kp,img)
	*/

	

	Ptr<SIFT> sift = SIFT::create();
	Ptr<SIFT> siftchild = SIFT::create();

	// sift->detect(pt, keypoint);
	// siftchild->detect(cpt, keypointchild);

	sift->detectAndCompute(pt, noArray(), keypoint, descriptor, false);
	printf("Main photo have: %d keypoints\n", keypoint.size());

	siftchild->detectAndCompute(cpt, noArray(), keypointchild, descriptorchild, false);
	printf("Another photo have: %d keypoints\n", keypointchild.size());

	/*
	// draw keypoint
	Mat img_keypoints;
	drawKeypoints(pt, keypoint, img_keypoints);
	namedWindow("Display window", WINDOW_AUTOSIZE);		// Create a window for display.
	imshow("Display window", img_keypoints);                   // Show our image inside it.

	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/IMG_0364_out.jpg", img_keypoints);

	Mat img_keypoints1;
	drawKeypoints(cpt, keypointchild, img_keypoints1);
	namedWindow("Display window", WINDOW_AUTOSIZE);		// Create a window for display.
	imshow("Display window", img_keypoints1);                   // Show our image inside it.

	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/IMG_0364_out_1.jpg", img_keypoints1);

	// end
	*/

	// BFMatcher bf = BFMatcher(NORM_L1, false);

	vector<DMatch> matches;

	// bf.match(descriptor, descriptorchild, matches);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	// Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

	// matcher->match(descriptor, descriptorchild, matches);
	// matches = bf->knnMatch(descriptor, descriptorchild, 2);
	
	matcher->match(descriptor, descriptorchild, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptor.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	printf("Matched complete\n");

	Mat OutImg;

	vector<DMatch> goodmatches;
	printf("Size of match: %d\n", matches.size());

	//sort matches
	std::sort(matches.begin(), matches.end(), comparator);

	//append first 20% to goodmatch
	
	/*
	for (int i = 0; i < matches.size(); i++) {
		for (int j = i + 1; j < matches.size(); j++) {
			if (matches.at(i).distance < 0.75 * matches.at(j).distance) {
				goodmatches.push_back(matches.at(i));
				printf("Distance: %.6f\n", goodmatches.at(i).distance);
				break;
			}
		}
	}
	printf("Size of good match: %d\n", goodmatches.size());
	*/
	/*
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	*/
	int n = round(matches.size()*0.15);
	// int n = 10;
	for (int i = 0; i < n; i++) {
		goodmatches.push_back(matches.at(i));
		printf("Distance: %.6f\n", goodmatches.at(i).distance);
	}




	// drawMatches(pt, keypoint, cpt, keypointchild, goodmatches, OutImg);

	drawMatches(pt, keypoint, cpt, keypointchild, goodmatches, OutImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	printf("Draw matches sucessful\n");


	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", OutImg);                   // Show our image inside it.
	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/final_out_16.jpg", OutImg);

	// now to match it

	// M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)
	
	// findHomography()

	waitKey(0);

	// sift->detectAndCompute(pt, mask, keypoint, descriptor, false);
	// sift->detect(pt, keypoint);

	//-- Draw keypoints
	/*
	Mat img_keypoints;
	drawKeypoints(pt, keypoint, img_keypoints);
	*/

	/*
	namedWindow("Display window", WINDOW_AUTOSIZE);		// Create a window for display.
	imshow("Display window", img_keypoints);                   // Show our image inside it.
	// imwrite("C:/Users/kien.nm173206/Desktop/SIFT/20200823_234943_out.jpg", img_keypoints);
	waitKey(0);
	*/

	/*
	Feature2D::detect(pt, keypoint);
	Feature2D::detectAndCompute(pt, mask, keypoint, descriptor, false);
	*/
	/*
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", photo);                   // Show our image inside it.
	waitKey(0);
	*/

	/*
	Mat img_keypoints;
	drawKeypoints(cpt, keypointchild, img_keypoints);
	namedWindow("Display window", WINDOW_AUTOSIZE);		// Create a window for display.
	imshow("Display window", img_keypoints);                   // Show our image inside it.

	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/20200823_234957_out.jpg", img_keypoints);
	waitKey(0);
	*/
	// cv::Feature2D::detectAndCompute()
}