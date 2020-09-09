// OpenCVTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// Reference https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
// https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
// https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
// https://github.com/kanika2018/Object-Recognition-using-SIFT/blob/master/object_recog_SIFT.py
// https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html
// https://stackoverflow.com/questions/52425355/how-to-detect-multiple-objects-with-opencv-in-c
#include <opencv2/features2d.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"


#include <opencv2/core/types_c.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

#pragma warning(disable:4996)

using namespace std;
using namespace cv;

struct corner {
	Point corner1;
	Point corner2;
	Point corner3;
	Point corner4;
};

bool comparator(DMatch a, DMatch b)
{
	return a.distance < b.distance;
}

bool comp2f(const Point2f& a, const Point2f& b) {
	if (a.x <= b.x) {
		if (a.y <= b.y) {
			return true;
		}
	}
	return false;
}

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
	// treat two empty mat as identical as well
	if (mat1.empty() && mat2.empty()) {
		return true;
	}
	// if dimensionality of two mat is not identical, these two mat is not identical
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
		return false;
	}
	cv::Mat diff;
	cv::compare(mat1, mat2, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff);
	return nz == 0;
}

void MyPolygon(Mat img, Point p1, Point p2, Point p3, Point p4)
{

	int lineType = LINE_8;

	/** Create some points */
	Point p[1][8];


	p[0][0] = Point(p1.x, p1.y);
	p[0][1] = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
	p[0][2] = Point(p2.x, p2.y);
	p[0][3] = Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2);
	p[0][4] = Point(p3.x, p3.y);
	p[0][5] = Point((p3.x + p4.x) / 2, (p3.y + p4.y) / 2);
	p[0][6] = Point(p4.x, p4.y);
	p[0][7] = Point((p4.x + p1.x) / 2, (p4.y + p1.y) / 2);


	/*
	p[0][0] = Point(p1.x, p1.y);
	p[0][2] = Point(p2.x, p2.y);
	p[0][4] = Point(p3.x, p3.y);
	p[0][6] = Point(p4.x, p4.y);
	*/

	printf("_---------1\n");

	const Point* ppt[1] = { p[0] };
	int npt[] = { 8 };

	fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), lineType);
	printf("_---------2\n");

}



int main() {
	int countobj = 0;
	int spot = 1;
	printf("How many object you want to spot in (0 to exit): ");
	scanf("%d", &spot);

	if (spot <= 0) {
		return 0;
	}

	vector<corner> location;

	string bigfile;
	string smallfile;

	printf("Path to scene file: ");
	getline(cin, bigfile);
	printf("Path to smaller file: ");
	getline(cin, smallfile);

	Mat photo = imread(bigfile);
	Mat childphoto = imread(smallfile);

	InputArray pt = cv::_InputArray::_InputArray(photo);
	InputArray cpt = cv::_InputArray::_InputArray(childphoto);

	InputArray mask = cv::_InputArray::_InputArray();
	InputArray maskchild = cv::_InputArray::_InputArray();

	vector<KeyPoint> keypoint, keypointchild;


	Mat descriptor, descriptorchild;




	Ptr<SIFT> sift = SIFT::create();
	Ptr<SIFT> siftchild = SIFT::create();

	// sift->detect(pt, keypoint);
	// siftchild->detect(cpt, keypointchild);

	sift->detectAndCompute(pt, noArray(), keypoint, descriptor, false);
	printf("Main photo have: %d keypoints\n", keypoint.size());

	siftchild->detectAndCompute(cpt, noArray(), keypointchild, descriptorchild, false);
	printf("Another photo have: %d keypoints\n", keypointchild.size());



	vector<DMatch> matches;

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);


	matcher->match(descriptor, descriptorchild, matches);

	double max_dist = 0; double min_dist = 100;

	// Quick calculation of max and min distances between keypoints
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

	//sort matches
	std::sort(matches.begin(), matches.end(), comparator);

	//append first 15% to goodmatch
	int n = round(matches.size()*0.15);
	// int m = round(matches.size()*0.5);
	// int n = 10;
	for (int i = 0; i < n; i++) {
		goodmatches.push_back(matches[i]);
		printf("Distance: %.6f, Child: %d, Main: %d\n", goodmatches.at(i).distance, goodmatches.at(i).queryIdx, goodmatches.at(i).trainIdx);
	}



	printf("Size of good match: %d\n", goodmatches.size());


	drawMatches(pt, keypoint, cpt, keypointchild, goodmatches, OutImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	printf("Draw matches sucessful\n");

	// now to match it


	// Localize the object
	vector<Point2f> scene;
	vector<Point2f> scenechild;

	for (int i = 0; i < goodmatches.size(); i++)
	{
		// Get the keypoints from the good matches
		scene.push_back(keypoint[goodmatches[i].queryIdx].pt);
		scenechild.push_back(keypointchild[goodmatches[i].trainIdx].pt);
	}

	printf("Get keypoint 1 successful\n");

	Mat H = findHomography(scenechild, scene, RANSAC);

	// Get the corners from the image_1 ( the object to be "detected" )
	vector<Point2f> scenechild_corners(4);
	vector<Point2f> scene_corners(4);
	scenechild_corners[0] = cvPoint(0, 0);
	scenechild_corners[1] = cvPoint(childphoto.cols, 0);
	scenechild_corners[2] = cvPoint(childphoto.cols, childphoto.rows);
	scenechild_corners[3] = cvPoint(0, childphoto.rows);


	perspectiveTransform(scenechild_corners, scene_corners, H);

	if ((scene_corners[0].x > 0 && scene_corners[0].x < photo.cols + 100 && scene_corners[0].y > 0 && scene_corners[0].y < photo.rows + 100) &&
		(scene_corners[1].x > 0 && scene_corners[1].x < photo.cols + 100 && scene_corners[1].y > 0 && scene_corners[1].y < photo.rows + 100) &&
		(scene_corners[2].x > 0 && scene_corners[2].x < photo.cols + 100 && scene_corners[2].y > 0 && scene_corners[2].y < photo.rows + 100) &&
		(scene_corners[3].x > 0 && scene_corners[3].x < photo.cols + 100 && scene_corners[3].y > 0 && scene_corners[3].y < photo.rows + 100)
		) {
		line(OutImg, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
		line(OutImg, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
		line(OutImg, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
		line(OutImg, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
		countobj++;
	}

	sort(scene_corners.begin(), scene_corners.end(), comp2f);


	Point sc[4];
	sc[0].x = round(scene_corners.at(0).x);
	sc[0].y = round(scene_corners.at(0).y);
	sc[1].x = round(scene_corners.at(1).x);
	sc[1].y = round(scene_corners.at(1).y);
	sc[2].x = round(scene_corners.at(2).x);
	sc[2].y = round(scene_corners.at(2).y);
	sc[3].x = round(scene_corners.at(3).x);
	sc[3].y = round(scene_corners.at(3).y);



	for (int i = 0; i < 4; i++) {
		printf("%.6f %.6f\n", scene_corners.at(i).x, scene_corners.at(i).y);
		printf("%d %d\n", sc[i].x, sc[i].y);
	}
	printf("Get corner 1 sucessful\n");

	// blur img to avoid keypoint generation
	MyPolygon(photo, sc[0], sc[1], sc[3], sc[2]);
	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/temp.jpg", photo);


	if (spot == 1) {
		;
	}
	else {
		for (int it = 2; it <= spot; it++) {
			InputArray pt = cv::_InputArray::_InputArray(photo);

			keypoint.clear();
			Mat descriptorlocal;

			sift->detectAndCompute(pt, noArray(), keypoint, descriptorlocal, false);
			printf("Recalculated photo have: %d keypoints\n", keypoint.size());

			matches.clear();

			Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

			matcher1->match(descriptorlocal, descriptorchild, matches);

			goodmatches.clear();
			printf("Size of match: %d\n", matches.size());

			//sort matches
			std::sort(matches.begin(), matches.end(), comparator);

			int n = round(matches.size()*0.15);

			// terminate when n < 4 (not reliable match)
			if (n == 4) {
				printf("Stop because there are no match left");
				break;
			}


			for (int i = 0; i < n; i++) {
				goodmatches.push_back(matches[i]);
			}

			scene.clear();
			scenechild.clear();

			printf("Size of good match: %d\n", goodmatches.size());

			for (int i = 0; i < goodmatches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				scene.push_back(keypoint[goodmatches[i].queryIdx].pt);
				scenechild.push_back(keypointchild[goodmatches[i].trainIdx].pt);
			}

			printf("Get keypoint i successful\n", it);

			// Mat H = findHomography(scenechild, scene, RANSAC);
			Mat H = findHomography(scenechild, scene, RANSAC);

			perspectiveTransform(scenechild_corners, scene_corners, H);

			if ((scene_corners[0].x > 0 && scene_corners[0].x < photo.cols + 100 && scene_corners[0].y > 0 && scene_corners[0].y < photo.rows + 100) &&
				(scene_corners[1].x > 0 && scene_corners[1].x < photo.cols + 100 && scene_corners[1].y > 0 && scene_corners[1].y < photo.rows + 100) &&
				(scene_corners[2].x > 0 && scene_corners[2].x < photo.cols + 100 && scene_corners[2].y > 0 && scene_corners[2].y < photo.rows + 100) &&
				(scene_corners[3].x > 0 && scene_corners[3].x < photo.cols + 100 && scene_corners[3].y > 0 && scene_corners[3].y < photo.rows + 100)
				) {
				line(OutImg, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
				line(OutImg, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
				line(OutImg, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
				line(OutImg, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

				countobj++;
			}


			sort(scene_corners.begin(), scene_corners.end(), comp2f);


			Point sc[4];
			sc[0].x = round(scene_corners.at(0).x);
			sc[0].y = round(scene_corners.at(0).y);
			sc[1].x = round(scene_corners.at(1).x);
			sc[1].y = round(scene_corners.at(1).y);
			sc[2].x = round(scene_corners.at(2).x);
			sc[2].y = round(scene_corners.at(2).y);
			sc[3].x = round(scene_corners.at(3).x);
			sc[3].y = round(scene_corners.at(3).y);





			for (int i = 0; i < 4; i++) {
				printf("%.6f %.6f\n", scene_corners.at(i).x, scene_corners.at(i).y);
				printf("%d %d\n", sc[i].x, sc[i].y);
			}
			printf("Get corner %d sucessful\n", it);

			//blur img to avoid keypoint generation
			MyPolygon(photo, sc[0], sc[1], sc[3], sc[2]);

		}
	}

	printf("Found: %d object in picture\n", countobj);
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", OutImg);                   // Show our image inside it.
	imwrite("C:/Users/kien.nm173206/Desktop/SIFT/final_out_52.jpg", OutImg);

	waitKey(0);

}


