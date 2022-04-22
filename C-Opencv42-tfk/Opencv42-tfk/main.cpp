#include<stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void InitMat(Mat& m, float* num)
{
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			m.at<float>(i, j) = *(num + i * m.rows + j);
}

int testmodel1() {
	float sz[] = { 1,1,1,1 };
	Mat input(1, 4, CV_32F);
	InitMat(input, sz);

	dnn::Net net = dnn::readNetFromTensorflow("frozen_graph.pb");
	net.setInput(input);
	Mat pred = net.forward();
	cout << pred << endl; //[32.246185]


	return 0;
}

int testmodel2() {
	Mat test_x = imread("1.png", 0);
	test_x = cv::dnn::blobFromImage(test_x,1.0,Size(28, 28));
	dnn::Net net = cv::dnn::readNetFromTensorflow("simple_frozen_graph.pb");
	net.setInput(test_x);
	Mat pred = net.forward();
	cout << pred << endl; //[32.246185]

	return 0;
}

int main() {
	testmodel1();
	testmodel2();
	return 0;
}