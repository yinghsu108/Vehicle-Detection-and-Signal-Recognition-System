/*
A Real-Time Forward Vehicle Detection and
Signals Recognition System in All-Weather Situations

1. Adaboost
2. vehicle detection & taillight detection(daytime)
3. vehicle detection & taillight detection(daytime)
4. signal recognition

*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>



using namespace cv;
using namespace std;

class classifier
{
public:
	double value;		//切在哪一點
	int dimension;		//維度
	double confidence;	//分類器信心值
	int p;				//方向性
};

class bounding_box
{
public:
	int left_top_x = INT_MAX;
	int left_top_y = INT_MAX;
	int right_bottom_x = 0;
	int right_bottom_y = 0;
	int area = 0;
	Point center=Point(0,0);

	bool use_flag = false;
};


class light_pair
{
public:
	int i;
	int j;
	double confidence = 0;
};


class Adaboost
{
public:
	//adaboost
	vector< vector<classifier>> Adaboost::read_file(String file, vector<double> &thresh);
	int Adaboost::predict(vector< vector<classifier>>cascade_classifier, vector<double> thresh, vector<double>histogram, double &confidence_value);

};


class Vehicle
{
public:

	Adaboost adaboost;

	//classify
	int Vehicle::classify(Mat img, Point &left_top, Point& right_bottom, Point center, vector< vector<classifier>>cascade_classifier, vector<double> thresh);

	//feature
	vector<int> Vehicle::LBP(Mat img);
	vector<int> Vehicle::LBP_H(Mat img);
	Mat Vehicle::LBP_graph(Mat img);
	vector<Mat> Vehicle::MB_LBP_graph(Mat img);

	//shadow detection
	Mat Vehicle::gray(Mat img);
	Mat Vehicle::sobel(Mat img, int vertical_Horizontal, int T1); //垂直或水平邊運算
	Mat Vehicle::CDF(Mat img, double T_scale);//累積密度函數
	void Vehicle::resolve(int a, int b,int label);//校正關係(合併標籤)
	vector<bounding_box> Vehicle::vehicle_detection(Mat img, double h_w_scale_min, double h_w_scale_max, double limit_density);//陰影偵測


	//taillight detection
	vector<bounding_box> Vehicle::connect_compoent(Mat img);
	Mat Vehicle::otsu(Mat img, int T);
	int Vehicle::boundingBOX_area_compute(bounding_box r1);

	double Vehicle::HS(bounding_box r1, bounding_box r2);//高度
	double Vehicle::AS(bounding_box r1, bounding_box r2);//面積
	double Vehicle::WS(bounding_box r1, bounding_box r2);//寬度
	double Vehicle::AR(bounding_box r1, bounding_box r2);

	int Vehicle::taillight_detection(Mat img, vector<bounding_box >&pair_bounding_box, vector<bounding_box >&light_bounding_box);//車燈偵測
};



class Vehicle_nighttime
{
public:

	Adaboost adaboost;

	//classify
	int Vehicle_nighttime::classify(Mat img, vector< vector<classifier>>cascade_classifier, vector<double> thresh);


	//feature
	vector<int> Vehicle_nighttime::LBP(Mat img);
	vector<int> Vehicle_nighttime::LBP_H(Mat img);
	Mat Vehicle_nighttime::LBP_graph(Mat img);
	vector<Mat> Vehicle_nighttime::MB_LBP_graph(Mat img);

	//Nighttime vehicle detection
	Mat Vehicle_nighttime::otsu(Mat img, int T);
	Mat Vehicle_nighttime::color_filter(Mat img, double limit_density);
	void Vehicle_nighttime::resolve(int a, int b, int label);
	vector<bounding_box> Vehicle_nighttime::connect_compoent(Mat img);

	Mat Vehicle_nighttime::image_enhancement(Mat img);

	double Vehicle_nighttime::HS(bounding_box r1, bounding_box r2);
	double Vehicle_nighttime::WS(bounding_box r1, bounding_box r2);
	double  Vehicle_nighttime::AS(bounding_box r1, bounding_box r2);
	double  Vehicle_nighttime::AR(bounding_box r1, bounding_box r2);


	vector<Rect> Vehicle_nighttime::vehicle_detection(Point x1, Mat original_img, Mat img, vector< vector<classifier>>cascade_classifier, vector<double> thresh, double limit_density);
	int Vehicle_nighttime::taillight_detection(Mat img, vector<bounding_box >&pair_bounding_box, vector<bounding_box >&light_bounding_box);
};



class Signal
{
public:

	Adaboost adaboost;

	vector<double>Signal::featureTOfrequency(vector<double>hsv_v);
	vector<double>Signal::normalize(vector<double>hsv_v);
	double Signal::compute_intensity(Mat img, bounding_box bounding_box);
	void Signal::signal_recognition(int daytime_nighttime,int frame_sample, int label, int signal_state_pre, int &signal_state, vector<double >intensity_L, vector<double >intensity_R, vector<bounding_box >pair_bounding_box, vector<bounding_box >light_bounding_box, vector< vector<classifier>>cascade_classifier, vector<double> thresh);//車燈辨識



};

