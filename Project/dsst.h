#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include <vector>
#include "fhog.h"


using namespace cv;
using namespace std;


#define PI 3.141592653589793

class DSST
{
public:

	int frame = 1;

	//taillight recognition
	int taillight_frame = 0;
	int signal_state = 0; //§À¿O«H¸¹ª¬ºA
	vector<double>intensity_L;
	vector<double>intensity_R;


	//feature
	int feature_flag = 0; //0:gray feature ,1:hog feature¡C
	int cell_size = 1; //1:gray feature, 4: hog feature¡C


	bool init = false;

	float interp_factor = 0.075;
	float padding = 1;
	float lambda = 1e-4;
	float output_sigma_factor = 0.01;
	float scale_sigma_factor = 0.18;
	Rect result_rect;
	Point pos;
	Size target_sz;
	Size window_sz;
	Size patch_sz;



	Mat yf;
	Mat cos_window;
	Mat cos_window_scale;

	vector<Mat> A_new;
	Mat B_new;




	//scale
	int n_scales =33;
	float *scaleFactors;
	float currentScaleFactor = 1;
	float scale_step = 1.05;
	float  scale_lr = 0.075;
	float scale_lambda = 1e-4;

	int base_width;
	int base_height;
	int  scale_max_area = 512;
	int scale_model_width;
	int scale_model_height;



	Mat sf_den;
	Mat sf_num;
	Mat s_hann;
	Mat ysf;


	FHoG f_hog;
	//feature
	vector<Mat> DSST::GRAY(Mat img);

	//dsst scale
	void DSST::dsstInit(Rect &roi, Mat image);
	Mat DSST::get_scale_sample(Mat & image);

	Mat DSST::createHanningMatsForScale();
	Mat DSST::computeYsf();
	void DSST::train_scale(Mat image, bool ini);
	void DSST::update_roi();
	Point2i DSST::detect_scale(Mat image);
	vector<Mat> DSST::GetFeatures_scale(Mat patch);

	/*---------------------------------------------------*/
	void DSST::Init(Mat image, Rect rect_init);
	Rect DSST::Update(Mat image,bool track_size);
	void DSST::Learn(Mat &patch, float lr);


	void DSST::hann2d(Mat& m);
	Mat DSST::GetSubwindow(Mat im, Point pos, Size sz);	//, Mat& cos_window);
	vector<Mat> DSST::GetFeatures(Mat patch);

	Mat DSST::CreateGaussian1(int n, double sigma, int ktype);
	Mat DSST::CreateGaussian2(Size sz, double sigma, int ktype);
	Mat DSST::GaussianShapedLabels(float sigma, Size sz);

	Mat DSST::ComplexMul(Mat x1, Mat x2);
	Mat DSST::ComplexDiv(Mat x1, Mat x2);

	Size DSST::FloorSizeScale(cv::Size sz, double scale_factor);
	Point DSST::FloorPointScale(cv::Point p, double scale_factor);

};