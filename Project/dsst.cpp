#include "dsst.h"


vector<Mat> DSST::GRAY(Mat img)
{
	vector<Mat> x_vector;

	x_vector.push_back(img);

	return x_vector;
}




void DSST::dsstInit(Rect &roi, Mat image)
{

	currentScaleFactor = 1;

	base_width = roi.width;
	base_height = roi.height;

	scaleFactors = new float[n_scales];
	float ceilS = ceil(n_scales / 2.0f);

	ysf = computeYsf();
	s_hann = createHanningMatsForScale();
	for (int i = 0; i < n_scales; i++)
	{
		scaleFactors[i] = pow(scale_step, ceilS - i - 1);
	}

	float scale_model_factor = 1;

	if (base_width * base_height > scale_max_area)
	{
		scale_model_factor = sqrt(scale_max_area / (float)(base_width * base_height));
	}
	scale_model_width = (int)(base_width * scale_model_factor);
	scale_model_height = (int)(base_height * scale_model_factor);


	cos_window_scale.create(FloorSizeScale(Size(scale_model_width, scale_model_height), 1. / cell_size), CV_32FC1);
	hann2d(cos_window_scale);


	train_scale(image, true);

}

void DSST::train_scale(Mat image, bool ini)
{
	Mat xsf = get_scale_sample(image);

	if (ini)
	{
		int totalSize = xsf.rows;
		ysf = repeat(ysf, totalSize, 1);

	}

	Mat new_sf_num;
	mulSpectrums(ysf, xsf, new_sf_num, 0, true);

	Mat new_sf_den;
	mulSpectrums(xsf, xsf, new_sf_den, 0, true);

	reduce(new_sf_den, new_sf_den, 0, CV_REDUCE_SUM);

	if (ini)
	{
		sf_den = new_sf_den;
		sf_num = new_sf_num;
	}
	else
	{
		sf_den = sf_den* (1 - scale_lr) + new_sf_den*scale_lr;
		sf_num = sf_num* (1 - scale_lr) + new_sf_num*scale_lr;
	}

	update_roi();
}

void DSST::update_roi()
{

	result_rect.width = base_width * currentScaleFactor;
	result_rect.height = base_height * currentScaleFactor;
}

Mat DSST::get_scale_sample(Mat & image)
{

	Mat xsf;
	int totalSize;

	for (int i = 0; i < n_scales; i++)
	{
		float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
		float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

		Size sz = Size(patch_width, patch_height);
		Mat im_patch = GetSubwindow(image, pos, sz);
		Mat im_patch_resized;

		resize(im_patch, im_patch_resized, Size(scale_model_width, scale_model_height), 0, 0, CV_INTER_LINEAR);

		/*
		if (i == 16)
		{
		imshow("patch1", im_patch_resized);
		}*/

		/*--------------
		Get gray feature
		---------------*/

		vector<Mat>x_vector;
		x_vector = GetFeatures_scale(im_patch_resized);

		if (i == 0)
		{
			totalSize = x_vector[0].cols * x_vector[0].rows * x_vector.size();
			xsf = Mat(Size(n_scales, totalSize), CV_32F, float(0));

		}


		Mat FeaturesMap = Mat(Size(1, totalSize), CV_32F);
		int j = 0;
		for (int k = 0; k<x_vector.size(); k++)
		{
			for (int y = 0; y < x_vector[k].rows; y++)
			{
				float* data = x_vector[k].ptr<float>(y);
				for (int x = 0; x < x_vector[k].cols; x++)
				{
					float* data_out = FeaturesMap.ptr<float>(j);
					data_out[0] = data[x];
					j++;
				}
			}
		}

		float mul = s_hann.at<float >(0, i);
		FeaturesMap = mul * FeaturesMap;
		FeaturesMap.copyTo(xsf.col(i));
	}

	dft(xsf, xsf, (DFT_ROWS | DFT_COMPLEX_OUTPUT));

	return xsf;
}
Point2i DSST::detect_scale(Mat image)
{
	Mat xsf = get_scale_sample(image);

	Mat add_temp;

	reduce(ComplexMul(sf_num, xsf), add_temp, 0, CV_REDUCE_SUM);
	Mat scale_response;

	idft(ComplexDiv(add_temp, sf_den + Scalar(scale_lambda, 0)), scale_response, DFT_SCALE | DFT_REAL_OUTPUT);


	Point2i pi;
	double pv;
	minMaxLoc(scale_response, NULL, &pv, NULL, &pi);

	return pi;
}





Mat DSST::createHanningMatsForScale()
{
	Mat hann_s = Mat(Size(n_scales, 1), CV_32F, cv::Scalar(0));
	for (int i = 0; i < hann_s.cols; i++)
	{
		for (int j = 0; j < hann_s.rows; j++)
		{
			hann_s.at<float >(j, i) = 0.5 * (1 - cos(2 * CV_PI * i / (hann_s.cols - 1)));

		}
	}
	return hann_s;

}


Mat DSST::computeYsf()
{
	float scale_sigma2 = n_scales / sqrt(n_scales) * scale_sigma_factor;
	scale_sigma2 = scale_sigma2 * scale_sigma2;
	Mat res(Size(n_scales, 1), CV_32F, float(0));
	float ceilS = ceil(n_scales / 2.0f);

	for (int i = 0; i < n_scales; i++)
	{
		res.at<float>(0, i) = exp(-0.5 * pow(i + 1 - ceilS, 2) / scale_sigma2);
	}

	dft(res, res, DFT_COMPLEX_OUTPUT);
	return res;
}






/*---------------------------------------------------
---------------------------------------------------*/

void DSST::Init(Mat image, Rect rect_init) {
	result_rect = rect_init;
	pos = Point(rect_init.x + cvFloor((float)(rect_init.width) / 2.),
		rect_init.y + cvFloor((float)(rect_init.height) / 2.));

	dsstInit(rect_init, image);

	if (sqrt(base_width*base_height)>300)
	{
		patch_sz.width = base_width / 2;
		patch_sz.height = base_height / 2;
	}
	else
	{
		patch_sz.width = base_width;
		patch_sz.height = base_height;
	}



	target_sz = rect_init.size();
	window_sz = FloorSizeScale(target_sz, 1 + padding);
	float output_sigma = sqrt(float(target_sz.area())) * output_sigma_factor / cell_size;
	dft(GaussianShapedLabels(output_sigma, FloorSizeScale(patch_sz, 1. / cell_size)), yf, DFT_COMPLEX_OUTPUT);

	//cos_window.create(yf.size(), CV_32FC1);
	//hann2d(cos_window);



	Mat patch = GetSubwindow(image, pos, window_sz);
	resize(patch, patch, patch_sz, 0, 0, CV_INTER_LINEAR);

	Learn(patch, 1.);

}


void DSST::Learn(Mat &patch, float lr) {

	vector<Mat> f = GetFeatures(patch);
	vector<Mat> F(f.size());
	vector<Mat> A(f.size());
	Mat temp;


	for (unsigned int i = 0; i < f.size(); i++)
		dft(f[i], F[i], DFT_COMPLEX_OUTPUT);


	Mat B = Mat::zeros(F[0].size(), CV_32FC2);
	for (unsigned int i = 0; i < f.size(); i++)
	{
		mulSpectrums(yf, F[i], A[i], 0, true);
		mulSpectrums(F[i], F[i], temp, 0, true);
		B += temp;
	}


	if (lr > 0.99) {

		A_new.clear();
		for (unsigned int i = 0; i < f.size(); i++)
		{
			A_new.push_back(A[i]);
		}
		B_new = B;

	}
	else {
		for (unsigned int i = 0; i < f.size(); i++)
		{
			A_new[i] = (1 - lr)*A_new[i] + lr*A[i];
		}
		B_new = (1 - lr)*B_new + lr*B;
	}


}



Rect DSST::Update(Mat image,bool track_size) {

	frame++;

	window_sz = FloorSizeScale(result_rect.size(), 1 + padding);
	Mat patch = GetSubwindow(image, pos, window_sz);
	resize(patch, patch, patch_sz, 0, 0, CV_INTER_LINEAR);

	vector<Mat> z = GetFeatures(patch);
	vector<Mat> zf_vector(z.size());

	for (unsigned int i = 0; i < z.size(); ++i)
		dft(z[i], zf_vector[i], DFT_COMPLEX_OUTPUT);

	Mat temp = Mat::zeros(A_new[0].size(), CV_32FC2);

	for (unsigned int i = 0; i < z.size(); ++i)
	{
		temp += ComplexMul(zf_vector[i], A_new[i]);
	}

	Mat response;
	idft(ComplexDiv(temp, B_new + Scalar(lambda, 0)), response, DFT_SCALE | DFT_REAL_OUTPUT);


	Point maxLoc;
	minMaxLoc(response, NULL, NULL, NULL, &maxLoc);


	maxLoc.x = maxLoc.x - response.cols / 2 + 1;
	maxLoc.y = maxLoc.y - response.rows / 2 + 1;

	pos.x += cell_size * maxLoc.x*(float(window_sz.width) / patch_sz.width);
	pos.y += cell_size * maxLoc.y*(float(window_sz.height) / patch_sz.height);

	//pos.x += cell_size * maxLoc.x;
	//pos.y += cell_size * maxLoc.y;

	result_rect.x = pos.x - result_rect.width / 2;
	result_rect.y = pos.y - result_rect.height / 2;

	
	//Update scale
	if (track_size == true)
	{
		Point2i scale_pi = detect_scale(image);
		currentScaleFactor = currentScaleFactor * scaleFactors[scale_pi.x];
		train_scale(image, false);
	}
	






	window_sz = FloorSizeScale(result_rect.size(), 1 + padding);
	patch = GetSubwindow(image, pos, window_sz);
	resize(patch, patch, patch_sz, 0, 0, CV_INTER_LINEAR);

	Learn(patch, interp_factor);

	return result_rect;
}





void DSST::hann2d(Mat& m)
{
	Mat a(m.rows, 1, CV_32FC1);
	Mat b(m.cols, 1, CV_32FC1);
	for (int i = 0; i < m.rows; i++)
	{
		float *data = a.ptr<float>(i);
		float t = 0.5 * (1 - cos(2 * CV_PI*i / (m.rows - 1)));
		data[0] = t;
	}
	for (int i = 0; i < m.cols; i++)
	{
		float *data = b.ptr<float>(i);
		float t = 0.5 * (1 - cos(2 * CV_PI*i / (m.cols - 1)));
		data[0] = t;
	}
	m = a * b.t();
}




Mat DSST::GetSubwindow(Mat im, Point pos, Size sz)
{
	vector<int> xs(sz.width);
	vector<int> ys(sz.height);
	for (int i = 0; i < sz.width; i++)
	{
		xs[i] = floor(pos.x) + i - floor(sz.width / 2);
		xs[i] = max(min(xs[i], im.cols - 1), 0);
	}
	for (int i = 0; i < sz.height; i++) {
		ys[i] = floor(pos.y) + i - floor(sz.height / 2);
		ys[i] = max(min(ys[i], im.rows - 1), 0);
	}

	if (im.channels() == 1)
	{
		Mat out(sz.height, sz.width, CV_8UC1);
		for (int i = 0; i < sz.height; i++)
		{
			uchar *data = im.ptr<uchar>(ys[i]);
			uchar *data_out = out.ptr<uchar>(i);
			for (int j = 0; j < sz.width; j++)
			{
				data_out[j] = (data[xs[j]]);
			}
		}
		return out;
	}
	else
	{
		Mat out(sz.height, sz.width, CV_8UC3);
		for (int i = 0; i < sz.height; i++)
		{
			Vec3b *data = im.ptr<Vec3b>(ys[i]);
			Vec3b *data_out = out.ptr<Vec3b>(i);
			for (int j = 0; j < sz.width; j++)
			{
				data_out[j][0] = (data[xs[j]][0]);
				data_out[j][1] = (data[xs[j]][1]);
				data_out[j][2] = (data[xs[j]][2]);
			}
		}
		return out;
	}

}

vector<Mat> DSST::GetFeatures_scale(Mat patch) {

	vector<Mat> x_vector;

	if (patch.channels() == 3)
		cvtColor(patch, patch, CV_BGR2GRAY);
	patch.convertTo(patch, CV_32FC1, 1.0 / 255);
	/*---------------
		1.GRAY feature
	注意:cellsize:改為 1
		2.HOG feature
	注意:cellsize:改為 4
	----------------*/
	if(feature_flag==0)
		x_vector = GRAY(patch);
	else
		x_vector = f_hog.extract(patch);
	
	

	for (unsigned int i = 0; i < x_vector.size(); ++i)
		x_vector[i] = x_vector[i].mul(cos_window_scale);

	return x_vector;
}

vector<Mat> DSST::GetFeatures(Mat patch) {

	vector<Mat> x_vector;

	if (patch.channels() == 3)
		cvtColor(patch, patch, CV_BGR2GRAY);
	patch.convertTo(patch, CV_32FC1, 1.0 / 255);

	/*---------------
		1.GRAY feature
	注意:cellsize:改為 1
		2.HOG feature
	注意:cellsize:改為 4
	----------------*/
	if (feature_flag == 0)
		x_vector = GRAY(patch);
	else
		x_vector = f_hog.extract(patch);
	

	cos_window.create(x_vector[0].size(), CV_32FC1);
	hann2d(cos_window);

	for (unsigned int i = 0; i < x_vector.size(); ++i)
		x_vector[i] = x_vector[i].mul(cos_window);

	return x_vector;
}


Mat DSST::CreateGaussian1(int n, double sigma, int ktype)
{

	Mat kernel(n, 1, ktype);
	float* cf = kernel.ptr<float>();
	double* cd = kernel.ptr<double>();

	double scale2X = -0.5 / (sigma*sigma * 2 * PI);

	int i;
	for (i = 0; i < n; i++)
	{
		double x = i - floor(n / 2) + 1;
		double t = exp(scale2X*x*x);
		if (ktype == CV_32F)
		{
			cf[i] = (float)t;
		}
		else
		{
			cd[i] = t;
		}
	}

	return kernel;
}

Mat DSST::CreateGaussian2(Size sz, double sigma, int ktype)
{
	Mat a = CreateGaussian1(sz.height, sigma, ktype);
	Mat b = CreateGaussian1(sz.width, sigma, ktype);
	return a*b.t();
}

Mat DSST::GaussianShapedLabels(float sigma, Size sz) {

	Mat labels = CreateGaussian2(sz, sigma, CV_32FC1);

	return labels;
}

Mat DSST::ComplexMul(Mat x1, Mat x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
	complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
	Mat result;
	merge(complex, result);
	return result;
}

Mat DSST::ComplexDiv(Mat x1, Mat x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	Mat cc = planes2[0].mul(planes2[0]);
	Mat dd = planes2[1].mul(planes2[1]);

	complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
	complex[1] = (-planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0])) / (cc + dd);
	Mat result;
	merge(complex, result);
	return result;
}

Size DSST::FloorSizeScale(cv::Size sz, double scale_factor) {
	if (scale_factor > 0.9999 && scale_factor < 1.0001)
		return sz;
	return Size(cvFloor(sz.width * scale_factor),
		cvFloor(sz.height * scale_factor));
}


Point DSST::FloorPointScale(cv::Point p, double scale_factor) {
	if (scale_factor > 0.9999 && scale_factor < 1.0001)
		return p;
	return cv::Point(cvFloor(p.x * scale_factor),
		cvFloor(p.y * scale_factor));
}
