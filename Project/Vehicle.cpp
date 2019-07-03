/*
國立臺灣科技大學
107 電機工程系

A Real-Time Forward Vehicle Detection and
Signals Recognition System in All-Weather Situations

Title: 全天候即時前車偵測與信號辨識系統
Author: Jia-Ying Hsu
Date: July, 2019

1. Adaboost
2. vehicle detection & taillight detection(daytime)
3. vehicle detection & taillight detection(daytime)
4. signal recognition

*/

#include "Vehicle.h"



//======================================================================================================================================================
//===================================================               Adaboost              ==============================================================
//======================================================================================================================================================

vector< vector<classifier>>  Adaboost::read_file(String file, vector<double> &thresh)
{
	vector<int> num_of_iter;
	fstream fin;
	fin.open(file, fstream::in);

	char buffer[1000];
	int n = 0;
	for (int i = 0; i < 1; i++)
	{
		fin >> buffer;
		n = atoi(buffer);
	}

	for (int i = 0; i < n; i++)
	{
		fin >> buffer;
		num_of_iter.push_back(atoi(buffer));
	}

	for (int i = 0; i < n; i++)
	{
		fin >> buffer;
		thresh.push_back(atof(buffer));
	}

	vector< vector<classifier>>cascade_classifier;
	classifier weak_classifier;

	for (int i = 0; i < n; i++)
	{
		vector<classifier> strong_classifier;
		for (int j = 0; j < num_of_iter[i]; j++)
		{
			fin >> buffer;
			weak_classifier.dimension = atoi(buffer);
			fin >> buffer;
			weak_classifier.value = atof(buffer);
			fin >> buffer;
			weak_classifier.confidence = atof(buffer);
			fin >> buffer;
			weak_classifier.p = atoi(buffer);
			strong_classifier.push_back(weak_classifier);
		}
		cascade_classifier.push_back(strong_classifier);
	}

	fin.close();
	return cascade_classifier;

}


int Adaboost::predict(vector< vector<classifier>>cascade_classifier, vector<double> thresh, vector<double>histogram, double &confidence_value)
{
	for (int i = 0; i < cascade_classifier.size(); i++)
	{
		confidence_value = 0;
		for (int j = 0; j < cascade_classifier[i].size(); j++)
		{
			int dim = cascade_classifier[i][j].dimension;

			if (histogram[dim] <= cascade_classifier[i][j].value)
				confidence_value = confidence_value + cascade_classifier[i][j].confidence*cascade_classifier[i][j].p;
			else
				confidence_value = confidence_value - cascade_classifier[i][j].confidence*cascade_classifier[i][j].p;

		}


		if (confidence_value < thresh[i])
		{
			return 0;
		}

	}

	return 1;

}



//======================================================================================================================================================
//====================================================               Daytime              ==============================================================
//======================================================================================================================================================

//==================================================
//=================   Classify  ====================
//==================================================

int Vehicle::classify(Mat img, Point &left_top, Point& right_bottom, Point center, vector< vector<classifier>>cascade_classifier, vector<double> thresh)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	//==================================================
	//================       Sobel      ================
	//==================================================
	Mat ss = sobel(gray, 0, 1);//垂直邊緣
	Mat result(img.rows, img.cols, CV_8UC1);//直方圖
	vector<double>histogram;
	histogram.assign(img.cols, 0);
	int average_h = 0;

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = ss.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data[x] == 255)
			{
				histogram[x]++;
				average_h++;
			}
		}
	}
	average_h = average_h / img.cols;

	for (int y = 0; y < result.rows; y++)
	{
		uchar *data = result.ptr<uchar>(y);
		for (int x = 0; x < result.cols; x++)
		{
			if (y <= histogram[x])
				data[x] = 255;
			else
				data[x] = 0;

		}
	}
	int L_label = 0;
	int R_label = result.cols / 2;
	int max = 0;
	for (int x = result.cols / 2; x > 0; x--)
	{
		if (histogram[x] > max)
		{
			max = histogram[x] - 5;
			L_label = x;
		}
	}
	max = 0;
	for (int x = result.cols / 2; x < result.cols; x++)
	{
		if (histogram[x] > max)
		{
			max = histogram[x] + 5;
			R_label = x;
		}
	}
	int height = (R_label - L_label);
	height = height < img.rows ? height : img.rows;



	Point l = Point(L_label, img.rows - height);
	Point r = Point(R_label, img.rows - 1);
	Rect roi(l, r);


	if (abs(histogram[L_label] - histogram[R_label]) > average_h)
	{
		return 0;
	}


	if ((histogram[L_label] < average_h) || (histogram[R_label] < average_h))
	{
		return 0;
	}
	//==================================================



	if (roi.width*roi.height > 64 * 64)
	{
		gray = gray(roi);
		img = img(roi);

		Mat reize_img;
		resize(gray, reize_img, Size(64, 64));
		vector<Mat>MB_LBP_img = MB_LBP_graph(reize_img);


		vector<double>histogram;
		for (int i = 0; i < MB_LBP_img.size(); i++)
		{
			Mat img = MB_LBP_img[i];
			vector<int>h = LBP_H(img);
			histogram.insert(histogram.end(), h.begin(), h.end());
		}
		double confidence;
		int classify_flag = adaboost.predict(cascade_classifier, thresh, histogram, confidence);



		if (classify_flag == 1)
		{
			//update vehicle roi
			int dw = abs((R_label - 0.5*result.cols) - (0.5*result.cols - L_label));
			double scale = 0.2;

			if (dw < height / 2)
			{
				scale = 1;
				if ((R_label - 0.5*result.cols) > (0.5*result.cols - L_label))
				{
					right_bottom.x = left_top.x + R_label;
					left_top.x = left_top.x + L_label - dw*scale;
					right_bottom.y = center.y;
					left_top.y = right_bottom.y + (-height * 1);

				}
				else
				{
					right_bottom.x = left_top.x + R_label + dw*scale;
					left_top.x = left_top.x + L_label;;
					right_bottom.y = center.y;
					left_top.y = right_bottom.y + (-height * 1);


				}
			}
			else
			{
				right_bottom.x = left_top.x + R_label + dw*scale + 5;
				left_top.x = left_top.x + L_label - dw*scale - 5;
				right_bottom.y = center.y;
				left_top.y = right_bottom.y + (-height * 1);
			}

			return 1;
		}
		else
			return 0;
	}
	else
	{
		return 0;
	}

}

//==================================================
//=================== feature ======================
//==================================================

//LBP
vector<int> Vehicle::LBP(Mat img)
{
	vector<int>h;

	h.assign(256, 0);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			h[int(data[x])]++;
		}
	}

	return h;
}

Mat Vehicle::LBP_graph(Mat img)
{
	vector<int>tmp;
	tmp.assign(8, 0);
	Mat result = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));

	for (int y = 1; y < img.rows - 1; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		uchar *data_out = result.ptr<uchar>(y);
		for (int x = 1; x < img.cols - 1; x++)
		{
			if (data[x - img.cols - 1] > data[x])
				tmp[7] = 1;
			else
				tmp[7] = 0;

			if (data[x - img.cols] > data[x])
				tmp[6] = 1;
			else
				tmp[6] = 0;

			if (data[x - img.cols + 1] > data[x])
				tmp[5] = 1;
			else
				tmp[5] = 0;

			if (data[x + 1] > data[x])
				tmp[4] = 1;
			else
				tmp[4] = 0;

			if (data[x + img.cols + 1] > data[x])
				tmp[3] = 1;
			else
				tmp[3] = 0;

			if (data[x + img.cols] > data[x])
				tmp[2] = 1;
			else
				tmp[2] = 0;

			if (data[x + img.cols - 1] > data[x])
				tmp[1] = 1;
			else
				tmp[1] = 0;

			if (data[x - 1] > data[x])
				tmp[0] = 1;
			else
				tmp[0] = 0;


			data_out[x] = int(pow(2, 7)*tmp[7] + pow(2, 6)*tmp[6] + pow(2, 5)*tmp[5] + pow(2, 4)*tmp[4] + pow(2, 3)*tmp[3] + pow(2, 2)*tmp[2] + pow(2, 1)*tmp[1] + tmp[0]);
		}
	}

	return result;
}






//LBP_H
vector<int> Vehicle::LBP_H(Mat img)
{
	vector<int>h;
	vector<int>histogram;
	int block = 4;	//一張圖分別切成4*4個block

	Mat block_img(img.rows / block, img.cols / block, CV_8UC1);
	for (int j = 0; j < block; j++)
	{
		for (int i = 0; i < block; i++)
		{
			for (int y = 0; y < img.rows / block; y++)
			{
				uchar* data = img.ptr<uchar>(j*(img.rows / block) + y);
				uchar* data_out = block_img.ptr<uchar>(y);
				for (int x = 0; x < img.cols / block; x++)
				{
					data_out[x] = data[i*(img.cols / block) + x];
				}
			}
			h = LBP(block_img);
			histogram.insert(histogram.end(), h.begin(), h.end());
		}
	}

	return histogram;
}


vector<Mat> Vehicle::MB_LBP_graph(Mat img)
{

	vector<Mat>result;

	//設定 scale
	vector<int>scale;
	scale.push_back(3);
	/*
	scale.push_back(6);
	scale.push_back(9);
	scale.push_back(12);
	scale.push_back(15);*/



	for (int s = 0; s < scale.size(); s++)
	{
		int cellsize = scale[s] / 3;
		int offset = cellsize / 2;

		Mat cellimage(img.rows, img.cols, CV_8UC1, Scalar(0));

		for (int j = offset; j < img.rows - offset; j++)
		{
			uchar *data_out = cellimage.ptr<uchar>(j);

			for (int i = offset; i < img.cols - offset; i++)
			{
				int temp = 0;
				for (int m = -offset; m < offset + 1; m++)
				{
					uchar *data = img.ptr<uchar>(m + j);
					for (int n = -offset; n < offset + 1; n++)
					{
						temp += data[n + i];

					}
				}
				temp /= cellsize*cellsize;

				data_out[i] = uchar(temp);
			}
		}

		Mat r = LBP_graph(cellimage);
		result.push_back(r);

	}
	return result;
}










Mat Vehicle::sobel(Mat img, int vertical_Horizontal, int T1)
{
	Mat result(img.rows, img.cols, CV_8UC1);

	vector<int> GX1; vector< vector<int> > GX;
	GX1.assign(3, 0); GX.assign(3, GX1);
	vector<int> GY1; vector< vector<int> > GY;
	GY1.assign(3, 0); GY.assign(3, GY1);

	double sumX = 0;
	double sumY = 0;
	double sum;
	double gg;
	double max_sumX = 0;
	double max_sum = 0;



	//sobel x 垂直運算子
	GX[0][0] = -3; GX[1][0] = 0; GX[2][0] = 3;
	GX[0][1] = -10; GX[1][1] = 0; GX[2][1] = 10;
	GX[0][2] = -3; GX[1][2] = 0; GX[2][2] = 3;

	//sobel y 水平運算子
	GY[0][0] = -3; GY[1][0] = -10; GY[2][0] = -3;
	GY[0][1] = 0; GY[1][1] = 0;  GY[2][1] = 0;
	GY[0][2] = 3; GY[1][2] = 10; GY[2][2] = 3;

	/*
	//sobel x 垂直運算子
	GX[0][0] = -1; GX[1][0] = 0; GX[2][0] = 1;
	GX[0][1] = -2; GX[1][1] = 0; GX[2][1] = 2;
	GX[0][2] = -1; GX[1][2] = 0; GX[2][2] = 1;

	//sobel y 水平運算子
	GY[0][0] = -1; GY[1][0] = -2; GY[2][0] = -1;
	GY[0][1] = 0; GY[1][1] = 0;  GY[2][1] = 0;
	GY[0][2] = 1; GY[1][2] = 2; GY[2][2] = 1;*/



	vector<double> tmp1; vector< vector<double> > tmp;
	tmp1.assign(img.rows + 1, 0); tmp.assign(img.cols + 1, tmp1);

	vector<double> grad1; vector< vector<double> > grad;
	grad1.assign(img.rows + 1, 0); grad.assign(img.cols + 1, grad1);

	vector<double> l_mag1; vector< vector<double> > l_mag;
	l_mag1.assign(img.rows + 1, 0); l_mag.assign(img.cols + 1, l_mag1);


	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			tmp[x][y] = data[x];
		}
	}


	//sobel Algorithm
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			sumX = 0;
			sumY = 0;
			if (x < 1 || x > img.cols - 2)
				sumX = 0;
			else if (y < 1 || y > img.rows - 1)
				sumY = 0;
			else
			{

				if (vertical_Horizontal == 0)
				{

					//Convolution x
					for (int j = -1; j <= 1; j++)
					{
						for (int i = -1; i <= 1; i++)
						{
							sumX = sumX + GX[i + 1][j + 1] * tmp[x + i][y + j];
						}
					}
				}
				else
				{
					//Convolution y
					for (int j = -1; j <= 1; j++)
					{
						for (int i = -1; i <= 1; i++)
						{
							sumY = sumY + GY[i + 1][j + 1] * tmp[x + i][y + j];
						}
					}

				}
			}

			//gg = atan2(abs(sumY), abs(sumX));
			//sum = sqrt(pow(sumX,2) + pow(sumY,2));

			sum = abs(sumX) + abs(sumY);

			//grad[x][y] = gg;
			l_mag[x][y] = sum;

			if (sum > max_sum)
				max_sum = sum;

		}
	}

	int pixel = 0;
	for (int y = 0; y < img.rows; y++)
	{
		uchar *data_out = result.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			pixel = l_mag[x][y] / max_sum * 255;
			if (pixel > T1)
			{
				data_out[x] = 255;
			}
			else
			{
				data_out[x] = 0;

			}

		}
	}



	return result;
}


Mat Vehicle::gray(Mat img)
{
	Mat result(img.rows, img.cols, CV_8UC1);

	//GRAY
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img.ptr<Vec3b>(y);
		uchar *data_out = result.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][0] * 0.114 + data[x][1] * 0.587 + data[x][0] * 0.299;
		}
	}

	return result;
}


//==================================================
//============== Vehicle detection =================
//==================================================

//cumulative density function 
//累積密度函數
Mat Vehicle::CDF(Mat img, double T_scale)
{
	vector<double>cum;
	cum.assign(256, 0);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			cum[data[x]]++;
		}
	}

	int threshold;
	int t = (img.cols*img.rows)*T_scale;//調整多少陰影%當作門檻值
	for (int i = 0; i < cum.size() - 1; i++)
	{
		cum[i + 1] += cum[i];
		if (cum[i + 1] > t)
		{
			threshold = i;
			break;
		}
	}


	Mat result(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		uchar *data_out = result.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data[x] < threshold&&data[x] < 50)//陰影像素低於80
			{
				data_out[x] = 255;
			}
			else
			{
				data_out[x] = 0;
			}
		}
	}

	//cout << "threshold_CDF: " << threshold << endl;
	//型態學(侵蝕膨脹)

	erode(result, result, Mat());
	erode(result, result, Mat());
	dilate(result, result, Mat());
	dilate(result, result, Mat());
	dilate(result, result, Mat());
	dilate(result, result, Mat());
	erode(result, result, Mat());
	erode(result, result, Mat());
	return result;
}




int label = 1;//標籤
vector<int> rtabel;
vector<int> imagee1;
vector< vector<int> > imagee;

void Vehicle::resolve(int a, int b, int label)//校正關係(合併標籤)
{

	int temp;
	if (rtabel[a] < rtabel[b])
	{
		temp = rtabel[b];
		rtabel[b] = rtabel[a];
		for (int i = 1; i <= label; i++)
		{
			if (rtabel[i] == temp)
			{
				rtabel[i] = rtabel[b];
			}
		}
	}
	else if (rtabel[b] < rtabel[a])
	{
		temp = rtabel[a];
		rtabel[a] = rtabel[b];
		for (int i = 1; i <= label; i++)
		{
			if (rtabel[i] == temp)
			{
				rtabel[i] = rtabel[a];
			}
		}
	}
}




vector<bounding_box> Vehicle::vehicle_detection(Mat img, double h_w_scale_min, double h_w_scale_max, double limit_density)
{

	imagee1.assign(img.rows + 1, 0);
	imagee.assign(img.cols + 1, imagee1);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (x != 0 && y != 0 && (data[x] == 255))
			{
				imagee[x][y] = 1;

			}
			else
			{
				imagee[x][y] = 0;
			}
		}
	}


	rtabel.push_back(0);
	rtabel.push_back(1);

	//first_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//fast_connect_component
			int c1, c2, c3, c4;
			if (x == 0 || y == 0)
			{
				imagee[x][y] == 0;
			}

			else if (imagee[x][y] != 0)
			{
				c1 = imagee[x - 1][y];
				c2 = imagee[x - 1][y - 1];
				c3 = imagee[x][y - 1];
				c4 = imagee[x + 1][y - 1];

				if (c3 != 0)
				{
					imagee[x][y] = c3;
				}
				else if (c1 != 0)
				{
					imagee[x][y] = c1;
					if (c4 != 0)//記錄此標籤關係，第二次掃描進行校正
					{
						resolve(c4, c1, label);
					}
				}
				else if (c2 != 0)
				{
					imagee[x][y] = c2;
					if (c4 != 0)
					{
						resolve(c2, c4, label);
					}
				}
				else if (c4 != 0)
				{
					imagee[x][y] = c4;
				}
				else
				{
					imagee[x][y] = label;
					label = label + 1;
					rtabel.push_back(label);

				}
			}
		}
	}

	vector<int>table;
	table.assign(rtabel.size(), 0);

	//bounding_box
	vector<bounding_box>b_b;
	bounding_box b_;
	b_b.assign(table.size(), b_);

	//two_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			int l = imagee[x][y];
			if (l != 0)
			{
				imagee[x][y] = rtabel[imagee[x][y]];
				table[imagee[x][y]]++;

				l = imagee[x][y];
				b_b[l].center += Point(x, y);//計算陰影中心累加
				if (x < b_b[l].left_top_x)
					b_b[l].left_top_x = x;
				else if (x > b_b[l].right_bottom_x)
					b_b[l].right_bottom_x = x;

				if (y < b_b[l].left_top_y)
					b_b[l].left_top_y = y;
				else if (y > b_b[l].right_bottom_y)
					b_b[l].right_bottom_y = y;
			}
		}
	}



	vector<int>table1;
	for (int i = 1; i < table.size(); i++)
	{
		if (table[i] > 200)// && table[i] < 15000)//標籤總pixels大於和小於多少門檻值
		{
			table1.push_back(i);
		}

	}

	//陰影長寬比例 & density必須大於0.5以上

	vector<bounding_box>result_bounding_box;
	bounding_box t;
	for (int i = 0; i < table1.size(); i++)
	{
		double scale = double(b_b[table1[i]].right_bottom_y - b_b[table1[i]].left_top_y) / double(b_b[table1[i]].right_bottom_x - b_b[table1[i]].left_top_x);

		if (scale > h_w_scale_min&&scale < h_w_scale_max)
		{


			t.center = b_b[table1[i]].center / table[table1[i]];//計算陰影中心

			//中間到底部密度
			int width = (b_b[table1[i]].right_bottom_x - b_b[table1[i]].left_top_x);
			int height = (b_b[table1[i]].right_bottom_y - t.center.y);
			int total = 0;
			for (int y = t.center.y; y < b_b[table1[i]].right_bottom_y; y++)//t.center.y  b_b[table1[i]].left_top_y
			{
				uchar *data = img.ptr<uchar>(y);
				for (int x = b_b[table1[i]].left_top_x; x < b_b[table1[i]].right_bottom_x; x++)
				{
					if (data[x] == 255)
						total++;
				}
			}
			double density = total / double(width*height);


			//上面到底部密度
			height = (t.center.y - b_b[table1[i]].left_top_y);
			total = 0;
			for (int y = b_b[table1[i]].left_top_y; y < t.center.y; y++)
			{
				uchar *data = img.ptr<uchar>(y);
				for (int x = b_b[table1[i]].left_top_x; x < b_b[table1[i]].right_bottom_x; x++)
				{
					if (data[x] == 255)
						total++;
				}
			}
			double density1 = total / double(width*height);


			if (density > limit_density || density1 > limit_density)//判斷density
			{


				t.left_top_x = t.center.x - width*0.7;
				t.left_top_y = t.center.y - width*0.85;
				t.right_bottom_x = t.center.x + width*0.7;
				t.right_bottom_y = b_b[table1[i]].right_bottom_y;

				/*
				t.left_top_x = b_b[table1[i]].left_top_x;//陰影
				t.left_top_y = b_b[table1[i]].left_top_y;
				t.right_bottom_x = b_b[table1[i]].right_bottom_x;
				t.right_bottom_y = b_b[table1[i]].right_bottom_y;*/

				result_bounding_box.push_back(t);

			}

		}
	}


	label = 1;
	rtabel.clear();
	return result_bounding_box;
}






//==================================================
//============== taillight detection ===============
//==================================================

vector<bounding_box> Vehicle::connect_compoent(Mat img)
{
	imagee1.assign(img.rows + 1, 0);
	imagee.assign(img.cols + 1, imagee1);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (x != 0 && y != 0 && (data[x] == 255))
			{
				imagee[x][y] = 1;

			}
			else
			{
				imagee[x][y] = 0;
			}
		}
	}

	rtabel.push_back(0);
	rtabel.push_back(1);

	//first_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//fast_connect_component
			int c1, c2, c3, c4;

			if (x == 0 || y == 0)
			{
				imagee[x][y] == 0;
			}
			else if (imagee[x][y] != 0)
			{
				c1 = imagee[x - 1][y];
				c2 = imagee[x - 1][y - 1];
				c3 = imagee[x][y - 1];
				c4 = imagee[x + 1][y - 1];

				if (c3 != 0)
				{
					imagee[x][y] = c3;
				}
				else if (c1 != 0)
				{
					imagee[x][y] = c1;
					if (c4 != 0)//記錄此標籤關係，第二次掃描進行校正
					{
						resolve(c4, c1, label);
					}
				}
				else if (c2 != 0)
				{
					imagee[x][y] = c2;
					if (c4 != 0)
					{
						resolve(c2, c4, label);
					}
				}
				else if (c4 != 0)
				{
					imagee[x][y] = c4;
				}
				else
				{
					imagee[x][y] = label;
					label = label + 1;
					rtabel.push_back(label);

				}


			}
		}
	}

	vector<int>table;
	table.assign(rtabel.size(), 0);
	//bounding_box
	vector<bounding_box>b_b;
	bounding_box b_;
	b_b.assign(table.size(), b_);

	//two_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			int l = imagee[x][y];
			if (l != 0)
			{
				imagee[x][y] = rtabel[imagee[x][y]];
				table[imagee[x][y]]++;//area

				l = imagee[x][y];
				b_b[l].center += Point(x, y);//計算累加座標
				if (x < b_b[l].left_top_x)
					b_b[l].left_top_x = x;
				else if (x > b_b[l].right_bottom_x)
					b_b[l].right_bottom_x = x;

				if (y < b_b[l].left_top_y)
					b_b[l].left_top_y = y;
				else if (y > b_b[l].right_bottom_y)
					b_b[l].right_bottom_y = y;
			}
		}
	}


	vector<int>table1;
	for (int i = 1; i < table.size(); i++)
	{
		if (table[i] > 20)//標籤總pixels大於和小於多少門檻值
		{
			table1.push_back(i);
		}
	}



	vector<bounding_box>result_bounding_box;
	bounding_box t;

	for (int i = 0; i < table1.size(); i++)
	{
		t.center = b_b[table1[i]].center / table[table1[i]];//計算中心
		t.area = table[table1[i]];
		t.left_top_x = b_b[table1[i]].left_top_x;
		t.left_top_y = b_b[table1[i]].left_top_y;
		t.right_bottom_x = b_b[table1[i]].right_bottom_x;
		t.right_bottom_y = b_b[table1[i]].right_bottom_y;

		result_bounding_box.push_back(t);

	}


	label = 1;
	rtabel.clear();
	return result_bounding_box;
}


bool big(light_pair a, light_pair b)
{
	return a.confidence > b.confidence;
}



int Vehicle::boundingBOX_area_compute(bounding_box r1)
{
	int w1 = abs(r1.right_bottom_x - r1.left_top_x);
	int h1 = abs(r1.right_bottom_y - r1.left_top_y);
	return w1*h1;
}


double Vehicle::HS(bounding_box r1, bounding_box r2)
{
	double h1 = r1.right_bottom_y - r1.left_top_y;
	double h2 = r2.right_bottom_y - r2.left_top_y;

	double score = (1 - (abs(h1 - h2) / (1 * (h1 + h2)))) * 100;

	return score;
}


double Vehicle::AS(bounding_box r1, bounding_box r2)
{
	double a1 = r1.area;
	double a2 = r2.area;

	double score = (1 - (abs(a1 - a2) / (1 * (a1 + a2)))) * 100;

	return score;
}

double Vehicle::WS(bounding_box r1, bounding_box r2)
{
	double w1 = r1.right_bottom_x - r1.left_top_x;
	double w2 = r2.right_bottom_x - r2.left_top_x;


	double score = (1 - (abs(w1 - w2) / (1 * (w1 + w2)))) * 100;

	return score;

}


double Vehicle::AR(bounding_box r1, bounding_box r2)
{
	double w1 = r1.right_bottom_x - r1.left_top_x;
	double w2 = r2.right_bottom_x - r2.left_top_x;

	double aspect_ratio = abs(r2.center.x - r1.center.x) / (0.5*(w1 + w2));

	return aspect_ratio;
}


Mat Vehicle::otsu(Mat img, int T)
{
	vector<double>h;
	h.assign(256, 0);
	int max1 = 0, max2 = 0;
	int threshold;
	double sd = 0.0, sd1 = 0.0;

	for (int y = 0; y < img.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			h[data[x]]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		h[i] = h[i] / (img.rows*img.cols);
	}

	double u00 = 0, u11 = 0;
	double u0 = 0, u1 = 0, u = 0;
	double w0 = 0, w1 = 0, u0tmp = 0, u1tmp = 0;
	double sum = 0, sum1 = 0, s = 0;
	double min = INFINITY;
	double tmp1 = 0, tmp2 = 0, m_tmp1 = 0, m_tmp2 = 0;
	for (int t = 0; t < 256; t++)
	{
		w0 = 0.0; u0tmp = 0.0; w1 = 0.0; u1tmp = 0.0;
		sum = 0.0; sum1 = 0.0;
		tmp1 = 0, tmp2 = 0;
		for (int i = 0; i <= t; i++)
		{
			w0 += h[i];
			u0tmp += i* h[i];
			if (h[i] > tmp1)
			{
				tmp1 = h[i];
				m_tmp1 = i;
			}
		}
		for (int i = t + 1; i <= 255; i++)
		{
			w1 += h[i];
			u1tmp += i* h[i];
			if (h[i] > tmp2)
			{
				tmp2 = h[i];
				m_tmp2 = i;
			}
		}

		u0 = u0tmp / w0;//平均值
		u1 = u1tmp / w1;

		for (int i = 0; i <= t; i++)
		{
			sum = sum + pow((u0 - i), 2)*h[i];
		}
		for (int i = t + 1; i <= 255; i++)
		{
			sum1 = sum1 + pow((u1 - i), 2)*h[i];
		}

		s = (sum + sum1);

		if (s <= min)
		{
			min = s;
			threshold = t;
			max1 = m_tmp1;//峰值最大 
			max2 = m_tmp2;
			u00 = u0;//平均值
			u11 = u1;
			sd = sqrt(sum / w0);//標準差
			sd1 = sqrt(sum1 / w1);
		}
	}



	Mat result = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		uchar* data_out = result.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data[x] >= threshold&&data[x] >= T)
				data_out[x] = 255;
			else
				data_out[x] = 0;
		}
	}



	return result;

}



int Vehicle::taillight_detection(Mat img, vector<bounding_box >&pair_bounding_box, vector<bounding_box >&light_bounding_box)
{
	//otsu Lab 色彩空間
	Mat img_lab;
	cvtColor(img, img_lab, CV_BGR2Lab);
	Mat result_lab_a = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_lab.ptr<Vec3b>(y);
		uchar *data_out = result_lab_a.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][1];
		}
	}
	result_lab_a = otsu(result_lab_a, 130);
	dilate(result_lab_a, result_lab_a, Mat());
	erode(result_lab_a, result_lab_a, Mat());
	//imshow("result_lab_a", result_lab_a);



	//otsu Ycrcb 色彩空間
	Mat result_ycrcb_cr = Mat(img.rows, img.cols, CV_8UC1);
	Mat img_ycrcb;
	cvtColor(img, img_ycrcb, CV_BGR2YCrCb);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_ycrcb.ptr<Vec3b>(y);
		uchar *data_out = result_ycrcb_cr.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][1];
		}
	}
	result_ycrcb_cr = otsu(result_ycrcb_cr, 130);
	dilate(result_ycrcb_cr, result_ycrcb_cr, Mat());
	erode(result_ycrcb_cr, result_ycrcb_cr, Mat());
	//imshow("result_ycrcb_cr", result_ycrcb_cr);






	//Lab&&Ycrvcb 
	Mat result_lab_ycrcb = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		uchar *data_lab_a = result_lab_a.ptr<uchar>(y);
		uchar *data_ycrcb_cr = result_ycrcb_cr.ptr<uchar>(y);
		uchar *data_out = result_lab_ycrcb.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data_lab_a[x] == 255 && data_ycrcb_cr[x] == 255)
				data_out[x] = 255;
			else
				data_out[x] = 0;
		}
	}
	dilate(result_lab_ycrcb, result_lab_ycrcb, Mat());
	erode(result_lab_ycrcb, result_lab_ycrcb, Mat());
	//imshow("result_lab_ycrcb", result_lab_ycrcb);




	vector<bounding_box > ccl_box = connect_compoent(result_lab_ycrcb);
	light_bounding_box = ccl_box;


	vector<light_pair>pair;
	light_pair t;

	double area_total = 0;
	for (int i = 0; i < ccl_box.size(); i++)
	{
		area_total += ccl_box[i].area;
	}

	for (int i = 0; i < ccl_box.size(); i++)
	{
		for (int j = i + 1; j < ccl_box.size(); j++)
		{

			double x_distance = abs(ccl_box[j].center.x - ccl_box[i].center.x);

			int h = abs(ccl_box[j].center.y - ccl_box[i].center.y);
			int hi = abs(ccl_box[i].right_bottom_y - ccl_box[i].left_top_y);
			int hj = abs(ccl_box[j].right_bottom_y - ccl_box[j].left_top_y);

			int Gh = (h <= 1.2*hi) || (h <= 1.2*hj) ? 1 : 0;
			int Gd = AR(ccl_box[i], ccl_box[j]) >= 1 ? 1 : 0;
			int Gx = x_distance >= 0.33*img.cols ? 1 : 0;
			int Gy_top = (ccl_box[j].center.y + ccl_box[i].center.y) / 2 >= 0.25*img.rows ? 1 : 0;
			int Gy_bottom = (ccl_box[j].center.y + ccl_box[i].center.y) / 2 <= 0.75*img.rows ? 1 : 0;

			if (Gh && Gd && Gx && Gy_bottom && Gy_top)
			{
				double area = (ccl_box[i].area + ccl_box[j].area);
				t.i = i;
				t.j = j;
				t.confidence = area;
				pair.push_back(t);

			}


		}
	}



	sort(pair.begin(), pair.end(), big);
	bounding_box pp;
	if (pair.size() > 0)
	{
		int tmp;
		if (ccl_box[pair[0].i].center.x > ccl_box[pair[0].j].center.x)
		{
			tmp = pair[0].i;
			pair[0].i = pair[0].j;
			pair[0].j = tmp;
		}

		pp.left_top_x = ccl_box[pair[0].i].left_top_x;
		pp.left_top_y = ccl_box[pair[0].i].left_top_y;
		pp.right_bottom_x = ccl_box[pair[0].i].right_bottom_x;
		pp.right_bottom_y = ccl_box[pair[0].i].right_bottom_y;
		pp.center = ccl_box[pair[0].i].center;
		pair_bounding_box.push_back(pp);
		pp.left_top_x = ccl_box[pair[0].j].left_top_x;
		pp.left_top_y = ccl_box[pair[0].j].left_top_y;
		pp.right_bottom_x = ccl_box[pair[0].j].right_bottom_x;
		pp.right_bottom_y = ccl_box[pair[0].j].right_bottom_y;
		pp.center = ccl_box[pair[0].j].center;
		pair_bounding_box.push_back(pp);

		return 1;
	}
	else
	{

		return 0;

	}
}
//======================================================================================================================================================
//====================================================              Nighttime             ==============================================================
//======================================================================================================================================================

//==================================================
//=================   Classify  ====================
//==================================================

int Vehicle_nighttime::classify(Mat img, vector< vector<classifier>>cascade_classifier, vector<double> thresh)
{

	Mat reize_img;
	resize(img, reize_img, Size(64, 64));
	reize_img = image_enhancement(reize_img);


	//adaboost classifier
	cvtColor(reize_img, reize_img, CV_BGR2GRAY);
	vector<Mat>MB_LBP_img = MB_LBP_graph(reize_img);


	double confidence;
	vector<double>histogram;
	for (int i = 0; i < MB_LBP_img.size(); i++)
	{
		Mat img = MB_LBP_img[i];
		vector<int>h = LBP_H(img);
		histogram.insert(histogram.end(), h.begin(), h.end());
	}

	int classify_flag = adaboost.predict(cascade_classifier, thresh, histogram, confidence);


	return classify_flag;
}



//==================================================
//=================== feature ======================
//==================================================

//LBP
vector<int> Vehicle_nighttime::LBP(Mat img)
{
	vector<int>h;

	h.assign(256, 0);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			h[int(data[x])]++;
		}
	}

	return h;
}

Mat Vehicle_nighttime::LBP_graph(Mat img)
{
	vector<int>tmp;
	tmp.assign(8, 0);
	Mat result = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));

	for (int y = 1; y < img.rows - 1; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		uchar *data_out = result.ptr<uchar>(y);
		for (int x = 1; x < img.cols - 1; x++)
		{
			if (data[x - img.cols - 1] > data[x])
				tmp[7] = 1;
			else
				tmp[7] = 0;

			if (data[x - img.cols] > data[x])
				tmp[6] = 1;
			else
				tmp[6] = 0;

			if (data[x - img.cols + 1] > data[x])
				tmp[5] = 1;
			else
				tmp[5] = 0;

			if (data[x + 1] > data[x])
				tmp[4] = 1;
			else
				tmp[4] = 0;

			if (data[x + img.cols + 1] > data[x])
				tmp[3] = 1;
			else
				tmp[3] = 0;

			if (data[x + img.cols] > data[x])
				tmp[2] = 1;
			else
				tmp[2] = 0;

			if (data[x + img.cols - 1] > data[x])
				tmp[1] = 1;
			else
				tmp[1] = 0;

			if (data[x - 1] > data[x])
				tmp[0] = 1;
			else
				tmp[0] = 0;


			data_out[x] = int(pow(2, 7)*tmp[7] + pow(2, 6)*tmp[6] + pow(2, 5)*tmp[5] + pow(2, 4)*tmp[4] + pow(2, 3)*tmp[3] + pow(2, 2)*tmp[2] + pow(2, 1)*tmp[1] + tmp[0]);
		}
	}

	return result;
}






//LBP_H
vector<int> Vehicle_nighttime::LBP_H(Mat img)
{
	vector<int>h;
	vector<int>histogram;
	int block = 4;	//一張圖分別切成4*4個block

	Mat block_img(img.rows / block, img.cols / block, CV_8UC1);
	for (int j = 0; j < block; j++)
	{
		for (int i = 0; i < block; i++)
		{
			for (int y = 0; y < img.rows / block; y++)
			{
				uchar* data = img.ptr<uchar>(j*(img.rows / block) + y);
				uchar* data_out = block_img.ptr<uchar>(y);
				for (int x = 0; x < img.cols / block; x++)
				{
					data_out[x] = data[i*(img.cols / block) + x];
				}
			}
			h = LBP(block_img);
			histogram.insert(histogram.end(), h.begin(), h.end());
		}
	}

	return histogram;
}


vector<Mat> Vehicle_nighttime::MB_LBP_graph(Mat img)
{

	vector<Mat>result;

	//設定 scale
	vector<int>scale;
	scale.push_back(3);


	for (int s = 0; s < scale.size(); s++)
	{
		int cellsize = scale[s] / 3;
		int offset = cellsize / 2;

		Mat cellimage(img.rows, img.cols, CV_8UC1, Scalar(0));

		for (int j = offset; j < img.rows - offset; j++)
		{
			uchar *data_out = cellimage.ptr<uchar>(j);

			for (int i = offset; i < img.cols - offset; i++)
			{
				int temp = 0;
				for (int m = -offset; m < offset + 1; m++)
				{
					uchar *data = img.ptr<uchar>(m + j);
					for (int n = -offset; n < offset + 1; n++)
					{
						temp += data[n + i];

					}
				}
				temp /= cellsize*cellsize;

				data_out[i] = uchar(temp);
			}
		}

		Mat r = LBP_graph(cellimage);
		result.push_back(r);

	}
	return result;
}


//==================================================
//============== Vehicle detection =================
//==================================================

Mat Vehicle_nighttime::otsu(Mat img, int T)
{
	vector<double>h;
	h.assign(256, 0);
	int max1 = 0, max2 = 0;
	int threshold;
	double sd = 0.0, sd1 = 0.0;

	for (int y = 0; y < img.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			h[data[x]]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		h[i] = h[i] / (img.rows*img.cols);
	}

	double u00 = 0, u11 = 0;
	double u0 = 0, u1 = 0, u = 0;
	double w0 = 0, w1 = 0, u0tmp = 0, u1tmp = 0;
	double sum = 0, sum1 = 0, s = 0;
	double min = INFINITY;
	double tmp1 = 0, tmp2 = 0, m_tmp1 = 0, m_tmp2 = 0;
	for (int t = 0; t < 256; t++)
	{
		w0 = 0.0; u0tmp = 0.0; w1 = 0.0; u1tmp = 0.0;
		sum = 0.0; sum1 = 0.0;
		tmp1 = 0, tmp2 = 0;
		for (int i = 0; i <= t; i++)
		{
			w0 += h[i];
			u0tmp += i* h[i];
			if (h[i] > tmp1)
			{
				tmp1 = h[i];
				m_tmp1 = i;
			}
		}
		for (int i = t + 1; i <= 255; i++)
		{
			w1 += h[i];
			u1tmp += i* h[i];
			if (h[i] > tmp2)
			{
				tmp2 = h[i];
				m_tmp2 = i;
			}
		}

		u0 = u0tmp / w0;//平均值
		u1 = u1tmp / w1;

		for (int i = 0; i <= t; i++)
		{
			sum = sum + pow((u0 - i), 2)*h[i];
		}
		for (int i = t + 1; i <= 255; i++)
		{
			sum1 = sum1 + pow((u1 - i), 2)*h[i];
		}

		s = (sum + sum1);

		if (s <= min)
		{
			min = s;
			threshold = t;
			max1 = m_tmp1;//峰值最大 
			max2 = m_tmp2;
			u00 = u0;//平均值
			u11 = u1;
			sd = sqrt(sum / w0);//標準差
			sd1 = sqrt(sum1 / w1);
		}
	}



	Mat result = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		uchar* data_out = result.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data[x] >= threshold&&data[x] >= T)
				data_out[x] = 255;
			else
				data_out[x] = 0;
		}
	}

	return result;

}


Mat Vehicle_nighttime::color_filter(Mat img, double limit_density)
{
	//otsu lab 色彩空間
	/*
	Mat img_lab;
	cvtColor(img, img_lab, CV_BGR2Lab);
	Mat result_lab_a = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
	Vec3b *data = img_lab.ptr<Vec3b>(y);
	uchar *data_out = result_lab_a.ptr<uchar>(y);
	for (int x = 0; x < img.cols; x++)
	{
	data_out[x] = data[x][1];
	}
	}
	result_lab_a = otsu(result_lab_a, 130);//threshold

	dilate(result_lab_a, result_lab_a, Mat());
	erode(result_lab_a, result_lab_a, Mat());
	//imshow("result_lab_a", result_lab_a);
	*/


	//otsu Ycrcb 色彩空間
	Mat img_ycrcb;
	cvtColor(img, img_ycrcb, CV_BGR2YCrCb);
	Mat result_ycrcb_cr = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_ycrcb.ptr<Vec3b>(y);
		uchar *data_out = result_ycrcb_cr.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][1];
		}
	}
	result_ycrcb_cr = otsu(result_ycrcb_cr, 140);//threshold
	dilate(result_ycrcb_cr, result_ycrcb_cr, Mat());
	erode(result_ycrcb_cr, result_ycrcb_cr, Mat());
	//imshow("result_ycrcb_cr", result_ycrcb_cr);




	Mat result_ycrcb_y = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_ycrcb.ptr<Vec3b>(y);
		uchar *data_out = result_ycrcb_y.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{

			if (data[x][0] >= 240)	//threshold
				data_out[x] = 255;
			else
				data_out[x] = 0;
		}
	}

	erode(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	erode(result_ycrcb_y, result_ycrcb_y, Mat());
	//imshow("result_ycrcb_y", result_ycrcb_y);


	//Lab&&Ycrvcb 
	//紅色
	Mat result_lab_ycrcb_r = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		//uchar *data1 = result_lab_a.ptr<uchar>(y);
		uchar *data2 = result_ycrcb_cr.ptr<uchar>(y);
		uchar *data_out = result_lab_ycrcb_r.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			//if (data1[x] == 255 || data2[x] == 255)
			if (data2[x] == 255)
				data_out[x] = 0;
			else
				data_out[x] = 255;
		}
	}
	erode(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	dilate(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	dilate(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	erode(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	//imshow("result_lab_ycrcb_r", result_lab_ycrcb_r);



	Mat result_lab_ycrcb_r1 = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	vector<bounding_box > ccl_box = connect_compoent(result_lab_ycrcb_r);
	for (int i = 0; i < ccl_box.size(); i++)
	{

		int w = abs(ccl_box[i].right_bottom_x - ccl_box[i].left_top_x);
		int h = abs(ccl_box[i].right_bottom_y - ccl_box[i].left_top_y);
		double density = ccl_box[i].area*1.0 / (w*h);
		//if (ccl_box[i].area<5000)
		if (density >= limit_density&&ccl_box[i].area < 5000)
		{
			for (int y = ccl_box[i].left_top_y - 1; y < ccl_box[i].right_bottom_y + 1; y++)
			{
				uchar *data = result_lab_ycrcb_r.ptr<uchar>(y);
				uchar *data1 = result_ycrcb_y.ptr<uchar>(y);
				uchar *data_out = result_lab_ycrcb_r1.ptr<uchar>(y);
				for (int x = ccl_box[i].left_top_x - 1; x < ccl_box[i].right_bottom_x + 1; x++)
				{
					if (data[x] == 255 && data1[x] == 255)
						data_out[x] = 255;
					else
						data_out[x] = 0;
				}
			}
		}

	}
	dilate(result_lab_ycrcb_r1, result_lab_ycrcb_r1, Mat());
	erode(result_lab_ycrcb_r1, result_lab_ycrcb_r1, Mat());




	//imshow("result_lab_ycrcb_r1", result_lab_ycrcb_r1);
	return result_lab_ycrcb_r1;
}

void Vehicle_nighttime::resolve(int a, int b, int label)//校正關係(合併標籤)
{

	int temp;
	if (rtabel[a] < rtabel[b])
	{
		temp = rtabel[b];
		rtabel[b] = rtabel[a];
		for (int i = 1; i <= label; i++)
		{
			if (rtabel[i] == temp)
			{
				rtabel[i] = rtabel[b];
			}
		}
	}
	else if (rtabel[b] < rtabel[a])
	{
		temp = rtabel[a];
		rtabel[a] = rtabel[b];
		for (int i = 1; i <= label; i++)
		{
			if (rtabel[i] == temp)
			{
				rtabel[i] = rtabel[a];
			}
		}
	}

}

vector<bounding_box> Vehicle_nighttime::connect_compoent(Mat img)
{
	imagee1.assign(img.rows + 1, 0);
	imagee.assign(img.cols + 1, imagee1);

	for (int y = 0; y < img.rows; y++)
	{
		uchar *data = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (x != 0 && y != 0 && (data[x] == 255))
			{
				imagee[x][y] = 1;

			}
			else
			{
				imagee[x][y] = 0;
			}
		}
	}

	rtabel.push_back(0);
	rtabel.push_back(1);

	//first_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//fast_connect_component
			int c1, c2, c3, c4;

			if (x == 0 || y == 0)
			{
				imagee[x][y] == 0;
			}
			else if (imagee[x][y] != 0)
			{
				c1 = imagee[x - 1][y];
				c2 = imagee[x - 1][y - 1];
				c3 = imagee[x][y - 1];
				c4 = imagee[x + 1][y - 1];

				if (c3 != 0)
				{
					imagee[x][y] = c3;
				}
				else if (c1 != 0)
				{
					imagee[x][y] = c1;
					if (c4 != 0)//記錄此標籤關係，第二次掃描進行校正
					{
						resolve(c4, c1, label);
					}
				}
				else if (c2 != 0)
				{
					imagee[x][y] = c2;
					if (c4 != 0)
					{
						resolve(c2, c4, label);
					}
				}
				else if (c4 != 0)
				{
					imagee[x][y] = c4;
				}
				else
				{
					imagee[x][y] = label;
					label = label + 1;
					rtabel.push_back(label);

				}


			}
		}
	}

	vector<int>table;
	table.assign(rtabel.size(), 0);
	//bounding_box
	vector<bounding_box>b_b;
	bounding_box b_;
	b_b.assign(table.size(), b_);

	//two_scan
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			int l = imagee[x][y];
			if (l != 0)
			{
				imagee[x][y] = rtabel[imagee[x][y]];
				table[imagee[x][y]]++;//area

				l = imagee[x][y];
				b_b[l].center += Point(x, y);//計算累加座標
				if (x < b_b[l].left_top_x)
					b_b[l].left_top_x = x;
				else if (x > b_b[l].right_bottom_x)
					b_b[l].right_bottom_x = x;

				if (y < b_b[l].left_top_y)
					b_b[l].left_top_y = y;
				else if (y > b_b[l].right_bottom_y)
					b_b[l].right_bottom_y = y;
			}
		}
	}


	vector<int>table1;
	for (int i = 1; i < table.size(); i++)
	{
		if (table[i] > 20)//標籤總pixels大於和小於多少門檻值
		{
			table1.push_back(i);
		}
	}



	vector<bounding_box>result_bounding_box;
	bounding_box t;

	for (int i = 0; i < table1.size(); i++)
	{
		t.center = b_b[table1[i]].center / table[table1[i]];//計算中心
		t.area = table[table1[i]];
		t.left_top_x = b_b[table1[i]].left_top_x;
		t.left_top_y = b_b[table1[i]].left_top_y;
		t.right_bottom_x = b_b[table1[i]].right_bottom_x;
		t.right_bottom_y = b_b[table1[i]].right_bottom_y;

		result_bounding_box.push_back(t);

	}


	label = 1;
	rtabel.clear();
	return result_bounding_box;
}


#define Gamma 3
Mat Vehicle_nighttime::image_enhancement(Mat img)
{

	Mat result(img.size(), CV_32FC3);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img.ptr<Vec3b>(y);
		Vec3f *data_out = result.ptr<Vec3f>(y);
		for (int x = 0; x < img.cols; x++)
		{

			//log
			data_out[x][0] = log(1 + data[x][0]);
			data_out[x][1] = log(1 + data[x][1]);
			data_out[x][2] = log(1 + data[x][2]);
			/*
			//Gamma 影像加強
			data_out[x][0] = pow(data[x][0], Gamma);
			data_out[x][1] = pow(data[x][1], Gamma);
			data_out[x][2] = pow(data[x][2], Gamma);*/

			/*
			data_out[x][0] = pow(log(1 + data[x][0]), Gamma);
			data_out[x][1] = pow(log(1 + data[x][1]), Gamma);
			data_out[x][2] = pow(log(1 + data[x][2]), Gamma);*/

		}
	}

	normalize(result, result, 0, 255, CV_MINMAX);
	convertScaleAbs(result, result);


	Mat result1(img.size(), CV_32FC3);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = result.ptr<Vec3b>(y);
		Vec3f *data_out = result1.ptr<Vec3f>(y);
		for (int x = 0; x < img.cols; x++)
		{
			//Gamma 影像加強
			data_out[x][0] = pow(data[x][0], Gamma);
			data_out[x][1] = pow(data[x][1], Gamma);
			data_out[x][2] = pow(data[x][2], Gamma);

		}
	}
	normalize(result1, result1, 0, 255, CV_MINMAX);
	convertScaleAbs(result1, result1);



	return result1;
}




double Vehicle_nighttime::HS(bounding_box r1, bounding_box r2)
{
	double h1 = r1.right_bottom_y - r1.left_top_y;
	double h2 = r2.right_bottom_y - r2.left_top_y;

	double score = (1 - (abs(h1 - h2) / (1 * (h1 + h2)))) * 100;

	return score;
}

double Vehicle_nighttime::WS(bounding_box r1, bounding_box r2)
{
	double w1 = r1.right_bottom_x - r1.left_top_x;
	double w2 = r2.right_bottom_x - r2.left_top_x;


	double score = (1 - (abs(w1 - w2) / (1 * (w1 + w2)))) * 100;

	return score;
}



double Vehicle_nighttime::AS(bounding_box r1, bounding_box r2)
{
	double a1 = r1.area;
	double a2 = r2.area;

	double score = (1 - (abs(a1 - a2) / (1 * (a1 + a2)))) * 100;

	return score;
}

double  Vehicle_nighttime::AR(bounding_box r1, bounding_box r2)
{
	double w1 = r1.right_bottom_x - r1.left_top_x;
	double w2 = r2.right_bottom_x - r2.left_top_x;

	double aspect_ratio = abs(r2.center.x - r1.center.x) / (0.5*(w1 + w2));

	return aspect_ratio;
}





bool sortfunction(bounding_box i, bounding_box j) { return (i.center.x < j.center.x); }//sort 函數

vector<Rect> Vehicle_nighttime::vehicle_detection(Point x1, Mat original_img, Mat img, vector< vector<classifier>>cascade_classifier, vector<double> thresh, double limit_density)
{

	Mat filter;
	filter = color_filter(img, limit_density);
	//imshow("candidate extraction", filter);

	vector<bounding_box > ccl_box = connect_compoent(filter);

	//排序車燈 ccl_box 由x中心座標左至右
	sort(ccl_box.begin(), ccl_box.end(), sortfunction);


	////draw候選車燈
	//for (int i = 0; i < ccl_box.size(); i++)
	//{
	//	rectangle(img, Point(ccl_box[i].left_top_x, ccl_box[i].left_top_y), Point(ccl_box[i].right_bottom_x, ccl_box[i].right_bottom_y), Scalar(0, 255, 0), 2);
	//	//circle(img, ccl_box[i].center, 2, Scalar(0, 0, 255), -1);
	//}


	vector<light_pair>result_pair;
	for (int i = 0; i < ccl_box.size(); i++)//left
	{
		vector<light_pair>pair;
		light_pair t;
		if (!ccl_box[i].use_flag)
		{
			for (int j = i + 1; j < ccl_box.size(); j++)//right
			{
				if (!ccl_box[j].use_flag)
				{
					double x_distance = abs(ccl_box[j].center.x - ccl_box[i].center.x);
					double h = abs(ccl_box[j].center.y - ccl_box[i].center.y);
					double hi = abs(ccl_box[i].right_bottom_y - ccl_box[i].left_top_y);
					double hj = abs(ccl_box[j].right_bottom_y - ccl_box[j].left_top_y);

					int Gh = (h <= 1.2*hi) && (h <= 1.2*hj) ? 1 : 0;
					int ar = AR(ccl_box[i], ccl_box[j]);
					int Gd = ar >= 2 & ar <= 14 ? 1 : 0;
					int Gx = (x_distance >= 60) && (x_distance < 150) ? 1 : 0;
					if (Gh == 1 && Gd == 1 && Gx == 1)
					{
						int left_label = i;
						int right_label = j;
						if (ccl_box[left_label].left_top_x < ccl_box[right_label].left_top_x)
						{
							left_label = i;
							right_label = j;
						}
						else
						{
							left_label = j;
							right_label = i;
						}
						Point center = Point(0.5*(ccl_box[left_label].left_top_x + ccl_box[right_label].right_bottom_x), 0.5*(ccl_box[left_label].left_top_y + ccl_box[right_label].right_bottom_y));
						int width = ccl_box[right_label].right_bottom_x - ccl_box[left_label].left_top_x;
						int offset_distance = 0.075*(width);
						Point offset = Point(offset_distance * 2, offset_distance);
						Point left_top = Point(ccl_box[left_label].left_top_x, (center.y - width / 2)) + x1 - offset;
						Point right_bottom = Point(ccl_box[right_label].right_bottom_x, (center.y + width / 2)) + x1 + offset;
						Rect roi(left_top, right_bottom);
						Mat c = original_img(roi);
						int  classify_flag = classify(c, cascade_classifier, thresh);
						if (classify_flag)
						{
							double hs_score = HS(ccl_box[i], ccl_box[j]);
							double ws_score = WS(ccl_box[i], ccl_box[j]);
							double as_score = AS(ccl_box[i], ccl_box[j]);
							double score = 0.3*hs_score + 0.2*ws_score + 0.5*as_score;

							if (score >= 30)
							{
								t.i = left_label;
								t.j = right_label;
								t.confidence = score;
								pair.push_back(t);
							}
						}
					}
				}
			}
		}

		double max = 0;
		double max_label = 0;
		for (int k = 0; k < pair.size(); k++)
		{
			if (pair[k].confidence > max)
			{
				max = pair[k].confidence;
				max_label = k;

			}
		}
		if (pair.size() >= max_label&&pair.size() > 0)
		{
			t.i = pair[max_label].i;
			t.j = pair[max_label].j;


			//已使用
			ccl_box[t.i].use_flag = true;
			ccl_box[t.j].use_flag = true;
			result_pair.push_back(t);

		}

	}


	vector<Rect>result_bounding_box;
	for (int i = 0; i < result_pair.size(); i++)
	{
		int left_label = result_pair[i].i;
		int right_label = result_pair[i].j;

		Point center = Point(0.5*(ccl_box[left_label].left_top_x + ccl_box[right_label].right_bottom_x), 0.5*(ccl_box[left_label].left_top_y + ccl_box[right_label].right_bottom_y));
		int width = ccl_box[right_label].right_bottom_x - ccl_box[left_label].left_top_x;
		int offset_distance = 0.075*(width);
		Point offset = Point(offset_distance * 2, offset_distance);
		Point left_top = Point(ccl_box[left_label].left_top_x, (center.y - width / 2)) + x1 - offset;
		Point right_bottom = Point(ccl_box[right_label].right_bottom_x, (center.y + width / 2)) + x1 + offset;


		//taillight
		//Point left_top = Point(ccl_box[left_label].left_top_x, ccl_box[left_label].left_top_y)+x1;
		//Point right_bottom = Point(ccl_box[right_label].right_bottom_x, ccl_box[right_label].right_bottom_y)+x1;


		Rect roi(left_top, right_bottom);

		result_bounding_box.push_back(roi);

	}


	return result_bounding_box;
}


//==================================================
//============== taillight detection ===============
//==================================================

int Vehicle_nighttime::taillight_detection(Mat img, vector<bounding_box >&pair_bounding_box, vector<bounding_box >&light_bounding_box)
{
	//otsu Lab 色彩空間
	Mat img_lab;
	cvtColor(img, img_lab, CV_BGR2Lab);
	Mat result_lab_a = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_lab.ptr<Vec3b>(y);
		uchar *data_out = result_lab_a.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][1];
		}
	}
	result_lab_a = otsu(result_lab_a, 0);
	//imshow("result_lab_a", result_lab_a);

	/*
	Mat result_lab_l = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
	Vec3b *data = img_lab.ptr<Vec3b>(y);
	uchar *data_out = result_lab_l.ptr<uchar>(y);
	for (int x = 0; x < img.cols; x++)
	{
	data_out[x] = data[x][0];
	}
	}
	result_lab_l = otsu(result_lab_l,0);*/
	//imshow("result_lab_l", result_lab_l);



	//otsu Ycrcb 色彩空間
	Mat img_ycrcb;
	cvtColor(img, img_ycrcb, CV_BGR2YCrCb);

	Mat result_ycrcb_cr = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_ycrcb.ptr<Vec3b>(y);
		uchar *data_out = result_ycrcb_cr.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][1];
		}
	}
	result_ycrcb_cr = otsu(result_ycrcb_cr, 0);
	//imshow("result_ycrcb_cr", result_ycrcb_cr);


	Mat result_ycrcb_y = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		Vec3b *data = img_ycrcb.ptr<Vec3b>(y);
		uchar *data_out = result_ycrcb_y.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			data_out[x] = data[x][0];
		}
	}
	result_ycrcb_y = otsu(result_ycrcb_y, 240);//設定亮度threshold
	erode(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	dilate(result_ycrcb_y, result_ycrcb_y, Mat());
	erode(result_ycrcb_y, result_ycrcb_y, Mat());
	//imshow("result_ycrcb_y", result_ycrcb_y);



	//Lab&&Ycrvcb 
	//紅色
	Mat result_lab_ycrcb_r = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
		uchar *data1 = result_lab_a.ptr<uchar>(y);
		uchar *data2 = result_ycrcb_cr.ptr<uchar>(y);
		uchar *data_out = result_lab_ycrcb_r.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (data1[x] == 255 || data2[x] == 255)
				data_out[x] = 255;
			else
				data_out[x] = 0;
		}
	}

	dilate(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	erode(result_lab_ycrcb_r, result_lab_ycrcb_r, Mat());
	//imshow("result_lab_ycrcb_r", result_lab_ycrcb_r);

	//亮度
	/*
	Mat result_lab_ycrcb_l = Mat(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++)
	{
	uchar *data1 = result_lab_l.ptr<uchar>(y);
	uchar *data2 = result_ycrcb_y.ptr<uchar>(y);
	uchar *data_out = result_lab_ycrcb_l.ptr<uchar>(y);
	for (int x = 0; x < img.cols; x++)
	{
	if (data1[x] == 255 && data2[x] == 255)
	data_out[x] = 255;
	else
	data_out[x] = 0;
	}
	}

	erode(result_lab_ycrcb_l, result_lab_ycrcb_l, Mat());*/
	//imshow("result_lab_ycrcb_l", result_lab_ycrcb_l);




	vector<bounding_box > ccl_box_result;
	bounding_box ccl;

	vector<bounding_box > ccl_box = connect_compoent(result_lab_ycrcb_r);
	for (int i = 0; i < ccl_box.size(); i++)
	{
		Point left_top(ccl_box[i].left_top_x, ccl_box[i].left_top_y);
		Point right_bottom(ccl_box[i].right_bottom_x, ccl_box[i].right_bottom_y);
		Rect roi(left_top, right_bottom);


		Mat c = result_lab_ycrcb_r(roi);
		Mat d = result_ycrcb_y(roi);

		Mat result_c = Mat(c.rows, c.cols, CV_8UC1);
		for (int y = 0; y < c.rows; y++)
		{
			uchar *data1 = c.ptr<uchar>(y);
			uchar *data2 = d.ptr<uchar>(y);
			uchar *data_out = result_c.ptr<uchar>(y);
			for (int x = 0; x < c.cols; x++)
			{
				if (data1[x] == 0 && data2[x] == 255)
					data_out[x] = 255;
				else
					data_out[x] = 0;
			}
		}

		vector<bounding_box >ccl_result_c = connect_compoent(result_c);
		Point x1 = left_top;
		for (int j = 0; j < ccl_result_c.size(); j++)
		{
			left_top = Point(ccl_result_c[j].left_top_x, ccl_result_c[j].left_top_y) + x1;
			right_bottom = Point(ccl_result_c[j].right_bottom_x, ccl_result_c[j].right_bottom_y) + x1;
			ccl.left_top_x = left_top.x;
			ccl.left_top_y = left_top.y;
			ccl.right_bottom_x = right_bottom.x;
			ccl.right_bottom_y = right_bottom.y;
			ccl.area = ccl_result_c[j].area;
			ccl.center = ccl_result_c[j].center + x1;

			ccl_box_result.push_back(ccl);
		}
	}



	light_bounding_box = ccl_box_result;


	vector<light_pair>pair;
	light_pair t;

	for (int i = 0; i < ccl_box_result.size(); i++)
	{
		for (int j = i + 1; j < ccl_box_result.size(); j++)
		{
			double x_distance = abs(ccl_box_result[j].center.x - ccl_box_result[i].center.x);

			int h = abs(ccl_box_result[j].center.y - ccl_box_result[i].center.y);
			int hi = abs(ccl_box_result[i].right_bottom_y - ccl_box_result[i].left_top_y);
			int hj = abs(ccl_box_result[j].right_bottom_y - ccl_box_result[j].left_top_y);


			int Gh = (h <= 1.2*hi) || (h <= 1.2*hj) ? 1 : 0;
			int Gd = AR(ccl_box_result[i], ccl_box_result[j]) >= 1 ? 1 : 0;
			int Gx = x_distance >= 0.33*img.cols ? 1 : 0;
			int Gy_top = (ccl_box_result[j].center.y + ccl_box_result[i].center.y) / 2 >= 0.25*img.rows ? 1 : 0;
			int Gy_bottom = (ccl_box_result[j].center.y + ccl_box_result[i].center.y) / 2 <= 0.75*img.rows ? 1 : 0;

			if (Gh && Gd && Gx && Gy_bottom && Gy_top)
			{
				double area = (ccl_box_result[i].area + ccl_box_result[j].area);
				t.i = i;
				t.j = j;
				t.confidence = area;
				pair.push_back(t);

			}

		}
	}



	sort(pair.begin(), pair.end(), big);
	bounding_box pp;
	if (pair.size() > 0)
	{
		int tmp;
		if (ccl_box_result[pair[0].i].center.x > ccl_box_result[pair[0].j].center.x)
		{
			tmp = pair[0].i;
			pair[0].i = pair[0].j;
			pair[0].j = tmp;
		}

		pp.left_top_x = ccl_box_result[pair[0].i].left_top_x;
		pp.left_top_y = ccl_box_result[pair[0].i].left_top_y;
		pp.right_bottom_x = ccl_box_result[pair[0].i].right_bottom_x;
		pp.right_bottom_y = ccl_box_result[pair[0].i].right_bottom_y;
		pp.center = ccl_box_result[pair[0].i].center;
		pair_bounding_box.push_back(pp);
		pp.left_top_x = ccl_box_result[pair[0].j].left_top_x;
		pp.left_top_y = ccl_box_result[pair[0].j].left_top_y;
		pp.right_bottom_x = ccl_box_result[pair[0].j].right_bottom_x;
		pp.right_bottom_y = ccl_box_result[pair[0].j].right_bottom_y;
		pp.center = ccl_box_result[pair[0].j].center;
		pair_bounding_box.push_back(pp);

		return 1;
	}
	else
	{

		return 0;
	}

}












//======================================================================================================================================================
//====================================================                Signal              ==============================================================
//======================================================================================================================================================

//==================================================
//==============  Signal recognition ===============
//==================================================

vector<double>Signal::featureTOfrequency(vector<double>hsv_v)
{
	int frame_sample = hsv_v.size();
	vector<double>hsv_v_frequency;
	hsv_v_frequency.assign(frame_sample, 0);
	Mat result = Mat(1, frame_sample, CV_32FC1, Scalar(0));
	for (int y = 0; y < result.rows; y++)
	{
		float  *data = result.ptr<float>(y);
		for (int x = 0; x < result.cols; x++)
		{
			data[x] = hsv_v[x];
		}
	}

	Mat planes[] = { Mat_<float>(result), Mat::zeros(result.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);//分成 實部planes[0]  虛部planes[1] 
	magnitude(planes[0], planes[1], planes[0]);//振幅


	for (int y = 0; y < planes[0].rows; y++)
	{
		float  *data = planes[0].ptr<float >(y);
		for (int x = 0; x < planes[0].cols; x++)
		{
			hsv_v_frequency[x] = (data[x]);
		}
	}

	return hsv_v_frequency;
}

//normalize 0~1
vector<double>Signal::normalize(vector<double>hsv_v)
{
	vector<double>new_hsv_v;
	new_hsv_v.assign(hsv_v.size(), 0);
	double max = 0;
	double min = INT_MAX;
	double average = 0;

	for (int k = 0; k < hsv_v.size(); k++)
	{
		if (hsv_v[k] > max)
			max = hsv_v[k];

		if (hsv_v[k] < min)
			min = hsv_v[k];

		average += hsv_v[k];
	}
	average = average / hsv_v.size();


	for (int k = 0; k < hsv_v.size(); k++)
	{
		new_hsv_v[k] = (hsv_v[k] - average) / (max - min);
	}

	return new_hsv_v;
}


double Signal::compute_intensity(Mat img, bounding_box bounding_box)
{
	Point left_top = Point(bounding_box.left_top_x, bounding_box.left_top_y);
	Point right_bottom = Point(bounding_box.right_bottom_x, bounding_box.right_bottom_y);
	Rect roi(left_top, right_bottom);
	Mat roi_img = img(roi);
	double average_v = 0;

	/*
	Mat roi_img_gray;
	cvtColor(roi_img, roi_img_gray, CV_BGR2GRAY);
	for (int y = 0; y < roi_img_gray.rows; y++)
	{
		uchar *data = roi_img_gray.ptr<uchar>(y);
		for (int x = 0; x < roi_img_gray.cols; x++)
		{
			average_v += data[x];
		}
	}
	*/


	Mat roi_img_hsv;
	cvtColor(roi_img, roi_img_hsv, CV_BGR2HSV);

	for (int y = 0; y < roi_img_hsv.rows; y++)
	{
		Vec3b *data = roi_img_hsv.ptr<Vec3b>(y);
		for (int x = 0; x < roi_img_hsv.cols; x++)
		{
			average_v += data[x][2];
		}
	}



	//average_v = average_v / (roi_img.rows*roi_img.cols);
	//average_v = average_v / (img.rows*img.cols);

	return average_v;
}






//recognition_signal
//0:燈不亮	1:左轉	2:右轉	3.危險警告	4.煞車  5.左轉+煞車  6.右轉+煞車  7.警告+煞車
void Signal::signal_recognition(int daytime_nighttime,int frame_sample, int label, int signal_state_pre ,int &signal_state, vector<double >intensity_L, vector<double >intensity_R, vector<bounding_box >pair_bounding_box, vector<bounding_box >light_bounding_box, vector< vector<classifier>>cascade_classifier, vector<double> thresh)
{

	//Brake signal detection
	int brake_classify = 0;
	double threshold_scale = 1.0;
	if (daytime_nighttime == 0)
		threshold_scale = 1.5;
	else
		threshold_scale = 1;

	if (pair_bounding_box.size() == 2)
	{
		int w1 = abs(pair_bounding_box[0].right_bottom_x - pair_bounding_box[0].left_top_x);
		int w2 = abs(pair_bounding_box[1].right_bottom_x - pair_bounding_box[1].left_top_x);
		int h1 = abs(pair_bounding_box[0].right_bottom_y - pair_bounding_box[0].left_top_y);
		int h2 = abs(pair_bounding_box[1].right_bottom_y - pair_bounding_box[1].left_top_y);


		//偵測第三煞車燈
		int third_brake = 0;
		for (int i = 0; i < light_bounding_box.size(); i++)
		{
			if ((pair_bounding_box[0].center.y > light_bounding_box[i].center.y) && (pair_bounding_box[1].center.y > light_bounding_box[i].center.y))
			{
				if ((pair_bounding_box[0].right_bottom_x < light_bounding_box[i].left_top_x) && (pair_bounding_box[1].left_top_x > light_bounding_box[i].right_bottom_x))
				{
					double scale = (light_bounding_box[i].right_bottom_y - light_bounding_box[i].left_top_y)*1.0 / (light_bounding_box[i].right_bottom_x - light_bounding_box[i].left_top_x);//煞車燈一定是長方形
					int brake_h = abs(light_bounding_box[i].right_bottom_y - light_bounding_box[i].left_top_y);

					if ((scale <= threshold_scale) && (brake_h < h1) && (brake_h < h2))
						third_brake = 1;
				}
			}
		}

		//車尾燈強度比較
		int intensity_brake = 0;
		int pre_label=0;
		if (label == 0)
			pre_label = frame_sample - 1;
		else
			pre_label =label - 1;

		if (intensity_L[label] > intensity_L[pre_label] && intensity_R[label] > intensity_R[pre_label])
		{
			intensity_brake = 1;
		}

		if (signal_state_pre >= 4)
		{
			if (third_brake)
				brake_classify = 1;
		}
		else
		{
			if (intensity_brake)
			{
				if (third_brake)
					brake_classify = 1;
			}

		}



	
	}



	//Turn signal detection
	int taillight_classify_L = 0;
	int taillight_classify_R = 0;

	vector<double >hsv_v_L_normalize = normalize(intensity_L);
	vector<double >hsv_v_R_normalize = normalize(intensity_R);

	vector<double >hsv_v_frequency_L = featureTOfrequency(hsv_v_L_normalize);
	vector<double >hsv_v_frequency_R = featureTOfrequency(hsv_v_R_normalize);

	double confidence_value;
	taillight_classify_L = adaboost.predict(cascade_classifier, thresh, hsv_v_frequency_L, confidence_value);
	taillight_classify_R = adaboost.predict(cascade_classifier, thresh, hsv_v_frequency_R, confidence_value);


	if (taillight_classify_L&&taillight_classify_R)
		signal_state = 3 + brake_classify * 4;
	else if (taillight_classify_L)
		signal_state = 1 + brake_classify * 4;
	else if (taillight_classify_R)
		signal_state = 2 + brake_classify * 4;
	else
		signal_state = 0 + brake_classify * 4;

}



