/*
A Real-Time Forward Vehicle Detection and
Signal Recognition System in All-Weather Situations

Title: 全天候即時前車偵測與信號辨識系統
Author: Jia-Ying Xu
Date: July, 2019

*/

#include <opencv2/opencv.hpp>

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string> 
#include <ctime>
#include <ratio>
#include <chrono>


#include "Vehicle.h"
#include "dsst.h"
#include "nms.h"
#include "nms_nighttime.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

high_resolution_clock::time_point t1;
high_resolution_clock::time_point t2;
high_resolution_clock::time_point t3;
high_resolution_clock::time_point t4;


Adaboost adaboost;
Vehicle vehicle;
Vehicle_nighttime vehicle_night;
Signal signal;
NMS nms;
NMS_nighttime nms_nighttime;

vector<Rect> detection_object;
vector<Rect> detection_object_pre1;
vector<Rect> detection_object_pre2;

vector<Rect> tracking_object;
vector<int> tracking_object_siganl;


DSST track;
vector<DSST> tracker;


int frame_sample = 40;//取樣signal recognition N frame


//lighting recognition
int lighting_recognition(Mat img, int frame);
int light_frame = 30;//取樣lighting recognition N frame
vector<int>lighting;



//Running time
double lighting_recognition_time = 0;
double vehicle_detection_time = 0;
double vehicle_tracking_time = 0;
double signal_recognition_time = 0;
double rununing_time = 0;


int main()
{
	//==================================================
	//=====================  file  =====================
	//==================================================

	vector< vector<classifier>>cascade_classifier_daytime;
	vector<double> thresh_daytime;
	String cascade_file = "cascade_classifier_daytime.txt";
	cascade_classifier_daytime = adaboost.read_file(cascade_file, thresh_daytime);

	vector< vector<classifier>>cascade_classifier_nighttime;
	vector<double> thresh_nighttime;
	cascade_file = "cascade_classifier_nighttime.txt";
	cascade_classifier_nighttime = adaboost.read_file(cascade_file, thresh_nighttime);


	vector< vector<classifier>>cascade_classifier_taillight;
	vector<double> thresh_taillight;
	cascade_file = "cascade_classifier_taillight_0615.txt";
	cascade_classifier_taillight = adaboost.read_file(cascade_file, thresh_taillight);


	//==================================================
	//================       Video      ================
	//==================================================

	string video_name= "Sample-9";

	VideoCapture video("F:\\影片\\論文影片\\" + video_name + ".MP4");

	Size videoSize = Size((int)video.get(CV_CAP_PROP_FRAME_WIDTH), (int)video.get(CV_CAP_PROP_FRAME_HEIGHT));

	//output file
	fstream fout;
	fout.open("result\\" + video_name + ".txt", fstream::out);

	//output video
	//VideoWriter writer;
	//writer.open("video_result\\"+ video_name +".avi", CV_FOURCC('D', 'I', 'V', 'X'), 60, videoSize);

	//ROI setting
	//daytime
	Point x1 = Point(videoSize.width*0.2, videoSize.height*0.4);
	Point x2 = Point(videoSize.width*0.7, videoSize.height*0.85);
	Rect ROI(x1, x2);

	//nighttime
	Point n1 = Point(videoSize.width*0.2, videoSize.height*0.65);
	Point n2 = Point(videoSize.width*0.7, videoSize.height*0.85);
	Rect n_ROI(n1, n2);

	Mat daytime_roi = imread("daytime_roi.png");
	Mat nighttime_roi = imread("nighttime_roi.png");

	//lighting ROI
	lighting.assign(light_frame, 0);
	Point x3 = Point(videoSize.width * 0, videoSize.height * 0);
	Point x4 = Point(videoSize.width * 1, videoSize.height*0.15);
	Rect lighting_ROI(x3, x4);



	Mat img;
	int frame = 0;
	int frame_i = 0;

	while (true)
	{
		video >> img;
		if (img.empty() || cvWaitKey(1) == 27) {
			break;
		}
		frame++;

		Mat lighting_img = img(lighting_ROI);
		Mat img_clone = img.clone();
		Mat crop_img = img(ROI);
		Mat crop_img1 = img(n_ROI);


		t1 = high_resolution_clock::now();
		t3 = high_resolution_clock::now();

		int daytime_nighttime = lighting_recognition(lighting_img, frame);//lighting_recognition

		t4 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t4 - t3);
		lighting_recognition_time += time_span.count() * 1000;

		if (daytime_nighttime==0)
		{
			t3 = high_resolution_clock::now();
			//==================================================
			//=========== Vehicle Detection(daytime) ===========
			//==================================================
			Mat crop_img_gray;
			cvtColor(crop_img, crop_img_gray, CV_BGR2GRAY);
			Mat crop_img_gray1 = vehicle.CDF(crop_img_gray, 0.1);
			for (int y = 0; y < crop_img_gray1.rows; y++)
			{
				Vec3b *data = daytime_roi.ptr<Vec3b>(y);
				uchar *data_out = crop_img_gray1.ptr<uchar>(y);
				for (int x = 0; x < crop_img_gray1.cols; x++)
				{
					if (data[x][0] == 255)
					{
						data_out[x] = data_out[x];
					}
					else
					{
						data_out[x] = 0;
					}
				}
			}

			vector<bounding_box>result_bounding_box = vehicle.vehicle_detection(crop_img_gray1, 0.05, 0.8, 0.65);
			detection_object.erase(detection_object.begin(), detection_object.end());

			for (int i = 0; i < result_bounding_box.size(); i++)
			{

				Point left_top = Point(result_bounding_box[i].left_top_x, result_bounding_box[i].left_top_y) + x1;
				Point right_bottom = Point(result_bounding_box[i].right_bottom_x, result_bounding_box[i].right_bottom_y) + x1;
				Point center = result_bounding_box[i].center + x1;
				Rect roi(left_top, right_bottom);


				Mat c = img_clone(roi);

				int detection_flag = vehicle.classify(c, left_top, right_bottom, center, cascade_classifier_daytime, thresh_daytime);

				if (detection_flag == 1)
				{
					Rect roi1(left_top, right_bottom);
					detection_object.push_back(roi1);
				}

				//else
				//rectangle(img, left_top, right_bottom, Scalar(0, 0, 255), 2);

				//circle(img, result_bounding_box[i].center + x1, 2, Scalar(0, 255, 0), -1);
			}

			//==================================================

			t4 = high_resolution_clock::now();
			time_span = duration_cast<duration<double>>(t4 - t3);
			vehicle_detection_time += time_span.count() * 1000;


			t3 = high_resolution_clock::now();
			//==================================================
			//=========== Vehicle Tracking(daytime) ============
			//==================================================

			vector<int>del_index;

			//erase
			del_index = nms.del_out_range(x1, x2, tracking_object, 80 * 80, 400 * 400);
			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}

			//每個tacking_object追蹤達30frame驗證是不是車子，不是車子則delete tracking object
			for (int i = 0; i < tracker.size(); i++)
			{
				if (tracker[i].frame == 30)
				{
					Mat c = img(tracking_object[i]);
					Mat gray; cvtColor(img, gray, CV_BGR2GRAY);
					Mat reize_img;
					resize(gray, reize_img, Size(64, 64));
					vector<Mat>MB_LBP_img = vehicle.MB_LBP_graph(reize_img);
					vector<double>histogram;
					for (int i = 0; i < MB_LBP_img.size(); i++)
					{
						Mat img = MB_LBP_img[i];
						vector<int>h = vehicle.LBP_H(img);
						histogram.insert(histogram.end(), h.begin(), h.end());
					}
					double confidence;
					int classify_flag = adaboost.predict(cascade_classifier_daytime, thresh_daytime, histogram, confidence);


					if (classify_flag == 1)
					{
						tracker[i].frame == 1;
					}
					else
					{
						tracking_object.erase(tracking_object.begin() + i);
						tracker.erase(tracker.begin() + i);
					}

				}
			}


			del_index = nms.nms_detection_tracking(detection_object, tracking_object, 0.5);

			//將detection object 加入 tracking
			for (int i = 0; i < detection_object.size(); i++)
			{
				tracking_object.push_back(detection_object[i]);
				track.feature_flag = 0;//0:gray feature ,1:hog feature。
				track.cell_size = 1;//1:gray feature, 4: hog feature。
				track.intensity_L.assign(frame_sample, 0);
				track.intensity_R.assign(frame_sample, 0);
				tracker.push_back(track);
			}

			//將紀錄歷史車燈資訊從(tracking_object to detection_object)
			for (int i = 0; i < del_index.size(); i++)
			{
				int l = del_index[i];
				for (int j = 0; j < tracking_object.size(); j++)
				{
					if (nms.IOU(tracking_object[l], tracking_object[j]) > 0.5)
					{
						tracker[j].intensity_L = tracker[l].intensity_L;
						tracker[j].intensity_R = tracker[l].intensity_R;
						tracker[j].signal_state = tracker[l].signal_state;//signal state
						tracker[j].taillight_frame = tracker[l].taillight_frame;

					}
				}
			}

			//update taillight_frame
			for (int i = 0; i < tracker.size(); i++)
			{
				tracker[i].taillight_frame = tracker[i].taillight_frame + 1;

			}


			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}


			//tracking update
			for (int i = 0; i < tracker.size(); i++)
			{
				if (!tracker[i].init)
				{
					tracker[i].Init(img, tracking_object[i]);
					tracker[i].init = true;

				}
				else
				{
					tracking_object[i] = tracker[i].Update(img, true);//true:代表追蹤框size會改變
				}
			}



			del_index = nms.nms_tracking(tracking_object, 0.1);
			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}
			//==================================================

			t4 = high_resolution_clock::now();
			time_span = duration_cast<duration<double>>(t4 - t3);
			vehicle_tracking_time += time_span.count() * 1000;


		}
		else
		{
			t3 = high_resolution_clock::now();
			//==================================================
			//========== Vehicle Detection(Nighttime) ==========
			//==================================================

			detection_object.erase(detection_object.begin(), detection_object.end());
			detection_object = vehicle_night.vehicle_detection(n1, img_clone, crop_img1, cascade_classifier_nighttime, thresh_nighttime, 0.45);

			//==================================================

			t4 = high_resolution_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t4 - t3);
			vehicle_detection_time += time_span.count() * 1000;

			t3 = high_resolution_clock::now();
			//==================================================
			//========== Vehicle Tracking(Nighttime) ===========
			//==================================================
			vector<int>del_index;

			//連續detection frame(t)與frame(t-1)、frame(t-2)，IOU1大於0.8 才加入追蹤。
			vector<Rect> add_tracking_object;
			vector<bool>add = nms_nighttime.nms_detection_detection_pre(detection_object, detection_object_pre1, 0.5);
			vector<bool>add1 = nms_nighttime.nms_detection_detection_pre(detection_object, detection_object_pre2, 0.5);


			for (int i = 0; i < detection_object.size(); i++)
			{
				if (add[i] && add1[i])
				{
					add_tracking_object.push_back(detection_object[i]);
				}
			}


			//erase
			del_index = nms_nighttime.del_out_range(x1, x2, tracking_object, 80 * 80, 500 * 500);
			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}


			//每個tacking_object追蹤達30frame驗證是不是車子，不是車子則delete tracking object
			for (int i = 0; i < tracker.size(); i++)
			{
				if (tracker[i].frame == 29)
				{
					Mat c = img(tracking_object[i]);
					int  classify_flag = vehicle_night.classify(c, cascade_classifier_nighttime, thresh_nighttime);

					if (classify_flag == 1)
					{
						tracker[i].frame == 1;
					}
					else
					{
						tracking_object.erase(tracking_object.begin() + i);
						tracker.erase(tracker.begin() + i);
					}

				}
			}



			del_index = nms_nighttime.nms_detection_tracking(add_tracking_object, tracking_object, 0.5);

			//將detection object 加入 tracking
			for (int i = 0; i < add_tracking_object.size(); i++)
			{

				tracking_object.push_back(add_tracking_object[i]);
				track.feature_flag = 0;//0:gray feature ,1:hog feature。
				track.cell_size = 1;//1:gray feature, 4: hog feature。
				track.intensity_L.assign(frame_sample, 0);
				track.intensity_R.assign(frame_sample, 0);
				tracker.push_back(track);
			}


			//將紀錄歷史車燈資訊從(tracking_object to detection_object)
			for (int i = 0; i < del_index.size(); i++)
			{
				int l = del_index[i];
				for (int j = 0; j < tracking_object.size(); j++)
				{
					if (nms_nighttime.IOU(tracking_object[l], tracking_object[j]) > 0.5)
					{
						tracker[j].intensity_L = tracker[l].intensity_L;
						tracker[j].intensity_R = tracker[l].intensity_R;
						tracker[j].signal_state = tracker[l].signal_state;//signal state
						tracker[j].taillight_frame = tracker[l].taillight_frame;
					}
				}
			}


			//update taillight_frame
			for (int i = 0; i < tracker.size(); i++)
			{
				tracker[i].taillight_frame = tracker[i].taillight_frame + 1;

			}



			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}

			//tracking update
			for (int i = 0; i < tracker.size(); i++)
			{
				if (!tracker[i].init)
				{
					tracker[i].Init(img, tracking_object[i]);
					tracker[i].init = true;

				}
				else
				{
					tracking_object[i] = tracker[i].Update(img, true);//true:代表追蹤框size會改變

				}
			}



			del_index = nms_nighttime.nms_tracking(tracking_object, 0.1);
			for (int i = 0; i < del_index.size(); i++)
			{
				tracking_object.erase(tracking_object.begin() + del_index[i]);
				tracker.erase(tracker.begin() + del_index[i]);
				for (int j = 0; j < del_index.size(); j++)
				{
					del_index[j] = del_index[j] - 1;
				}
			}
			//==================================================

			//frame(t-2) detection_object
			detection_object_pre2.erase(detection_object_pre2.begin(), detection_object_pre2.end());
			detection_object_pre2 = detection_object_pre1;

			//frame(t-1) detection_object
			detection_object_pre1.erase(detection_object_pre1.begin(), detection_object_pre1.end());
			detection_object_pre1 = detection_object;


			t4 = high_resolution_clock::now();
			time_span = duration_cast<duration<double>>(t4 - t3);
			vehicle_tracking_time += time_span.count() * 1000;

		}
		
		for (int i = 0; i < tracking_object.size(); i++)
		{
			rectangle(img, tracking_object[i], Scalar(0, 255, 0), 2);
		}


		t3 = high_resolution_clock::now();
		//==================================================
		//================Taillight Detection ==============
		//==================================================

		tracking_object_siganl.erase(tracking_object_siganl.begin(), tracking_object_siganl.end());
		for (int i = 0; i < tracking_object.size(); i++)
		{
			Mat d = img_clone(tracking_object[i]);
			vector<bounding_box >pair_bounding_box;
			vector<bounding_box >light_bounding_box;
			int taillight_flag = 0;

			if (daytime_nighttime == 0)
			{
				taillight_flag = vehicle.taillight_detection(d, pair_bounding_box, light_bounding_box);//daytime
			}
			else
			{
				taillight_flag = vehicle_night.taillight_detection(d, pair_bounding_box, light_bounding_box);//nighttime
			}

			Point left_top(tracking_object[i].x, tracking_object[i].y);
			Point center_top(tracking_object[i].x + tracking_object[i].width / 2, tracking_object[i].y);
			Point right_top(tracking_object[i].x + tracking_object[i].width, tracking_object[i].y);

			//draw
			//for (int k = 0; k < light_bounding_box.size(); k++)
			//{
			//	Point l = Point(light_bounding_box[k].left_top_x, light_bounding_box[k].left_top_y) + left_top;
			//	Point r = Point(light_bounding_box[k].right_bottom_x, light_bounding_box[k].right_bottom_y) + left_top;
			//	rectangle(img, l, r, Scalar(255, 0, 0), 2);
			//	circle(img, light_bounding_box[k].center + left_top, 2, Scalar(0, 0, 255), -1);
			//}


			double pre_hsv_v_L = 0;
			double pre_hsv_v_R = 0;

			int label;
			int frame_tracker = tracker[i].taillight_frame;
			if (frame_tracker % frame_sample == 0)
				label = frame_sample - 1;
			else
				label = (frame_tracker% frame_sample) - 1;

			if (taillight_flag == 1)
			{

				Point l = Point(pair_bounding_box[0].left_top_x, pair_bounding_box[0].left_top_y) + left_top;
				Point r = Point(pair_bounding_box[0].right_bottom_x, pair_bounding_box[0].right_bottom_y) + left_top;
				//rectangle(img, l, r, Scalar(0, 0, 255), 2);

				l = Point(pair_bounding_box[1].left_top_x, pair_bounding_box[1].left_top_y) + left_top;
				r = Point(pair_bounding_box[1].right_bottom_x, pair_bounding_box[1].right_bottom_y) + left_top;
				//rectangle(img, l, r, Scalar(0, 0, 255), 2);

				tracker[i].intensity_L[label] = signal.compute_intensity(d, pair_bounding_box[0]);
				tracker[i].intensity_R[label] = signal.compute_intensity(d, pair_bounding_box[1]);

				pre_hsv_v_L = tracker[i].intensity_L[label];
				pre_hsv_v_R = tracker[i].intensity_R[label];

			}
			else
			{
				tracker[i].intensity_L[label] = pre_hsv_v_L;
				tracker[i].intensity_R[label] = pre_hsv_v_R;

			}

			//==================================================
			//============== Taillight Recognition =============
			//==================================================
			int signal_state_pre = tracker[i].signal_state; //frame(t-1)尾燈信號狀態

			signal.signal_recognition(daytime_nighttime,frame_sample,label, signal_state_pre,tracker[i].signal_state, tracker[i].intensity_L, tracker[i].intensity_R, pair_bounding_box, light_bounding_box, cascade_classifier_taillight, thresh_taillight);
			
			int signal_classify = 0;
			
			if (signal_state_pre == tracker[i].signal_state)
			{
				signal_classify = tracker[i].signal_state;
			}
			
			if (signal_classify > 4)
			{

				signal_classify = signal_classify - 4;
				putText(img, "B", center_top, 0, 2, Scalar(0, 0, 255), 3);
				if (signal_classify == 1)
					putText(img, "L", left_top, 0, 2, Scalar(0, 255, 255), 3);
				else if (signal_classify == 2)
					putText(img, "R", right_top, 0, 2, Scalar(0, 255, 255), 3);
				else if (signal_classify == 3)
				{
					putText(img, "W", left_top, 0, 2, Scalar(0, 255, 255), 3);
					putText(img, "W", right_top, 0, 2, Scalar(0, 255, 255), 3);
				}
			}
			else
			{
				if (signal_classify == 1)
					putText(img, "L", left_top, 0, 2, Scalar(0, 255, 255), 3);
				else if (signal_classify == 2)
					putText(img, "R", right_top, 0, 2, Scalar(0, 255, 255), 3);
				else if (signal_classify == 3)
				{
					putText(img, "W", left_top, 0, 2, Scalar(0, 255, 255), 3);
					putText(img, "W", right_top, 0, 2, Scalar(0, 255, 255), 3);
				}
				else if (signal_classify == 4)
					putText(img, "B", center_top, 0, 2, Scalar(0, 0, 255), 3);
			}

			tracking_object_siganl.push_back(signal_classify);
			//==================================================
		}



		t4 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t4 - t3);
		signal_recognition_time += time_span.count() * 1000;
		//==================================================

		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		rununing_time += time_span.count() * 1000;

		//==================================================
		//=====================  Output  ===================
		//==================================================

		rectangle(img, x1, x2, Scalar(255, 0, 0), 2);

		//output video
		//writer.write(img);


		//write img
		//if ((frame-1) % 1 == 0)
		//{
		//	sprintf(path, "f:\\影片\\實驗結果影像\\result.mp4\\");
		//	sprintf(path1, "%d.jpg", frame_i);
		//	strcat(path, path1);
		//	imwrite(path, img);
		//	frame_i++;
		//}


		//輸出bounding box file result
		fout << frame << " ";
		fout << tracking_object.size() << " ";
		for (int i = 0; i < tracking_object.size(); i++)
		{
			fout << tracking_object[i].x << " ";
			fout << tracking_object[i].y << " ";
			fout << tracking_object[i].width << " ";
			fout << tracking_object[i].height << " ";
			fout << tracking_object_siganl[i] << " ";//signal recognition
		}
		fout << endl;

		char buffer[100];
		if (daytime_nighttime == 0)
		{
			sprintf(buffer, "daytime");
			putText(img, buffer, Point(5, 30), 0, 1, Scalar(0, 255, 0), 2);

		}
		else
		{
			sprintf(buffer, "nighttime");
			putText(img, buffer, Point(5, 30), 0, 1, Scalar(0, 255, 0), 2);
		}

		imshow("result", img);
		//==================================================
	}
	cout << "Video name: " << video_name << " frame: " << frame << endl;
	cout << "Average lighting_recognition time: " << lighting_recognition_time / frame << " mseconds." << endl;
	cout << "Average vehicle_detection time: " << vehicle_detection_time / frame << " mseconds." << endl;
	cout << "Average vehicle_tracking time: " << vehicle_tracking_time / frame << " mseconds." << endl;
	cout << "Average signal recognition time: " << signal_recognition_time / frame << " mseconds." << endl;
	cout << "Average running time: " << rununing_time / frame << " mseconds." << endl;

	waitKey();
	return 0;
}




//lighting recognition
//0: daytime   1: nighttime
int lighting_recognition(Mat img, int frame)
{
	Mat img_hsv;
	cvtColor(img, img_hsv, CV_BGR2HSV);


	vector<int>v; v.assign(256, 0);
	for (int y = 0; y < img_hsv.rows; y++)
	{
		Vec3b *data = img_hsv.ptr<Vec3b>(y);
		for (int x = 0; x < img_hsv.cols; x++)
		{

			v[int(data[x][2])] = v[int(data[x][2])] + 1;
		}
	}


	int  max_v_label = 0;
	int max = 0;
	for (int i = 0; i < v.size(); i++)
	{
		if (v[i] > max)
		{
			max = v[i];
			max_v_label = i;
		}
	}



	int tmp = 0;
	if ((frame%light_frame) > 0)
		tmp = (frame%light_frame) - 1;
	else
		tmp = light_frame - 1;

	lighting[tmp] = max_v_label;

	int size = 0;
	if (frame >= light_frame)
		size = light_frame;
	else
		size = frame;

	double total_lighting = 0;
	for (int i = 0; i < size; i++)
	{
		total_lighting = total_lighting + lighting[i];
	}
	int average_v_label = 0;
	average_v_label = total_lighting / size;


	int Max_V = 100;//門檻值

	if (average_v_label >= Max_V)
		return 0;
	else
		return 1;

}

