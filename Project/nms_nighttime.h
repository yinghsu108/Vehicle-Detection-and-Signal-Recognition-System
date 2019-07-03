#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include <vector>



using namespace cv;
using namespace std;


class NMS_nighttime
{
public:

	double NMS_nighttime::IOU(Rect r1, Rect r2);
	double NMS_nighttime::IOU1(Rect r1, Rect r2);

	vector<bool> NMS_nighttime::nms_detection_detection_pre(vector<Rect> detection_object, vector<Rect> tracking_object, double nms_threshold);

	vector<int> NMS_nighttime::del_middle_tracking_vehicle(vector<Rect> proposals);
	vector<int> NMS_nighttime::nms_detection_tracking(vector<Rect> detection_object, vector<Rect> tracking_object, double nms_threshold);
	vector<int>NMS_nighttime::nms_tracking(vector<Rect> proposals, double nms_threshold);
	vector<int> NMS_nighttime::del_out_range(Point left_top, Point right_bottom, vector<Rect> tracking_object, int min_area, int max_area);

};