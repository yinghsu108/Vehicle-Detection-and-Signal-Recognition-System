#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include <vector>



using namespace cv;
using namespace std;


class NMS
{
public:

	double NMS::IOU(Rect r1, Rect r2);
	void NMS::nms_detection(vector<Rect>& proposals, double nms_threshold);
	vector<int> NMS::nms_tracking(vector<Rect> proposals, double nms_threshold);

	vector<int> NMS::nms_detection_tracking(vector<Rect> detection_object, vector<Rect> tracking_object, double nms_threshold);
	vector<int> NMS::del_out_range(Point left_top, Point right_bottom,vector<Rect>tracking_object, int min_area, int max_area);

};
