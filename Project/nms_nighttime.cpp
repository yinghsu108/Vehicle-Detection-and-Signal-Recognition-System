#include "nms_nighttime.h"


double NMS_nighttime::IOU(Rect r1, Rect r2)
{
	int x1 = max(r1.x, r2.x);
	int y1 = max(r1.y, r2.y);
	int x2 = min(r1.x + r1.width, r2.x + r2.width);
	int y2 = min(r1.y + r1.height, r2.y + r2.height);
	int w = max(0, (x2 - x1 + 1));
	int h = max(0, (y2 - y1 + 1));
	double inter = w * h;
	double o = inter / (r1.area() + r2.area() - inter);

	if (r1.area() <= inter || r2.area() <= inter)
		return 1;
	else
		return (o >= 0) ? o : 0;
}


double NMS_nighttime::IOU1(Rect r1, Rect r2)
{
	int x1 = max(r1.x, r2.x);
	int y1 = max(r1.y, r2.y);
	int x2 = min(r1.x + r1.width, r2.x + r2.width);
	int y2 = min(r1.y + r1.height, r2.y + r2.height);
	int w = max(0, (x2 - x1 + 1));
	int h = max(0, (y2 - y1 + 1));
	double inter = w * h;
	double o = inter / (r1.area() + r2.area() - inter);


	return (o >= 0) ? o : 0;
}







void resolve(int a, int b, vector<int>& rtabel)//校正關係(合併標籤)
{

	int temp;
	if (rtabel[a] < rtabel[b])
	{
		temp = rtabel[b];
		rtabel[b] = rtabel[a];
		for (int i = 0; i < rtabel.size(); i++)
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
		for (int i = 0; i < rtabel.size(); i++)
		{
			if (rtabel[i] == temp)
			{
				rtabel[i] = rtabel[a];
			}
		}
	}

}


vector<bool> NMS_nighttime::nms_detection_detection_pre(vector<Rect> detection_object, vector<Rect> detection_object_pre, double nms_threshold)
{
	vector<bool>add(detection_object.size(), false);

	for (int i = 0; i < detection_object_pre.size(); i++) {
		for (int j = 0; j < detection_object.size(); j++)
		{
			if (!add[j])
			{
				if (IOU1(detection_object_pre[i], detection_object[j]) >= nms_threshold)
				{
					add[j] = true;
				}
			}
		}
	}
	return add;
}


vector<int> NMS_nighttime::del_middle_tracking_vehicle(vector<Rect> proposals)
{
	vector<int> index;
	for (int i = 0; i < proposals.size(); ++i) {
		index.push_back(i);
	}

	for (int i = 0; i < proposals.size(); i++) {
		for (int j = i + 1; j < proposals.size(); j++) {
			if (IOU(proposals[i], proposals[j]) >0)//overlap
			{
				resolve(i, j, index);
			}
		}
	}

	vector<int> count_index(index.size(), 0);
	vector<bool> del(proposals.size(), false);

	for (int i = 0; i < index.size(); i++)
	{
		//cout << i << ": " << index[i] << endl;
		count_index[index[i]]++;
	}

	for (int i = 0; i < count_index.size(); i++)
	{
		//cout << i << ": " << count_index[i] << endl;
		if (count_index[i] == 3)
		{
			int min_x_index;
			int max_x_index;
			int min_x_value = 9999;
			int max_x_value = 0;

			for (int j = 0; j < index.size(); j++)
			{
				if (i == index[j])
				{
					if (proposals[j].x < min_x_value)
					{
						min_x_value = proposals[j].x;
						min_x_index = j;
					}

					if ((proposals[j].x + proposals[j].width)> max_x_value)
					{
						max_x_value = proposals[j].x + proposals[j].width;
						max_x_index = j;
					}
				}
			}


			for (int k = 0; k < index.size(); k++)
			{
				if (i == index[k])
				{
					if (k != min_x_index && k != max_x_index)
					{
						del[k] = true;

					}
				}
			}


		}
	}



	vector<int> del_index;
	for (int i = 0; i < proposals.size(); i++)
	{
		if (del[i])
			del_index.push_back(i);
	}

	return del_index;

}





vector<int> NMS_nighttime::nms_detection_tracking(vector<Rect> detection_object, vector<Rect> tracking_object, double nms_threshold)
{
	vector<bool> del(tracking_object.size(), false);

	for (int i = 0; i < detection_object.size(); i++) {
		for (int j = 0; j < tracking_object.size(); j++)
		{
			if (!del[j])
			{
				if (IOU(detection_object[i], tracking_object[j]) >= nms_threshold)
				{
					del[j] = true;
				}
			}
		}
	}

	vector<int> del_index;
	for (int i = 0; i < tracking_object.size(); i++)
	{
		if (del[i])
			del_index.push_back(i);
	}

	return del_index;
}


vector<int>NMS_nighttime::nms_tracking(vector<Rect> proposals, double nms_threshold)
{
	vector<int> scores;
	for (auto i : proposals) scores.push_back(i.area());

	vector<int> index;
	for (int i = 0; i < scores.size(); ++i) {
		index.push_back(i);
	}

	sort(index.begin(), index.end(), [&](int a, int b) {
		return scores[a] > scores[b];
	});



	vector<bool> del(scores.size(), false);
	for (int i = 0; i < index.size(); i++) {
		if (!del[index[i]]) {
			for (int j = i + 1; j < index.size(); j++) {
				if (IOU(proposals[index[i]], proposals[index[j]]) > nms_threshold) {
					del[index[j]] = true;
				}
			}
		}
	}
	vector<int> del_index;
	for (int i = 0; i < index.size(); i++) {
		if (del[i])
			del_index.push_back(i);
	}

	return del_index;
}




vector<int> NMS_nighttime::del_out_range(Point left_top, Point right_bottom, vector<Rect> tracking_object, int min_area, int max_area)
{
	vector<bool> del(tracking_object.size(), false);
	for (int i = 0; i < tracking_object.size(); i++)
	{
		Point center = Point(tracking_object[i].x + tracking_object[i].width / 2, tracking_object[i].y + tracking_object[i].height / 2);
		if ((center.x<left_top.x) || (center.x>right_bottom.x) || (center.y>right_bottom.y) || (tracking_object[i].area()<min_area) || (tracking_object[i].area()>max_area))
		{
			del[i] = true;
		}
	}


	vector<int> del_index;
	for (int i = 0; i < tracking_object.size(); i++)
	{
		if (del[i])
			del_index.push_back(i);
	}

	return del_index;

}