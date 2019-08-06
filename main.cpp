#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#define __CV4
//#define __HSV

using namespace std;
using namespace cv;

#if !defined(__CV4)
#define BGR2HSV CV_BGR2HSV
#define HSV2RGB CV_HSV2RGB
#define BGR2GRAY CV_BGR2GRAY
#else
#define BGR2HSV COLOR_BGR2HSV
#define HSV2RGB COLOR_HSV2RGB
#define BGR2GRAY COLOR_BGR2GRAY
#endif

double line_ingradient(int x1, int y1, int x2, int y2)
{
  return (double)(y1 - y2) / (x1 - x2);
}

void init_windows(double width, double height, double roi_height)
{

  namedWindow("original");

#ifdef __HSV
  namedWindow("hsv");
  moveWindow("hsv", width * 1.2, 0);
#endif

  namedWindow("white");
  moveWindow("white", 0, height * 1.4);

  namedWindow("gray");
  moveWindow("gray", width * 1.2, height * 1.4);

  namedWindow("canny_t");
  moveWindow("canny_t", width * 1.2 * 2, height * 1.4);
}

int main()
{
  VideoCapture cap("clockwise.mp4");
  Mat frame, gray, gaussian, canny_t, hsv;

  Scalar s_white(255, 255, 255);
  Scalar s_black(0, 0, 0);
  Scalar s_red(0, 0, 255);
  Scalar s_green(0, 255, 0);
  Scalar s_blue(255, 0, 0);

  double canny_t1 = 50;
  double canny_t2 = 150;

  double width = cap.get(CAP_PROP_FRAME_WIDTH);
  double height = cap.get(CAP_PROP_FRAME_HEIGHT);
  int roi_height = cvRound(height / 2.5);

  double gaussian_sgm = 2.0;
  int line_tsh = 20;
  double line_min_length = 20;
  double line_max_gap = 20;
  int mopology_itr = 1;

  double angle;
  int alpha = 100;

  int mid_x_bf = width / 2;

  vector<Vec4i> lines;

  Mat white(Size(width, roi_height), CV_8UC3, s_white);

  init_windows(width, height, roi_height);

  Point before_left_line[2];
  Point before_right_line[2];

#ifdef __HSV
  //For HSV proccessing
  Mat yellow_mask, white_mask;
  Scalar row_yellow(30 - 10, 100, 100);
  Scalar high_yellow(30 + 10, 255, 255);
  Scalar row_white(170, 170, 170);
  Scalar high_white(255, 255, 255);
#endif

  while (1)
  {
    cap >> frame;
    if (frame.empty())
    {
      waitKey(0);
      break;
    }

#ifdef __HSV
    // TODO: Split HSV processing and Lane detecting logics.
    // HSV image processing logic (Can remove)
    cvtColor(frame.rowRange(height - roi_height, height), hsv, BGR2HSV);
    inRange(hsv, row_yellow, high_yellow, yellow_mask);

    cvtColor(hsv, hsv, HSV2RGB);
    inRange(hsv, row_white, high_white, white_mask);

    dilate(yellow_mask, yellow_mask, Mat(), Point(-1, -1), mopology_itr);
    dilate(white_mask, white_mask, Mat(), Point(-1, -1), mopology_itr);
    hsv.setTo(Scalar(0, 0, 255), white_mask);
    hsv.setTo(Scalar(255, 255, 255), yellow_mask);
#endif

    //Before Line Detecting Logic
    cvtColor(frame.rowRange(height - roi_height, height).clone(), gray, BGR2GRAY);
    GaussianBlur(gray, gaussian, Size(3, 3), gaussian_sgm);
    Canny(gaussian, canny_t, canny_t1, canny_t2);
    HoughLinesP(canny_t, lines, 1, CV_PI / 180, line_tsh, line_min_length, line_max_gap);

    // Line Detecting Logic
    int left_min = width;
    int right_min = width;

    int right_min_x = cvRound(width / 4);
    int left_max_x = cvRound((width / 4) * 3);

    Point left_line[2];
    bool check_left = false;
    Point right_line[2];
    bool check_right = false;
    Point parallel_line[2];

    for (Vec4i l : lines)
    {
      angle = line_ingradient(l[0], l[1], l[2], l[3]);
      if (fabs(angle) > 0.5 && fabs(angle) < 5.6)
      {
        line(white, Point(l[0], l[1]), Point(l[2], l[3]), s_red, 1, LINE_AA);

        if (angle < 0)
        {
          if (left_max_x > l[0] && left_max_x > l[2])
          {
            int line_ = ((roi_height / 3) - l[1]) * ((l[0] - l[2]) / (l[1] - l[3])) + l[0];
            if (left_min > abs(line_ - width / 2))
            {
              left_min = abs(line_ - width / 2);
              left_line[0] = Point(l[0], l[1]);
              left_line[1] = Point(l[2], l[3]);
              check_left = true;
            }
          }
        }
        else if (angle > 0)
        {
          if (right_min_x < l[0] && right_min_x < l[2])
          {
            int line_ = ((roi_height / 3) - l[1]) * ((l[0] - l[2]) / (l[1] - l[3])) + l[0];
            if (right_min > abs(line_ - width / 2))
            {
              right_min = abs(line_ - width / 2);
              right_line[0] = Point(l[0], l[1]);
              right_line[1] = Point(l[2], l[3]);
              check_right = true;
            }
          }
        }
      }
    }

    //If can't find line
    if (check_left)
    {
      before_left_line[0] = left_line[0];
      before_left_line[1] = left_line[1];
    }
    else
    {
      left_line[0] = before_left_line[0];
      left_line[1] = before_left_line[1];
    }

    if (check_right)
    {
      before_right_line[0] = right_line[0];
      before_right_line[1] = right_line[1];
    }
    else
    {
      right_line[0] = before_right_line[0];
      right_line[1] = before_right_line[1];
    }

    //Making long line logics
    int left_dx = left_line[0].x - left_line[1].x;
    int left_dy = left_line[0].y - left_line[1].y;
    int right_dx = right_line[0].x - right_line[1].x;
    int right_dy = right_line[0].y - right_line[1].y;

    left_line[0].x += abs(left_dx) * alpha;
    left_line[0].y -= abs(left_dy) * alpha;
    left_line[1].x -= abs(left_dx) * alpha;
    left_line[1].y += abs(left_dy) * alpha;

    right_line[0].x += abs(right_dx) * alpha;
    right_line[0].y += abs(right_dy) * alpha;
    right_line[1].x -= abs(right_dx) * alpha;
    right_line[1].y -= abs(right_dy) * alpha;

    //Formular to find Vanishing point
    double left_gredient = (double)left_dy / left_dx;
    double right_gradient = (double)right_dy / right_dx;
    double left_y_itc = left_line[0].y - left_gredient * left_line[0].x;
    double right_y_itc = right_line[0].y - right_gradient * right_line[0].x;

    int mid_x = -(left_y_itc - right_y_itc) / (left_gredient - right_gradient);
    int mid_y = left_gredient * mid_x + left_y_itc;

    if (mid_y > 0 && mid_y < roi_height)
      mid_x = mid_x_bf;
    else
      mid_x_bf = mid_x;

    //Draw line
    line(white, left_line[0], left_line[1], s_black, 2);
    line(white, right_line[0], right_line[1], s_black, 2);
    line(white, Point(mid_x, 0), Point(mid_x, roi_height), s_blue, 2);

    //Result
    string str = string("x: ") + to_string((width / 2 - mid_x) * (-1));
    putText(white, str, Point(10, 20), 1, 1, s_black);

    // Frame showing logics
    imshow("original", frame);
    imshow("gray", gray);
    imshow("canny_t", canny_t);
    imshow("white", white);

#ifdef __HSV
    imshow("hsv", hsv);
#endif
    // Set whiteboard empty
    white.setTo(s_white);

    //Space := pause ESC := end
    int ke = waitKey(10);
    if (ke == ' ')
      waitKey();
    else if (ke == 27)
      break;
  }
  return 0;
}
