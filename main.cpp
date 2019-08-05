#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#define __CV4

using namespace std;
using namespace cv;

#if !defined ( __CV4 )
#  define BGR2HSV CV_BGR2HSV
#  define HSV2RGB CV_HSV2RGB
#  define BGR2GRAY CV_BGR2GRAY
#else
#  define BGR2HSV COLOR_BGR2HSV
#  define HSV2RGB COLOR_HSV2RGB
#  define BGR2GRAY COLOR_BGR2GRAY
#endif

double line_ingradient(int x1, int y1, int x2, int y2)
{
  return (double)(y1-y2)/(x1-x2);
}

void init_windows(double width, double height, double roi_height) {
  namedWindow("original");
  namedWindow("hsv");
  namedWindow("white");

  namedWindow("gray");
  namedWindow("canny_t");

  moveWindow("hsv",width*1.2,0);
  moveWindow("gray",width*1.2,height*1.4);
  moveWindow("canny_t",width*1.2*3,height*1.4*1.5);
  moveWindow("white",0, height*1.2);
}

int main()
{
  VideoCapture cap("anticlockwise.mp4");
  Mat frame, gray, gaussian, canny_t, hsv;
  Mat yellow_mask, white_mask;

  Scalar row_yellow(30-10,100,100);
  Scalar high_yellow(30+10,255,255);

  Scalar row_white(170,170,170);
  Scalar high_white(255,255,255);

  Vec3b c_yellow(0,0,255);
  Vec3b c_white(255,255,255);

  double canny_t1 = 50;
  double canny_t2 = 150;

  double width = cap.get(CAP_PROP_FRAME_WIDTH);
  double height = cap.get(CAP_PROP_FRAME_HEIGHT);
  int roi_height = cvRound(height/2.5);

  double gaussian_sgm = 2.0;
  int line_tsh = 20;
  double line_min_length = 20;
  double line_max_gap = 20;
  int mopology_itr = 1;

  double angle;
  int alpha = 1000;
  vector<Vec4i> lines;

  Mat white(Size(width,roi_height),CV_8UC3,Scalar(255,255,255));

  init_windows(width, height, roi_height);
  
  while(1)
    {
      cap >> frame;
      if(frame.empty()) {
        waitKey(0);
        break;
      }

      // TODO: Split HSV processing and Lane detecting logics.
      // HSV image processing logic
      cvtColor(frame.rowRange(height - roi_height,height),hsv,BGR2HSV);
      inRange(hsv,row_yellow,high_yellow,yellow_mask);

      cvtColor(hsv,hsv,HSV2RGB);
      inRange(hsv, row_white, high_white, white_mask); 

      dilate(yellow_mask,yellow_mask,Mat(),Point(-1,-1),mopology_itr);
      dilate(white_mask,white_mask,Mat(),Point(-1,-1),mopology_itr);
      hsv.setTo(Scalar(0,0,255),white_mask);
      hsv.setTo(Scalar(255,255,255),yellow_mask);

      //Before Line Detecting Logic
      cvtColor(frame.rowRange(height - roi_height, height),gray,BGR2GRAY);
      GaussianBlur(gray,gaussian,Size(3,3),gaussian_sgm);
      Canny(gaussian,canny_t,canny_t1,canny_t2);
      HoughLinesP(canny_t,lines,1,CV_PI/180,line_tsh,line_min_length,line_max_gap);

      // Line Detecting Logic
      int left_min = width;
      int right_min = width;
      Point left_line[2];
      Point right_line[2];
      for(Vec4i l : lines)
        {
          angle = line_ingradient(l[0],l[1],l[2],l[3]);
          if(fabs(angle)>0.5 && fabs(angle) < 5.6){
            line(white, Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),1,LINE_AA);

            if(angle<0)
              {
                int line_ = ((roi_height/3)-l[1])*((l[0]-l[2])/(l[1]-l[3]))+l[0];
                if(left_min > abs(line_-width/2)){
                  left_min = abs(line_-width/2);
                  left_line[0] = Point(l[0],l[1]);
                  left_line[1] = Point(l[2],l[3]);
                }
              }
            else if(angle>0)
              {
                int line_ = ((roi_height/3)-l[1])*((l[0]-l[2])/(l[1]-l[3]))+l[0];
                if(right_min > abs(line_-width/2)){
                  right_min = abs(line_-width/2);
                  right_line[0] = Point(l[0],l[1]);
                  right_line[1] = Point(l[2],l[3]);
                }
              }
          }
        }

      //Making long line
      int left_dx = abs(left_line[0].x - left_line[1].x);
      int left_dy = abs(left_line[0].y - left_line[1].y);
      int right_dx = abs(right_line[0].x - right_line[1].x);
      int right_dy = abs(right_line[0].y - right_line[1].y);

      left_line[0].x += left_dx * alpha;
      left_line[0].y -= left_dy * alpha;
      left_line[1].x -= left_dx * alpha;
      left_line[1].y += left_dy * alpha;

      right_line[0].x += right_dx *alpha;
      right_line[0].y += right_dy * alpha;
      right_line[1].x -= right_dx * alpha;
      right_line[1].y -= right_dy * alpha;

      line(white,left_line[0],left_line[1],Scalar(0,0,0),2);
      line(white,right_line[0],right_line[1],Scalar(0,0,0),2);

      // Frame showing logics
      imshow("original",frame);
      imshow("gray",gray);
      imshow("canny_t",canny_t);
      imshow("white",white);
      imshow("hsv",hsv);

      // Set whiteboard empty
      white.setTo(Scalar(255,255,255));

      if(waitKey(10) == 27)
        break;
    }
  return 0;
}
