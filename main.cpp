#include<iostream>
#include<cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

double line_ingradient(int x1, int y1, int x2, int y2)
{
    return (double)(y1-y2)/(x1-x2);
}

int main()
{
    VideoCapture cap("anticlockwise.mp4");
    Mat frame;
    Mat gray;
    Mat gaussian;
    Mat canny_t;
    Mat hsv;
    Mat yellow_mask;
    Mat white_mask;

    Scalar row_yellow(30-10,80,80);
    Scalar high_yellow(30+10,255,255);

    Scalar row_white(170,170,170);
    Scalar high_white(255,255,255);

    double canny_t1 = 50;
    double canny_t2 = 150;
    auto width = cap.get(CAP_PROP_FRAME_WIDTH);
    auto height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double gaussian_sgm = 2.0;
    int line_tsh = 20;
    double line_min_length = 10;
    double line_max_gap = 20;

    double angle;
    int alpha = 1000;

    Mat white(Size(width,height/2),CV_32FC3,Scalar(255,255,255));

    namedWindow("original");
    resizeWindow("original",width,height);
    namedWindow("hsv");
    resizeWindow("hsv",width, height/2);
    namedWindow("white");
    resizeWindow("white",width, height/2);
    namedWindow("gaussian");
    resizeWindow("gaussian",width,height/2);

    namedWindow("gray");
    resizeWindow("gray",width,height/2);
    namedWindow("canny_t");
    resizeWindow("canny",width,height/2);

    moveWindow("hsv",width*1.2,0);
    moveWindow("gray",width*1.2,height*1.4);
    moveWindow("canny_t",width*1.2*3,height*1.4*1.5);
    moveWindow("white",0, height*1.2);
    moveWindow("gaussian",width*1.2*2,height*1.4);
    
    while(1)
    {
        cap >> frame;
        if(frame.empty())
            break;
        cvtColor(frame.rowRange(height/2,height),hsv,CV_BGR2HSV);
        inRange(hsv,row_yellow,high_yellow,yellow_mask);

        cvtColor(hsv,hsv,CV_HSV2RGB);
        inRange(hsv, row_white, high_white, white_mask);
        hsv.setTo(Scalar(255,255,255),yellow_mask);
        hsv.setTo(Scalar(0,0,255),white_mask);
        

        cvtColor(frame.rowRange(height/2, height),gray,CV_BGR2GRAY);
        GaussianBlur(gray,gaussian,Size(3,3),gaussian_sgm);
        Canny(gaussian,canny_t,canny_t1,canny_t2);

        vector<Vec4i> lines;
        HoughLinesP(canny_t,lines,1,CV_PI/180,line_tsh,line_min_length,line_max_gap);
        for(Vec4i l : lines)
        {
            angle = line_ingradient(l[0],l[1],l[2],l[3]);
            if(fabs(angle)>0.26 && fabs(angle) < 3.7){
                int dx = abs(l[0] - l[2]);
                int dy = abs(l[1] - l[3]);
                if (angle > 0){
                    l[0] = l[0] + alpha * dx;
                    l[1] = l[1] + alpha * dy;
                    l[2] = l[2] - alpha * dx;
                    l[3] = l[3] - alpha * dy;
                }
                else
                {
                    l[0] = l[0] + alpha * dx;
                    l[1] = l[1] - alpha * dy;
                    l[2] = l[2] - alpha * dx;
                    l[3] = l[3] + alpha * dy;
                }
                line(white, Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),1,LINE_AA);
            }
        }
        

        imshow("original",frame);
        imshow("gray",gray);
        imshow("canny_t",canny_t);
        imshow("white",white);
        imshow("gaussian",gaussian);
        imshow("hsv",hsv);
        white.setTo(Scalar(255,255,255));
        if(waitKey(10) == 27)
            break;
    }
   return 0;


}