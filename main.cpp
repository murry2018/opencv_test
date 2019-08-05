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

    Scalar row_yellow(30-10,100,100);
    Scalar high_yellow(30+10,255,255);

    Scalar row_white(170,170,170);
    Scalar high_white(255,255,255);

    Vec3b c_yellow(0,0,255);
    Vec3b c_white(255,255,255);

    double canny_t1 = 50;
    double canny_t2 = 150;
    auto width = cap.get(CAP_PROP_FRAME_WIDTH);
    auto height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int roi_height = cvRound(height/2.5);
    int roi_height_y = height - roi_height;
    double gaussian_sgm = 2.0;
    int line_tsh = 20;
    double line_min_length = 20;
    double line_max_gap = 20;
    int mopology_itr = 1;
    int center = 0;
    double angle;
    int alpha = 1000;

    Mat white(Size(width,roi_height),CV_8UC3,Scalar(255,255,255));

    namedWindow("original");
    resizeWindow("original",width,height);
    namedWindow("hsv");
    resizeWindow("hsv",width, roi_height);
    namedWindow("white");
    resizeWindow("white",width, roi_height);
    namedWindow("gaussian");
    resizeWindow("gaussian",width,roi_height);

    namedWindow("gray");
    resizeWindow("gray",width,roi_height);
    namedWindow("canny_t");
    resizeWindow("canny",width,roi_height);

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

        cvtColor(frame.rowRange(roi_height_y,height),hsv,CV_BGR2HSV);
        inRange(hsv,row_yellow,high_yellow,yellow_mask);

        cvtColor(hsv,hsv,CV_HSV2RGB);
        inRange(hsv, row_white, high_white, white_mask); 

        dilate(yellow_mask,yellow_mask,Mat(),Point(-1,-1),mopology_itr);
        dilate(white_mask,white_mask,Mat(),Point(-1,-1),mopology_itr);
        hsv.setTo(Scalar(0,0,255),white_mask);
        hsv.setTo(Scalar(255,255,255),yellow_mask);

        cvtColor(frame.rowRange(roi_height_y, height),gray,CV_BGR2GRAY);
        GaussianBlur(gray,gaussian,Size(3,3),gaussian_sgm);
        Canny(gaussian,canny_t,canny_t1,canny_t2);

        vector<Vec4i> lines;
        HoughLinesP(canny_t,lines,1,CV_PI/180,line_tsh,line_min_length,line_max_gap);
        double y_sum = 0;
        int y_n = 0;
        double w_sum = 0;
        int w_n = 0;
        int left_min = width;
        int right_min = width;
        Point left_line[2];
        Point right_line[2];
        for(Vec4i l : lines)
        {
            angle = line_ingradient(l[0],l[1],l[2],l[3]);
            if(fabs(angle)>0.5 && fabs(angle) < 5.6){
                line(white, Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),1,LINE_AA);

                /*if(((hsv.at<Vec3b>(Point(l[0],l[1])) == c_white) && (hsv.at<Vec3b>(Point(l[2],l[3])) == c_white))) 
                {
                     if(angle<0){
                        line(white, Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,255,0),1,LINE_AA);
                        y_sum += ((roi_height-l[1])*((l[0]-l[2])/(l[1]-l[3]))+l[0]);
                        y_n++;
                    }
                }
                else if(((hsv.at<Vec3b>(Point(l[0],l[1])) == c_yellow) && (hsv.at<Vec3b>(Point(l[2],l[3])) == c_yellow))){
                    if(angle>0){
                        line(white, Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),1,LINE_AA);
                        w_sum += ((roi_height-l[1])*((l[0]-l[2])/(l[1]-l[3]))+l[0]);
                        w_n++;
                    }
                }*/
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

        /*int y_avg = (int)(y_sum/y_n);
        int w_avg = (int)(w_sum/w_n);
        if((((y_avg+w_avg)/2) < (width/2)+100) && (((y_avg+w_avg)/2) > (width/2)-100)){
            center = (y_avg+w_avg)/2;
        }
        line(gray,Point(y_avg,0),Point(y_avg,roi_height),Scalar(0,0,0),2);
        line(gray,Point(w_avg,0),Point(w_avg,roi_height),Scalar(0,0,0),2);
        line(gray,Point(center,0),Point(center,roi_height),Scalar(0,0,0),2);
        cout << (width/2) -center <<endl;
        */
       int left_dx = left_line[0].x - left_line[1].x;
       int left_dy = left_line[0].y - left_line[1].y;
       int right_dx = right_line[0].x - right_line[1].x;
       int right_dy = right_line[0].y - right_line[1].y;

       left_line[0].x += left_dx * alpha;
       left_line[0].y += left_dy * alpha;
       left_line[1].x -= left_dx * alpha;
       left_line[1].y -= left_dy * alpha;

       right_line[0].x += right_dx *alpha;
       right_line[0].y += right_dy * alpha;
       right_line[1].x -= right_dx * alpha;
       right_line[1].y -= right_dy * alpha;

        line(canny_t,Point(width/2,0),Point(width/2,roi_height),Scalar(255,255,255),2);
        line(canny_t,Point(0,roi_height/3),Point(width,roi_height/3),Scalar(255,255,255),2);
        line(gray,left_line[0],left_line[1],Scalar(0,0,0),2);
        line(gray,right_line[0],right_line[1],Scalar(0,0,0),2);
      
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