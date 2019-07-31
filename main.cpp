#include<iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("anticlockwise.mp4");
    Mat frame;
    Mat blur;
    Mat sharp;
    Mat binary;
    Mat bin_rg;

    double sigma = 3.0;
    double alpha = 3.0;
    int pos;
    auto width = cap.get(CAP_PROP_FRAME_WIDTH);
    auto height = cap.get(CAP_PROP_FRAME_HEIGHT);

    namedWindow("original");
    resizeWindow("original",width,height);
    namedWindow("blur");
    resizeWindow("blur",width,height);
    namedWindow("sharp");
    resizeWindow("sharp",width, height);
    namedWindow("binary");
    resizeWindow("binary",width,height);
    namedWindow("bin_rg");
    resizeWindow("bin_rg",width,height/2);

    moveWindow("blur",width*1.2,0);
    moveWindow("sharp",0,height*1.4);
    moveWindow("binary",width*1.2,height*1.4);
    moveWindow("bin_rg",width*1.2*2,height*1.4);

    while(1)
    {
        cap >> frame;
        if(frame.empty())
            break;

        GaussianBlur(frame, blur, Size(),sigma);
        sharp = (1+alpha)*frame - alpha * blur;

        cvtColor(frame,binary,CV_RGB2GRAY);
        bin_rg = binary.rowRange(height/2,height).clone();
        adaptiveThreshold(bin_rg,bin_rg,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,7,5);

        imshow("original",frame);
        imshow("blur",blur);
        imshow("sharp",sharp);
        imshow("binary",binary);
        imshow("bin_rg",bin_rg);

        if(waitKey(10) == 27)
            break;
    }
    return 0;


}