/* includes //{ */

/* some STL includes */
#include <stdlib.h>
#include <stdio.h>
#include <mutex>

/* some OpenCV includes */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<thread>
#include<string>
#include<vector>

/* callbackImage() method //{ */
class Detector
{
private:
 /* ros parameters */
  bool _gui_ = true;


   // ---------------------Color parameters----------------------------|
  const cv::Scalar                  color_red_one_min = cv::Scalar(0,100,75);        //RED
  const cv::Scalar                  color_red_one_max = cv::Scalar(5,255,255);     //RED

  const cv::Scalar                  color_red_two_min = cv::Scalar(175,100,75);      //RED
  const cv::Scalar                  color_red_two_max = cv::Scalar(180,255,255);    //RED
    
  const cv::Scalar                  color_blue_min = cv::Scalar(75,100,75);        //BLUE
  const cv::Scalar                  color_blue_max = cv::Scalar(150,255,255);       //BLUE
  
  const cv::Scalar                  color_orange_min = cv::Scalar(15,150,150);       //ORANGE
  const cv::Scalar                  color_orange_max = cv::Scalar(30,255,255);     //ORANGE
            
  const cv::Scalar                  color_yellow_min = cv::Scalar(25,150,150);       //YELLOW
  const cv::Scalar                  color_yellow_max = cv::Scalar(35,255,255);     //YELLOW
 
  const cv::Scalar                  color_green_min = cv::Scalar(35,150,150);      //GREEN
  const cv::Scalar                  color_green_max = cv::Scalar(75,255,255);      //GREEN
  
  const cv::Scalar                  color_purple_min = cv::Scalar(150,100,75);      //PURPLE
  const cv::Scalar                  color_purple_max = cv::Scalar(175,255,255);    //PURPLE
  
  const cv::Scalar                  color_black_min = cv::Scalar(0,0,0);           //BLACK
  const cv::Scalar                  color_black_max = cv::Scalar(180,255,30);      //BLACK
 

  // in BGR
  const cv::Scalar                  detection_color_blue = cv::Scalar(255,100,0);
  const cv::Scalar                  detection_color_red = cv::Scalar(0,0,255);
  const cv::Scalar                  detection_color_yellow = cv::Scalar(0,255,255);
  const cv::Scalar                  detection_color_orange = cv::Scalar(13,143,255);
  const cv::Scalar                  detection_color_purple = cv::Scalar(255,0,255);
  
  int blob_size = 200;   
  
  // | --------- Blob Parameters -------------------------------- |
  cv::Point2d statePt2D;
  cv::Point3d center3D;
  cv::Mat frame;
  cv::VideoCapture cap;
  int apiID = cv::CAP_ANY;      // 0 = autodetect default API
  int deviceID = 0;             // 0 = open default camera
  // | --------------------------- gui -------------------------- |
public:
   Detector(){
      if (_gui_) {

      int flags = cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED;
      cv::namedWindow("detected_objects", flags);
      }

      // | -------------- Detect blob using OpenCV --------------------------------|

      //--- INITIALIZE VIDEOCAPTURE
      // open the default camera using default API
      // cap.open(0);
      // OR advance usage: select any API backend
      // open selected camera using selected API
      cap.open(deviceID, apiID);
      // check if we succeeded
      if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
      }
    }
      void GrabRGBD();
      cv::Mat GaussianBlur(cv::Mat image);
      cv::Mat BGRtoHSV(cv::Mat image);
      cv::Mat ReturnColorMask(cv::Mat image);
      cv::Mat ReturnRedMask(cv::Mat image);
      cv::Mat ReturnOrangeMask(cv::Mat image);
      cv::Mat ReturnYellowMask(cv::Mat image);
      cv::Mat ReturnPurpleMask(cv::Mat image);
      cv::Mat ReturnBlueMask(cv::Mat image);
      std::vector<std::vector<cv::Point>> ReturnContours(cv::Mat image_threshold);
      cv::Point2f FindCenter(std::vector<std::vector<cv::Point>> contours, int ID);
      float FindRadius(std::vector<std::vector<cv::Point>> contours, int ID);
      int FindMaxAreaContourId(std::vector<std::vector<cv::Point>> contours);
    
      
};

void Detector::GrabRGBD() {

  const std::string color_encoding     = "bgr8";
  const std::string grayscale_encoding = "mono8";

 
  
  // wait for a new frame from camera and store it into 'frame'
  cap.read(frame);
  cv::Mat  cv_image = frame;
  // -->> Operations on image ----
//   cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
  // 1) smoothing
  cv::Mat     blurred_image   = Detector::GaussianBlur(cv_image);
  // 2) conversion to hsv
  cv::Mat     image_HSV       = Detector::BGRtoHSV(cv_image);
  // 3) finding mask
  cv::Mat     red_mask        = Detector::ReturnRedMask(image_HSV);
  cv::Mat     blue_mask       = Detector::ReturnBlueMask(image_HSV);
  // cv::Mat     yellow_mask     = ReturnYellowMask(image_HSV);
  cv::Mat     purple_mask     = Detector::ReturnPurpleMask(image_HSV);
  // cv::Mat     orange_mask     = ReturnOrangeMask(image_HSV);
  // 4) finding contours
  std::vector<std::vector<cv::Point>> contours_red = Detector::ReturnContours(red_mask);
  std::vector<std::vector<cv::Point>> contours_blue = Detector::ReturnContours(blue_mask);
  std::vector<std::vector<cv::Point>> contours_purple = Detector::ReturnContours(purple_mask);
  // std::vector<std::vector<cv::Point>> contours_orange = ReturnContours(orange_mask);
  
  // Image for detections
  cv::Mat drawing = cv::Mat::zeros(cv_image.size(), CV_8UC3 );
    // Red Mask
    if (contours_red.size()>0)
    {
      for (size_t i = 0;i<contours_red.size();i++)
      {
              double newArea = cv::contourArea(contours_red.at(i));
              if(newArea > blob_size)
              {   
                  // Finding blob's center       
                  cv::Point2f center = Detector::FindCenter(contours_red, i);
                  center3D.x = center.x;
                  center3D.y = center.y;
      
                  // Drawing 
                  statePt2D.x = center.x;
                  statePt2D.y = center.y;
                  cv::circle  (drawing, statePt2D, 5, detection_color_red, 10);
                  float radius = Detector::FindRadius(contours_red, i);
                  cv::circle  (drawing, statePt2D, int(radius), detection_color_red, 2 );
              }
      }
    }
    // Blue Mask

    if (contours_blue.size()>0)
    {
      for (size_t j = 0;j<contours_blue.size();j++)
      {
              double newArea = cv::contourArea(contours_blue.at(j));
              if(newArea > blob_size)
              {   
                  // Finding blob's center       
                  cv::Point2f center = Detector::FindCenter(contours_blue, j);
                  center3D.x = center.x;
                  center3D.y = center.y;
                  
                  // Drawing 
                  statePt2D.x = center.x;
                  statePt2D.y = center.y;
                  cv::circle  (drawing, statePt2D, 5, detection_color_blue, 10);
                  float radius = Detector::FindRadius(contours_blue, j);
                  cv::circle  (drawing, statePt2D, int(radius), detection_color_blue, 2 );
              }
      }
    } 
    // Purple mask
    if (contours_purple.size()>0)
    {
      for (size_t n = 0;n<contours_purple.size();n++)
      {
              double newArea = cv::contourArea(contours_purple.at(n));
              if(newArea > blob_size)
              {   
                  // Finding blob's center       
                  cv::Point2f center = Detector::FindCenter(contours_purple, n);
                  center3D.x = center.x;
                  center3D.y = center.y;
      
                  // Drawing 
                  statePt2D.x = center.x;
                  statePt2D.y = center.y;
                  cv::circle  (drawing, statePt2D, 5, detection_color_purple, 10);
                  float radius = Detector::FindRadius(contours_purple, n);
                  cv::circle  (drawing, statePt2D, int(radius), detection_color_purple, 2 );
              }
      }
    }

  /* show the image in gui (!the image will be displayed after calling cv::waitKey()!) */

  // if (_gui_) {
    // cv::imshow("original",cv_image);
  // }


  /* show the projection image in gui (!the image will be displayed after calling cv::waitKey()!) */

  if (_gui_) {
    cv::imshow("detected_objects", cv_image+drawing);
  }
  // | ------------------------------------------------------------|
  if (_gui_) {
    /* !!! needed by OpenCV to correctly show the images using cv::imshow !!! */
    cv::waitKey(1);
  }
}


/*| --------- BlobDet Function --------------------------------|*/

cv::Mat Detector::GaussianBlur(cv::Mat image)
{
      cv::Mat image_blurred;
      cv::GaussianBlur(image, image_blurred, cv::Size(5,5), 0);
      return  image_blurred;
}

cv::Mat Detector::BGRtoHSV(cv::Mat image)
{
      cv::Mat image_HSV;
      cv::cvtColor(image, image_HSV,cv::COLOR_BGR2HSV);
      return  image_HSV;
}

cv::Mat Detector::ReturnColorMask(cv::Mat image)
{
      cv::Mat          mask1,mask2,mask3,mask4,mask5,mask6,total;
      cv::inRange     (image, color_blue_min, color_blue_max, mask1);
      cv::inRange     (image, color_orange_min, color_orange_max, mask2);
      cv::inRange     (image, color_yellow_min, color_yellow_max, mask3);
      cv::inRange     (image, color_purple_min, color_purple_max, mask4);
      cv::inRange     (image, color_red_one_min, color_red_one_max, mask5);
      cv::inRange     (image, color_red_two_min, color_red_two_max, mask6); 

      total = mask1 | mask2 | mask3 | mask4 | mask5 | mask6;
      return total;
}
cv::Mat Detector::ReturnRedMask(cv::Mat image)
{
      cv::Mat          mask1,mask2,total;
      cv::inRange     (image, color_red_one_min, color_red_one_max, mask1);
      cv::inRange     (image, color_red_two_min, color_red_two_max, mask2);
      total = mask1 | mask2;
      return total;
}
cv::Mat Detector::ReturnBlueMask(cv::Mat image)
{
      cv::Mat total;
      cv::inRange  (image, color_blue_min, color_blue_max,total); 
      return total;
}

cv::Mat Detector::ReturnOrangeMask(cv::Mat image)
{
      cv::Mat total;
      cv::inRange  (image, color_orange_min, color_orange_max,total); 
      return total;
}
cv::Mat Detector::ReturnYellowMask(cv::Mat image)
{
      cv::Mat total;
      cv::inRange  (image, color_yellow_min, color_yellow_max,total); 
      return total;
}
cv::Mat Detector::ReturnPurpleMask(cv::Mat image)
{
      cv::Mat total;
      cv::inRange  (image, color_purple_min, color_purple_max,total); 
      return total;
}

std::vector<std::vector<cv::Point>> Detector::ReturnContours(cv::Mat image_threshold)
{
      std::vector<std::vector<cv::Point>> contours;       //contours are stored here
      std::vector<cv::Vec4i>              hierarchy;
      cv::findContours(image_threshold, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
      return contours;
}

cv::Point2f Detector::FindCenter(std::vector<std::vector<cv::Point>> contours, int ID)
{
      std::vector<cv::Point>  contour_poly   (contours.size());
      cv::Point2f             center         (contours.size());
      float                   radius         (contours.size());

      cv::approxPolyDP        (contours[ID], contour_poly, 3, true);
      cv::minEnclosingCircle  (contour_poly, center, radius);
      return center;
}

float Detector::FindRadius(std::vector<std::vector<cv::Point>> contours, int ID)
{
      std::vector<cv::Point>  contour_poly   (contours.size());
      cv::Point2f             center         (contours.size());
      float                   radius         (contours.size());

      cv::approxPolyDP        (contours[ID], contour_poly, 3, true);
      cv::minEnclosingCircle  (contour_poly, center, radius);
      return radius;
}

int Detector::FindMaxAreaContourId(std::vector<std::vector<cv::Point>> contours)
{
      // Function for finding maximal size contour
      double  maxArea          = 0;
      int     maxAreaContourId = -1;

      for (size_t i = 0;i<contours.size();i++)
      {
              double   newArea = cv::contourArea(contours.at(i));
              if(newArea > maxArea)
              {
                      maxArea = newArea;
                      maxAreaContourId = i;
              }
      }
      return maxAreaContourId;
}

/*  BlobDet //{ */

int main()
{
 

  Detector bd;
  while(true)
  {
      bd.GrabRGBD();
  }
}



