#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/preprocess_layer.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

   template <typename Dtype>
   void PreprocessLayer<Dtype> :: Reshape(const vector<Blob<Dtype>*>& bottom, 
            const vector<Blob<Dtype>*>& top)  {
            for(int i=0; i<bottom.size(); i++)  {
                CHECK_EQ( 4, bottom[0]->num_axes() ) << "Input must have 4 axes.";
                const int channels_  = bottom[i]->channels();
                const int height_ = bottom[i]->height();
                const int width_ = bottom[i]->width();
                top[0]->Reshape(bottom[i]->num(), channels_, height_, width_);  
                if(top.size()>1)  {
                      top[1]->ReshapeLike(*top[i]);
                }                     
            }
   }

   template <typename Dtype>
   void PreprocessLayer<Dtype> :: Forward_cpu( const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top)   {

              /*for(int i=0; i<bottom.size(); i++)  
                  top[i]->CopyFrom((*bottom[i]));*/

              // Blob to Mat to Blob
             for(int i=0; i<bottom.size(); i++)  {

                const int batch_size = bottom[i]->num();
                const int channel_num = bottom[i]->channels();
                const int height = bottom[i]->height();
                const int width = bottom[i]->width();

                std::cout << batch_size << " " << channel_num << " " << height << " " << width << std::endl;
                CHECK( height>0 && width>0 && channel_num>0 && batch_size>0) << "One of Num, "
                  "Channel, Height or Width is set as zero.";

                // 一个bottom中有 batch_size 张图像
                for(int n=0; n<batch_size; ++n)  {
                    Dtype* bottom_image_data = bottom[i]->mutable_cpu_data() + bottom[i]->offset(n);
                    vector<cv::Mat> channels;
                    // 处理 channel_num 个通道
                    for(int c=0; c<channel_num; ++c)  {                     
                      cv::Mat channel(height, width, CV_32FC1, bottom_image_data);
                      channels.push_back(channel);
                      bottom_image_data += height*width;
                    }
                     // 将bottom_image_data 转化为 Mat 图像
                    cv::Mat img;
                    cv::merge(channels, img);
                    
                    // ******* Canny  ******* //
                    cv::Mat CannyEdge;
                    img.convertTo(img, CV_8U);
                    cv::blur( img, CannyEdge, cv::Size(3,3) );
                    cv::Canny( CannyEdge, CannyEdge, 15, 15*3, 3 );
                    cv::Mat dstCanny(img.size(), img.type());
                    dstCanny= cv::Scalar::all(0);
                    img.copyTo( dstCanny, CannyEdge);

                    //*******Shapen********//
                    //cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0,-1,5,-1,0,-1,0);
                    //cv::filter2D(img, img_aug, img.depth(), kernel);
                    // 图像处理结果为src
                    cv::bilateralFilter(dstCanny, img, 15, 15*2, 15/2 );
                    cv::imshow("img", img);
                    cv::waitKey();

                    // 将 Mat 图像 img 转化为 top_blob
                    int top_index=0;                  
                    Dtype* top_image_data = top[i]->mutable_cpu_data() + top[i]->offset(n);
                    for(int h=0; h<height; ++h)  {
                         uchar* ptr = img.ptr<uchar>(h);
                         int img_index=0;
                         for(int c=0; c<channel_num; ++c)  {
                                for(int w=0; w<width; ++w)  {
                                    //top_index = ( c * height + h) * width + w;
                                    top_index = ( h * channel_num + c ) * width + w;
                                    top_image_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
                               }   // channel
                          } // width
                    }  // height

                    // *********** test ************//
                    top_image_data = top[i]->mutable_cpu_data() + top[i]->offset(n);
                    vector<cv::Mat> tests;                  
                    for(int c=0; c<channel_num; ++c)  {                     
                      cv::Mat channel(height, width, CV_32FC1, top_image_data);
                      tests.push_back(channel);
                      top_image_data += height*width;
                    }
                    cv::Mat test;
                    cv::merge(tests, test);
                    test.convertTo(test, CV_8U);
                    cv::imshow("transform", test);
                    cv::waitKey();


                } // batch_size
             } // bottom.size()
   }


INSTANTIATE_CLASS(PreprocessLayer);
REGISTER_LAYER_CLASS(Preprocess);
} // namespace caffe

