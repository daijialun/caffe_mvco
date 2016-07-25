#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/scharr_layer.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

   template <typename Dtype>
   void ScharrLayer<Dtype> :: Reshape(const vector<Blob<Dtype>*>& bottom, 
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
   void ScharrLayer<Dtype> :: Forward_cpu( const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top)   {


              // Blob to Mat to Blob
             for(int i=0; i<bottom.size(); i++)  {
                const int batch_size = bottom[i]->num();
                const int channel_num = bottom[i]->channels();
                const int height = bottom[i]->height();
                const int width = bottom[i]->width();

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
                     // **** bottom_image_data to Mat ***** //
                    cv::Mat src;
                    cv::merge(channels, src);

                    // ******* 双边滤波 ****** //
                    cv::Mat filter;
                    cv::bilateralFilter(src, filter, 10, 10*2, 10/2 );

                    // ******* Scharr ******* //
                    cv::Mat scharrGX, scharrGY;
                    cv::Mat scharrAGX, scharrAGY;
                    cv::Scharr(filter, scharrGX, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT );
                    cv::convertScaleAbs(scharrGX, scharrAGX);
                    cv::Scharr(filter, scharrGY, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT);
                    cv::convertScaleAbs(scharrGY, scharrAGY);
                    cv::addWeighted(scharrAGX, 0.5, scharrAGY, 0.5, 0, src);

                    // Mat to top_blob
                    int top_index=0;                  
                    Dtype* top_image_data = top[i]->mutable_cpu_data() + top[i]->offset(n);
                    for(int h=0; h<height; ++h)  {
                         uchar* ptr = src.ptr<uchar>(h);
                         int img_index=0;
                         for(int c=0; c<channel_num; ++c)  {
                                for(int w=0; w<width; ++w)  {
                                    //top_index = ( c * height + h) * width + w;
                                    top_index = ( h * channel_num + c ) * width + w;
                                    top_image_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
                                }   // channel
                          } // width
                    }  // height
              } // batch_size
       } // bottom.size()
   }

#ifdef CPU_ONLY
STUB_GPU(ScharrLayer);
#endif


INSTANTIATE_CLASS(ScharrLayer);
REGISTER_LAYER_CLASS(Scharr);
} // namespace caffe

