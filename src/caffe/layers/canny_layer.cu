#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudev.hpp>

#include "caffe/layers/canny_layer.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

   template <typename Dtype>
   void CannyLayer<Dtype> :: Forward_gpu( const vector<Blob<Dtype>*>& bottom, 
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
                    Dtype* bottom_image_data = bottom[i]->mutable_gpu_data() + bottom[i]->offset(n);
                    vector<cv::cudev::GpuMat> channels;
                    // 处理 channel_num 个通道
                    for(int c=0; c<channel_num; ++c)  {                     
                      cv::cudev::GpuMat channel(height, width, CV_32FC1, bottom_image_data);
                      channels.push_back(channel);
                      bottom_image_data += height*width;
                    }
                     // 将bottom_image_data 转化为 Mat 图像
                    cv::cuda::GpuMat img;
                    cv::cuda::merge(channels, img);
                    img.convertTo(img, CV_8UC1);
                    
                    // ******* Canny  ******* //
                    cv::cuda::GpuMat gfilter, CannyEdge, gdst;
                    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3,3));
                    filter->apply(img, gfilter);
                    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(15, 45, 3);
                    canny->detect( gfilter, CannyEdge);
                    cv::cuda::bilateralFilter(CannyEdge, gdst, 15, 15*2, 15/2);

                    // 将 Mat 图像 img 转化为 top_blob
                    int top_index=0;                  
                    Dtype* top_image_data = top[i]->mutable_gpu_data() + top[i]->offset(n);
                    for(int h=0; h<height; ++h)  {
                         uchar* ptr = gdst.ptr<uchar>(h);
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

INSTANTIATE_LAYER_GPU_FUNCS(CannyLayer);
} // namespace caffe

