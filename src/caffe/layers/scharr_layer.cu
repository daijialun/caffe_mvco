#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include <opencv2/cudafilters.hpp>

#include "caffe/layers/scharr_layer.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */
   template <typename Dtype>
   void ScharrLayer<Dtype> :: Forward_gpu( const vector<Blob<Dtype>*>& bottom, 
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
                     // **** Source Image Mat ***** //
                    cv::cudev::GpuMat src;
                    cv::cuda::merge(channels, src);

                    // ***** BilateralFilter ***** //
                    cv::cudev::GpuMat filter;
                    cv::cuda::bilateralFilter(src, filter, 10, 10*2, 10/2 );

                    // ******* Scharr ******* //
                    cv::cudev::GpuMat scharrDst;
                    cv::cudev::GpuMat  scharrGX, scharrGY;
                    cv::cudev::GpuMat  scharrAGX, scharrAGY;
                    cv::Ptr<cv::cuda::Filter> scharrx = cv::cuda::createScharrFilter(CV_32FC1,
			             CV_32FC1, 1, 0, 1, cv::BORDER_DEFAULT);
                    scharrx->apply( filter, scharrGX);
                    cv::cuda::abs(scharrGX, scharrAGX);
                    cv::Ptr<cv::cuda::Filter> scharry = cv::cuda::createScharrFilter(CV_32FC1,
			             CV_32FC1, 0, 1, 1, cv::BORDER_DEFAULT);
                    scharry->apply( filter, scharrGY);
                    cv::cuda::abs(scharrGY, scharrAGY);
                    cv::addWeighted(scharrAGX, 0.5, scharrAGY, 0.5, 0, scharrDst);

                    // Mat to top_blob
                    int top_index=0;                  
                    Dtype* top_image_data = top[i]->mutable_cpu_data() + top[i]->offset(n);
                    for(int h=0; h<height; ++h)  {
                         uchar* ptr = scharrDst.ptr<uchar>(h);
                         int img_index=0;
                         for(int c=0; c<channel_num; ++c)  {
                                for(int w=0; w<width; ++w)  {
                                    //top_index = ( c * height + h) * width + w;
                                    top_index = ( h * channel_num + c ) * width + w;
                                    top_image_data[top_index] = static_cast<Dtype>( cv::saturate_cast<uchar>((300*0.01)*ptr[img_index++]));
                                }   // channel
                          } // width
                    }  // height
              } // batch_size
       } // bottom.size()
   }

INSTANTIATE_LAYER_GPU_FUNCS(ScharrLayer);
} // namespace caffe

