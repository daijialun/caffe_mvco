#include <glog/gloging>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/comon.hpp"
#include "caffe/layer.hpp"
#include "caffe/protp/caffe.pb.h"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

   template <typename Dtype>
   class PreporcessLayer<Dtype> :: LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)   {

   }

   template <typename Dtype>
   class PreprocessLayer<Dtype> :: Forward_cpu( const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*> top)   {
             //for(int )
             shared_ptr<Blob<Dtype> > bottom_blob = bottom[0];
             shared_ptr<Blob<Dtype> > top_blob = top[0];
             const int batch_size = bottom_blob->num();
             const int channel_num = bottom_blob->channels();
             const int height = bottom_blob>height();
             const int width = bottom_blob->width();

             CHECK( height>0 && width>0 && channels>0 && batch_size>0) << "One of Num, "
                  "Channel, Height or Width is set as zero.";

              Dtype* bottom_image_data = bottom_blob->mutable_cpu_data();
              Dtype* top_image_data = top_blob->mutable_cpu_data();

              // 一个bottom中有 batch_size 张图像
              for(int n=0; n<batch_size; ++n)  {
                  bottom_image_data = bottom_blob->mutable_cpu_data() + bottom_blob->offset(n);
                  vector<Mat> channels;
                  // 处理 channel_num 个通道
                  for(int c=0; c<channel_num; ++c)  {
                      Mat = channel(height, width, CV_32FC1, bottom_image_data);
                      channels.push_back(channel);
                      bottom_image_blob_data += height*width;
                  }
                  // 将bottom_image_data 转化为 Mat 图像
                  Mat img;
                  Merge(channels, img);
                  Mat img_aug=src.clone();
                  Mat src=src.clone();
                  Mat kernel = (Mat_<float>(3,3) << 0,-1,0,-1,5,-1,0,-1,0);
                  filter2D(img, img_aug, img.depth(), kernel);
                  // 图像处理结果为src
                  bilateralFilter(img_aug, src, 15, 15*2, 15/2 );

                  // 将 Mat 图像 src 转化为 top_blob
                  int top_index=0;                  
                  top_image_data = top_blob->mutable_cpu_data() + top_blob->offset(n);
                  for(int h=0; h<height; ++h)  {
                        const uchar* ptr = src.ptr<uchar>(h);
                        int img_index=0;
                        for(int w=0; w<width; ++w)  {
                              for(int c=0; c<channel_num; ++c)  {
                                    top_index = ( c * height + h) * width + w;
                                    top_image_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
                              }   // channel
                        } // width
                   }  // height
              } // batch_size
   }

#ifdef CPU_ONLY
STUB_GPU(PreprocessLayer);
#endif

INSTANTIATE_CLASS(PreprocessLayer);
REGISTER_LAYER_CLASS(Preprocess);
} // namespace caffe

