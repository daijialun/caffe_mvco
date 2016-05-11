#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/preprocess_layer.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

   template <typename Dtype>
   void PreprocessLayer<Dtype> :: Reshape(const vector<Blob<Dtype>*>& bottom, 
            const vector<Blob<Dtype>*>& top)  {
            CHECK_EQ( 4, bottom[0]->num_axes() ) << "Input must have 4 axes.";
            const int channels_  = bottom[0]->channels();
            const int height_ = bottom[0]->height();
            const int width_ = bottom[0]->width();
            top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);  
            if(top.size()>1)  {
                  top[1]->ReshapeLike(*top[0]);
            }
   }

   template <typename Dtype>
   void PreprocessLayer<Dtype> :: Forward_cpu( const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top)   {
             for(int i=0; i<bottom.size(); i++)  {

                const int batch_size = bottom[i]->num();
                const int channel_num = bottom[i]->channels();
                const int height = bottom[i]->height();
                const int width = bottom[i]->width();

                CHECK( height>0 && width>0 && channel_num>0 && batch_size>0) << "One of Num, "
                  "Channel, Height or Width is set as zero.";

                Dtype* bottom_image_data = bottom[i]->mutable_cpu_data();

                // 一个bottom中有 batch_size 张图像
                for(int n=0; n<batch_size; ++n)  {
                    bottom_image_data = bottom[i]->mutable_cpu_data() + bottom[i]->offset(n);
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
                    cv::Mat img_aug=img.clone();
                    cv::Mat src=img.clone();
                    cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0,-1,5,-1,0,-1,0);
                    cv::filter2D(img, img_aug, img.depth(), kernel);
                    // 图像处理结果为src
                    cv::bilateralFilter(img_aug, src, 15, 15*2, 15/2 );

                    // 将 Mat 图像 src 转化为 top_blob
                    int top_index=0;                  
                    Dtype* top_image_data = top[i]->mutable_cpu_data();
                    top_image_data = top[i]->mutable_cpu_data() + top[i]->offset(n);
                    for(int h=0; h<height; ++h)  {
                         const Dtype* ptr = src.ptr<Dtype>(h);
                         int img_index=0;
                         for(int w=0; w<width; ++w)  {
                              for(int c=0; c<channel_num; ++c)  {
                                    top_index = ( c * height + h) * width + w;
                                    top_image_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
                               }   // channel
                          } // width
                    }  // height
                } // batch_size
             } // bottom.size()
   }


INSTANTIATE_CLASS(PreprocessLayer);
REGISTER_LAYER_CLASS(Preprocess);
} // namespace caffe

