#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/mean.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */

      template <typename Dtype>
       void MeanLayer<Dtype> :: LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                   const vector<Blob<Dtype>*>& top)  {
             MeanParameter mean_param = this->layer_param_.mean_param();
             // If use mean_file
             if( mean_param.has_mean_file() )  {
                   CHECK_EQ(mean_param.mean_value_size(), 0) <<
                         "Cannot specify mean_file and mean_value at the same time"; 
                   const string& mean_file = mean_param.mean_file();
                   if (Caffe::root_solver()) {
                             LOG(INFO) << "Loading mean file from: " << mean_file;
                   }
                   BlobProto blob_proto;
                    ReadProtoFromBinaryFileOrDie( mean_file.c_str(), &blob_proto);
                    data_mean_.FromProto(blob_proto);
             }
             // If use mean_value
             if( mean_param.mean_value_size() > 0)  {
                    CHECK( mean_param.has_mean_file() == false )  << 
                          "Cannot specify mean_file and mean_value at the same time"; 
                    for(int c=0; c< mean_param.mean_value_size(); ++c)  {
                          mean_values_.push_back( mean_param.mean_value(c) );
                    }
             }
       }

       template <typename Dtype>
       void MeanLayer<Dtype> :: Reshape(const vector<Blob<Dtype>*>& bottom, 
             const vector<Blob<Dtype>*>& top)  {

             for(int i=0; i<bottom.size(); i++)  {
                CHECK_EQ( 4, bottom[0]->num_axes() ) << "Input must have 4 axes.";
                const int channels_  = bottom[i]->channels();
                const int height_ = bottom[i]->height();
                const int width_ = bottom[i]->width();
                top[i]->Reshape(bottom[i]->num(), channels_, height_, width_);           
             }
       }

   template <typename Dtype>
   void MeanLayer<Dtype> :: Forward_cpu( const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top)   {  

             MeanParameter mean_param = this->layer_param_.mean_param();

             for(int i=0; i<bottom.size(); i++)  {
                    const int batch_size = bottom[i]->num();
                    const int channel_num = bottom[i]->channels();
                    const int height = bottom[i]->height();
                    const int width = bottom[i]->width();

                    CHECK( height>0 && width>0 && channel_num>0 && batch_size>0) << "One of Num, "
                     "Channel, Height or Width is set as zero.";

                    if( top[i]->count() == 0 )  {
                            top[i]->Reshape(batch_size, channel_num, height, width);
                    }

                   const Dtype* bottom_data = bottom[i]->cpu_data();
                   Dtype* top_data = top[i]->mutable_cpu_data();
                   int dim = data_mean_.count() / data_mean_.num();

                   if( mean_param.has_mean_file() )  {
                          CHECK_EQ( data_mean_.channels(), bottom[i]->channels() ) <<
                                "mean prototxt channels must be equal to bottom channels";                        
                          for(int n=0; n < batch_size; ++n)  {
                                int offset = bottom[i]->offset(n);
                                caffe_sub(dim, bottom_data + offset, 
                                      data_mean_.cpu_data(), top_data + offset);             
                          }
                   }

                   if( mean_param.mean_value_size() > 0 )  {
                          CHECK(mean_values_.size() == 1 || mean_values_.size() == bottom[i]->channels() ) <<
                                "Specify either 1 mean_value or as many as channels: " << channel_num;
                          if( mean_param.mean_value_size() == 1)  {
                                caffe_sub( data_mean_.count(), bottom_data, 
                                      &mean_values_[0], top_data);
                          } else  {
                                 for(int n=0; n < batch_size; ++n)  {
                                       for(int c=0; c < channel_num; ++c)  {
                                             int offset = bottom[i]->offset(n, c);
                                             caffe_sub(height * width, bottom_data + offset, 
                                                   &(mean_values_[c]), top_data + offset);
                                       }  
                                 }
                          }
                    }
             }
       }                   

INSTANTIATE_CLASS(MeanLayer);
REGISTER_LAYER_CLASS(Mean);
} // namespace caffe

