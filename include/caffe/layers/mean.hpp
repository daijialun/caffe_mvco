#ifndef CAFFE_MEAN_LAYER_HPP_
#define CAFFE_MEAN_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */
   template <typename Dtype>
   class MeanLayer : public Layer<Dtype> {
       public:
             explicit MeanLayer(const LayerParameter& param) 
                    : Layer<Dtype>(param) {}
             virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                   const vector<Blob<Dtype>*>& top);
             virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top);

             virtual inline const char* type() const { return "Mean"; }
             virtual inline int ExactNumBottomBlobs() const { return 1; }
             virtual inline int ExactNumTopBlobs() const { return 1; } 

             virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) ;
             //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
             //       const vector<Blob<Dtype>*>& top) ;

             virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {};
             //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      //const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

       protected:
             Blob<Dtype> data_mean_;
             vector<Dtype> mean_values_;

   };
} // namespace caffe

#endif