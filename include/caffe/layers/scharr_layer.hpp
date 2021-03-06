#ifndef CAFFE_SCHARR_LAYER_HPP_
#define CAFFE_SCHARR_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */
   template <typename Dtype>
   class ScharrLayer : public Layer<Dtype> {
       public:
             explicit ScharrLayer (const LayerParameter& param) 
                  : Layer<Dtype>(param) {} ;
             virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top);
           // LayerSetUp: implements common data layer setup functionality, and 
           // calls DataLayerSetUp to do special data layer setup for layer types.

             virtual inline const char* type() const { return "Scharr"; }
             virtual inline int ExactNumBottomBlobs() const { return 1; }
             virtual inline int ExactNumTopBlobs() const { return 1; }
             //virtual inline int MinBottomBlobs() const { return 1; }
             //virtual inline int MaxTopBlobs() const { return 1; }

       protected:     
             virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) ;
             virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) ;

             virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {};
             virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {};

   };
} // namespace caffe

#endif