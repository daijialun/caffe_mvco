#ifndef CAFFE_PREPROCESS_LAYER_HPP_
#define CAFFE_PREPROCESS_LAYER_HPP_

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
   * @brief   Preprocess the images in network	
   */
   template <typename Dtype>
   class PreprocessLayer : public Layer<Dtype> {
       public:
             explicit PreprocessLayer (const LayerParameter& param) 
                  : Layer<Dtype>(param) {} ;
             virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*>& top) {};
           // LayerSetUp: implements common data layer setup functionality, and 
           // calls DataLayerSetUp to do special data layer setup for layer types.
             virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*>& top) {};

             virtual inline const char* type() const { return "Preprocess"; }
             virtual inline int ExactNumBottomBlobs() const { return 1; }
             virtual inline int ExactNumTopBlobs() const { return 1; }
             //virtual inline int MinBottomBlobs() const { return 1; }
             //virtual inline int MaxTopBlobs() const { return 1; }

       protected:     
             virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) {};

             virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) {} ;

   };
} // namespace caffe

#endif