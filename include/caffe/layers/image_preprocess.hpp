#ifdef USE_OPENCV

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
   class PreprocessLayer : public layer<Dtype> {
       public:
             explicit PreprocessLayer (const LayerParameter& param) 
                  : layer<Dtype>(param) {} ;
           // LayerSetUp: implements common data layer setup functionality, and 
           // calls DataLayerSetUp to do special data layer setup for layer types.
             virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override {};

             virtual inline int ExactNumBottomBlobs() const { return 1; }
             virtual inline int ExactNumTopBlobs() const { return 1; }
             //virtual inline int MinBottomBlobs() const { return 1; }
             //virtual inline int MaxTopBlobs() const { return 1; }

             virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override;

             virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*> top) override;

             virtual inline const char* type() const { return "Preprocess"; }


       protected:
             int channels_;
             int height_, width_;
   };
} // namespace caffe

#endif