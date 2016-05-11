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
           explicit PreprocessLayer () {} ;
           // LayerSetUp: implements common data layer setup functionality, and 
           // calls DataLayerSetUp to do special data layer setup for layer types.
           virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override;
           virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override;
           virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override;
           
           virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                  const vector<Blob<Dtype>*> top) override;

           /*
           // Returns the exact number of bottom and top blobs required by the layer.
           virtual inline int ExactNumBottomBlobs() const { return };
           virtual int ExactNumTopBlobs() const;
           virtual bool EqualNumBottomTopBlobs() const;

           // Called by SetUp() to check that the number of bottom and top Blobs
           virtual void ChechBlobCounts(const vector<Blob<Dtype>*>& bottom, 
           				const vector<Blob<Dtype>*> top) override;  {
           	    if(ExactNumBottomBlobs >= 0)  {
           	    	CHECK_EQ(ExactNumBottomBlobs(), bottom.size() )
           	    	    << type() << " Layer takes" << ExactNumBottomBlobs()
           	    	    << "bottom blob as input.";
           	    }
           	    if(ExactNumTopBlobs >= 0)  {
           	    	CHECK_EQ(ExactNumTopBlobs(), bottom.size() )
           	    	    << type() << " Layer produces" << ExactNumTopBlobs()
           	    	    << "bottom blob as output.";
           	    }
           	    if( EqualNumBottomTopBlobs() )  {
           	    	CHECK_EQ(bottom.size(), top.size()) << type()
           	    	    << " Layer produces one top blob as output for each "
           	    	    << "bottom blob input. ";
           	    }
           }*/

           


   };
} // namespace caffe

#endif