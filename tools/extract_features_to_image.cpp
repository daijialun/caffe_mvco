#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::vector;
using std::string;
using cv::Mat;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }

LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);

int arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);


  LOG(ERROR)<< "Extacting Features";


  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_features; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);

  // ************** Modified *****************//
        const int batch_size = feature_blob->num();
        const int channels = feature_blob->channels();
        const int height = feature_blob->height();
        const int width = feature_blob->width();
        const Dtype* feature_blob_data;
        vector<Mat> mat_out_feature_image;
        for(int n=0; n<batch_size; n++)  {
            feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
            Mat mat_channel_image(height, width, CV_32FC1);
            float* _dst_data = reinterpret_cast<float *>(mat_channel_image.data);
            for(int c=0; c<channels; ++c)  {
              caffe::caffe_copy(height*width, feature_blob_data, _dst_data);
                mat_out_feature_image.push_back(mat_channel_image);
            }

           // std::cout << "well done " << n << std::endl;
      } //for (int n = 0; n < batch_size; ++n)
      std::cout <<mat_out_feature_image.size() << std::endl;


      for(int i=0; i<mat_out_feature_image.size(); i++)  {
            //Mat img;
            //mat_out_feature_image[i].convertTo(img, CV_8UC1);
            std::cout << i << std::endl;
            mat_out_feature_image[i].convertTo(mat_out_feature_image[i], CV_8U);
            cv::imshow("Mat feature", mat_out_feature_image[i]);
            cv::waitKey();
     }
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
