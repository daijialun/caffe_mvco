#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/three_base_data_layer.hpp"
#include "caffe/layers/three_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ThreeImageDataLayer<Dtype>::~ThreeImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ThreeImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.three_image_data_param().new_height();
  const int new_width  = this->layer_param_.three_image_data_param().new_width();
  const bool is_color  = this->layer_param_.three_image_data_param().is_color();
  string root_folder = this->layer_param_.three_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.three_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.three_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.three_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.three_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);



  // ***** Origin, Local and Global ***** //
  this->vec_transformed_data_[0].Reshape(top_shape);
  this->vec_transformed_data_[1].Reshape(top_shape);
  this->vec_transformed_data_[2].Reshape(top_shape);


  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.three_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;

  // ***** data_ should be three times than normal ***** //
  vector<int> three_top_shape(top_shape);
  three_top_shape[1] *= 3;

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(three_top_shape);
  }

  
  // ***** Origin, Local and Global ***** //
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
  top[2]->Reshape(top_shape);

  // ****** Origin Size *****//
  LOG(INFO) << "output origin size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // ****** Local Size *****//
  LOG(INFO) << "output local size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();

  // ****** Global Size *****//
  LOG(INFO) << "output global size: " << top[2]->num() << ","
      << top[2]->channels() << "," << top[2]->height() << ","
      << top[2]->width();

  // label
  vector<int> label_shape(1, batch_size);
  top[3]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ThreeImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ThreeImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->vec_transformed_data_[0].count());
  CHECK(this->vec_transformed_data_[1].count());
  CHECK(this->vec_transformed_data_[2].count());
  ThreeImageDataParameter three_image_data_param = this->layer_param_.three_image_data_param();
  const int batch_size = three_image_data_param.batch_size();
  const int new_height = three_image_data_param.new_height();
  const int new_width = three_image_data_param.new_width();
  const bool is_color = three_image_data_param.is_color();
  string root_folder = three_image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);

  this->vec_transformed_data_[0].Reshape(top_shape);
  this->vec_transformed_data_[1].Reshape(top_shape);
  this->vec_transformed_data_[2].Reshape(top_shape);

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  vector<int> three_top_shape(top_shape);
  three_top_shape[1] *= 3;
  batch->data_.Reshape(three_top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    // ***** The point position of origin, local  and global ***** // 
    // ***** Origin Location ***** //
    int offset_origin = batch->data_.offset(item_id);
    this->vec_transformed_data_[0].set_cpu_data(prefetch_data + offset_origin);
    // ***** Local Location ***** //
    int offset_local = batch->data_.offset(batch_size + item_id);
    this->vec_transformed_data_[1].set_cpu_data(prefetch_data + offset_local);
    // ***** Global Location ***** //
    int offset_global = batch->data_.offset(batch_size*2 + item_id);
    this->vec_transformed_data_[2].set_cpu_data(prefetch_data + offset_global);

    this->data_transformer_->Transform(cv_img, this->vec_transformed_data_);
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.three_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ThreeImageDataLayer);
REGISTER_LAYER_CLASS(ThreeImageData);

}  // namespace caffe
#endif  // USE_OPENCV

