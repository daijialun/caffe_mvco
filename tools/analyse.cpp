#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "boost/algorithm/string.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using caffe::Net;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");

DEFINE_string(model, "",
    "The model definition protocol buffer text file..");

DEFINE_string(weights, "",
    "The pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");

DEFINE_bool(confusion, false,
    "Optional; set whether save confusion matrix txt.");

DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

DEFINE_string(labels, "",
    "Optional: the names of each class for classification, "
    "default form is expressed by numbers");



// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
    if (FLAGS_gpu == "all") {
          int count = 0;
    #ifndef CPU_ONLY
          CUDA_CHECK(cudaGetDeviceCount(&count));
    #else
          NO_GPU;
    #endif
          for (int i = 0; i < count; ++i) {
                gpus->push_back(i);
          }
    } else if (FLAGS_gpu.size()) {
                  vector<string> strings;
                  boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
                  for (int i = 0; i < strings.size(); ++i) {
                  gpus->push_back(boost::lexical_cast<int>(strings[i]));
              }
    } else {
            CHECK_EQ(gpus->size(), 0);
    }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
        LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
        vector<int> gpus;
        get_gpus(&gpus);
        for (int i = 0; i < gpus.size(); ++i) {
        caffe::Caffe::SetDevice(gpus[i]);
        caffe::Caffe::DeviceQuery();
  }
  return 0;
}


int main(int argc, char** argv) {

        // Print output to stderr (while still logging).
        FLAGS_alsologtostderr = 1;
        // Set version
        gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
        // Usage message.
        gflags::SetUsageMessage("command line brew\n"
        "usage: analyse <args>\n\n"
        "args:\n"
        "  model 		The model definition protocol buffer text file."
        "  weights 		The pretrained model to analyse performace."
        "  gpus 		The GPU device used for analysis.");
        // Run tool or show usage.
        caffe::GlobalInit(&argc, &argv);

        // Analysis Code
        CHECK_GT(FLAGS_model.size(), 0)  << "Need a model definition to score.";
        CHECK_GT(FLAGS_weights.size(), 0) << "NeedNeed model weights to score.";

        vector<int> gpus;
        get_gpus(&gpus);
        if( gpus.size()!=0 )  {
            LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, gpus[0]);
            LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
            Caffe::SetDevice(gpus[0]);
            Caffe::set_mode(Caffe::GPU);
        }  else {
            LOG(INFO) << "Use CPU.";
            Caffe::set_mode(Caffe::CPU);
        }

        // Instantiate caffe net
        Net<float> caffe_net(FLAGS_model,  caffe::TEST);
        caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
         LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

          int layer_acy = 0;
          LOG(INFO) << "Layer: " << caffe_net.layers().size();
          for( int layer_i=0; layer_i<caffe_net.layers().size(); layer_i++ )  {
              if( string(caffe_net.layers()[layer_i]->type())=="Accuracy" 
                        || string(caffe_net.layers()[layer_i]->type())=="SoftMaxLoss")  {
                    layer_acy = layer_i;
                    break;
              }
         }
         //CHECK_LE( layer_i, caffe_net.layers().size() ) << "Please confirm accuracy or softmaxloss layer in prototxt";
         CHECK_GT(layer_acy, 0) << "Please confirm accuracy or softmaxloss layer in prototxt";

         CHECK_EQ(caffe_net.bottom_vecs()[layer_acy], 2) << "Accuracy Layer input blob size must be 2"; 
        //LOG(INFO) << "Bottom Vectors Size: " << caffe_net.bottom_vecs().size();

        //LOG(INFO) << ""
         /*vector<string> layername = caffe_net.layer_names();
          LOG(INFO) << "Layer: " << layername.size();
          for(int i=0; i<layername.size(); i++)  {
              LOG(INFO) << layername[i];
          }*/
         //for( auto layer_i:caffe_net)  {
         //       if( layer_i->type()=="Accuracy" || layer_i->type()=="SoftMaxLoss")
         //}
          /*vector<string> layername = caffe_net.layer_names();
          LOG(INFO) << "Layer: " << layername.size();
          for(int i=0; i<layername.size(); i++)  {
              LOG(INFO) << layername[i];
          }

          vector<string> blobname = caffe_net.blob_names ();
          LOG(INFO) << "Blob: " << blobname.size();
          for(int i=0; i<blobname.size(); i++)  {
              LOG(INFO) << blobname[i];
          }*/

         /*vector<int> test_score_output_id;
         vector<float> test_score;
         float loss = 0;
         for(int i=0; i<FLAGS_iterations; i++)  {         // Test for FLAGS_iterations times (50 times)
                float iter_loss;
                const vector<Blob<float>* >& result = caffe_net.Forward(&iter_loss);  // Forward return net_output_blobs_=>loss + accuracy
                loss += iter_loss;                                                                                    // loss,accuracy->count()=1, softmax->count()=labels.size()
                int idx = 0;
                for(int j=0; j<result.size(); j++)  {
                        const float* result_vec = result[j]->cpu_data();
                        for(int k=0; k<result[j]->count(); k++, idx++)  {   // loss, accuracy.cout() => 1 (top5.count()=>1)
                                const float score = result_vec[k];
                                if( i==0 )  {
                                        test_score.push_back(score);
                                        test_score_output_id.push_back(j);
                                }  else {
                                        test_score[idx] += score;
                                }
                                const std::string& output_name = caffe_net.blob_names()[
                                                caffe_net.output_blob_indices()[j]];
                                LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
                        }
                }
         }
         loss /= FLAGS_iterations;
         LOG(INFO)  << "Loss: " << loss;
         for(int i=0; i<test_score.size(); i++)  {
                const std::string& output_name = caffe_net.blob_names()[
                          caffe_net.output_blob_indices()[test_score_output_id[i]]];
                const float loss_weight = caffe_net.blob_loss_weights()[
                          caffe_net.output_blob_indices()[test_score_output_id[i]]];
                std::ostringstream loss_msg_stream;
                const float mean_score = test_score[i] / FLAGS_iterations;
                if (loss_weight) {
                        loss_msg_stream << " (* " << loss_weight
                                << " = " << loss_weight * mean_score << " loss)";
                }
                LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
          }*/
          return 0;
}

