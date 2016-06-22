#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "boost/algorithm/string.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <algorithm>

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

DEFINE_int32(top, 5,
    "Optional: the top correct and wrong class name.");


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
          //LOG(INFO) << "Layer: " << caffe_net.layers().size();
          for( int layer_i=0; layer_i<caffe_net.layers().size(); layer_i++ )  {
              if( string(caffe_net.layers()[layer_i]->type())=="Accuracy" 
                        || string(caffe_net.layers()[layer_i]->type())=="SoftMaxLoss")  {
                    layer_acy = layer_i;
                    break;
              }
         }
         //CHECK_LE( layer_i, caffe_net.layers().size() ) << "Please confirm accuracy or softmaxloss layer in prototxt";
         CHECK_GT(layer_acy, 0) << "Please confirm accuracy or softmaxloss layer in prototxt";

         CHECK_EQ(caffe_net.bottom_vecs()[layer_acy].size(), 2) << "Accuracy Layer input blob size must be 2"; 
        //LOG(INFO) << "Bottom Vectors Size: " << caffe_net.bottom_vecs().size();


          // ******** Initial Setting ******** //
          const int num_labels = caffe_net.bottom_vecs()[layer_acy][0]->channels();
          const int dim = caffe_net.bottom_vecs()[layer_acy][0]->count() / \
                        caffe_net.bottom_vecs()[layer_acy][1]->count();
          const int batch_size = caffe_net.bottom_vecs()[layer_acy][0]->num();

          // ********** Accuracy and Loss Initialization ********** //
          vector<int> test_score_output_id;
          vector<float> test_score;
          float loss = 0;

           // ********** Confusion Matrix Initialization ********** //
          int confus_matrix[dim+1][dim+1];
          for(int i=0; i<dim+1; i++)
              for(int j=0; j<dim+1; j++)  
                  confus_matrix[i][j]=0;
           std::ofstream out("confusion.txt");
           out.clear();

          

           // ********** Main Model ********** //
          for(int k=0; k<FLAGS_iterations; k++)  { 
                float iter_loss;
                const vector<Blob<float>* >& result = caffe_net.Forward(&iter_loss);  // Forward return net_output_blobs_=>loss + accuracy
                loss += iter_loss;
                int idx = 0;


                // ********** Part1: Accuracy and Loss ********** //
                for(int i=0; i<result.size(); i++)  {
                        const float* result_vec = result[i]->cpu_data();
                        for(int j=0; j<result[i]->count(); j++, idx++)  {   // loss, accuracy.cout() => 1 (top5.count()=>1)
                                const float score = result_vec[j];
                                if( k==0 )  {
                                        test_score.push_back(score);
                                        test_score_output_id.push_back(i);
                                }  else {
                                        test_score[idx] += score;
                                }
                                //const std::string& output_name = caffe_net.blob_names()[
                                //                caffe_net.output_blob_indices()[j]];
                                //LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
                        }
                }


                // ********** Part2: Confusion Matrix ********** //
                Blob<float>* blobOutput =  caffe_net.bottom_vecs()[layer_acy][0];   //LOG(INFO)  << blobOutput->count();  50x30x1x1
                Blob<float>* blobLabel =  caffe_net.bottom_vecs()[layer_acy][1];      //LOG(INFO) << blobLabel->count();      50x1x1x1
                //#ifdef CPU_ONLY                
                const float* bottom_data = blobOutput->cpu_data();
                const float* bottom_label = blobLabel->cpu_data();
                //#else
                //const float* bottom_data = blobOutput->gpu_data();
                //const float* bottom_label = blobLabel->gpu_data();//#endif

                for(int i=0; i<batch_size; i++)  {     
                      float max_value = -1;
                      int label_output = 0;
                      const int label_actual = static_cast<int>(bottom_label[i]);
                      for(int j=0; j<dim; j++)  {   
                                if( max_value < bottom_data[ i*dim+j ])  {
                                        max_value = bottom_data[ i*dim+j ];
                                        label_output = j;
                                }
                      }
                      confus_matrix[label_actual][label_output]++;
                }

          }


            // ********** Average Accuracy and Loss Compuation ********** //
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
            }


          // ********** Confusion Matrix Computation ********** //
          for(int i=0; i<dim; i++)  {
                for(int j=0; j<dim; j++)   {
                      confus_matrix[i][dim] += confus_matrix[i][j];
                      confus_matrix[dim][i] += confus_matrix[j][i];
                }
                confus_matrix[dim][dim] += confus_matrix[dim][i];
           }


           // ********** Analyse Top Correct and Top Wrong ********** //
           vector<std::pair<int, float> > id_accuracy;
           float average_accuracy=0;
           vector<string> class_names;
           vector<int> class_num;
           for(int i=0; i<dim; i++)  {
                id_accuracy.push_back( std::pair<int, float>(i, static_cast<float>(confus_matrix[i][i])/static_cast<float>(confus_matrix[i][dim]) ) );
                class_num.push_back(confus_matrix[i][dim]);
                average_accuracy += id_accuracy[i].second;
            }
            average_accuracy /= num_labels;
            if(FLAGS_labels.size()>0)  {
                  std::ifstream labels_file(FLAGS_labels);
                  CHECK_EQ(labels_file.is_open(), 1) << "Please confirm the labels txt exist.";
                  string class_name;
                  while( getline(labels_file, class_name) )  
                          class_names.push_back(class_name);
            }  else  {
                  for(int i=0; i<num_labels; i++)  {
                        string class_name; 
                        std::stringstream ss;
                        ss << i+1;
                        class_name = ss.str() + "th_class";
                        class_names.push_back(class_name);
                  }
            }
            vector<std::pair<int, float> > top_correct(id_accuracy);
            vector<std::pair<int, float> > top_wrong(id_accuracy);
            //std::map<int, float> top_correct(id_accuracy);
            //std::map<int, float> top_wrong(id_accuracy);
            std::sort(top_correct.begin(), top_correct.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) { return a.second > b.second;});
            std::sort(top_wrong.begin(), top_wrong.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) { return a.second < b.second;});
             LOG(INFO) << "Top 5 Correct Class: ";
            for(int i=0; i<5; i++)  {
                  LOG(INFO) << i+1 << ": " << class_names[top_correct[i].first] << top_correct[i].second;
            }
            LOG(INFO) << "Top 5 Wrong Class: ";
            for(int i=0; i<5; i++)  {
                  LOG(INFO) << i+1 << ": " << class_names[top_wrong[i].first] << top_wrong[i].second;
            }

        
        // ********** Write Confusion Matrix To File ********** //

            for(int i=0; i<dim+1; i++)  {
                  for(int j=0; j<dim+1; j++)   {
                      //std::cout.setf(std::ios::left); 
                      out.setf(std::ios::right);
                      out <<"\t" << confus_matrix[i][j];
                  }
                  if(i==dim)  
                        out << "\t" << average_accuracy <<"\n";
                   else
                        out << "\t" << id_accuracy[i].second <<"\n";
             }
             LOG(INFO) << "Confusion matrix txt has been written.";

             for(auto i:id_accuracy)  
                  std::cout <<i.second << " ";

           
          return 0;
}

