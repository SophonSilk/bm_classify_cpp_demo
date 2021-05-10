//
//  Classify.hpp
//  ResnetClassify
//
//  Created by Bitmain on 2021/5/9.
//  Copyright © 2021年 AnBaolei. All rights reserved.
//

#ifndef Classify_hpp
#define Classify_hpp

#include <string>
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"

typedef struct __tag_st_ClassifyResult{
  int class_id;
  float score;
}st_ClassifyResult;

class ClassifyNet {
public:
  ClassifyNet(const std::string bmodel, int device_id);
  ~ClassifyNet();
  void preForward(std::vector<cv::Mat>& images);
  void forward();
  std::vector<st_ClassifyResult> postForward();
  void enableProfile(TimeStamp *ts);
  int getBatchSize();

private:
  void preprocess(bm_image& in, bm_image& out);
  // handle of runtime contxt
  void *p_bmrt_;
  // handle of low level device 
  bm_handle_t bm_handle_;
  // model info 
  const bm_net_info_t *net_info_;
  // indicate current bmodel type INT8 or FP32
  bool is_int8_;
  // buffer of inference results
  float *output_;
  // input image shape used for inference call
  bm_shape_t input_shape_;
  // bm image objects for storing intermediate results
  bm_image* scaled_inputs_;
  // linear transformation arguments of BMCV
  bmcv_convert_to_attr linear_trans_param_;
  // for profiling
  TimeStamp *ts_ = nullptr;

  int batch_size_;
  const char **net_names_;
  int net_h_;
  int net_w_;
  int out_count_;
  int class_num_;
};

#endif /* Classify_hpp */
