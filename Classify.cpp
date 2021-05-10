//
//  Classify.cpp
//  ResnetClassify
//
//  Created by Bitmain on 2021/5/9.
//  Copyright © 2021年 AnBaolei. All rights reserved.
//

#include "Classify.hpp"

using namespace std;

ClassifyNet::ClassifyNet(const std::string bmodel, int device_id) {
  bool ret = true;

  bm_dev_request(&bm_handle_, device_id);
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_) {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel from file
  ret = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!ret) {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(1);
  }
  
  bmrt_get_network_names(p_bmrt_, &net_names_);
  std::cout << "> Load model " << net_names_[0] << " successfully" << std::endl;

  // get model info by model name
  net_info_ = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  if (NULL == net_info_) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  // get data type
  if (NULL == net_info_->input_dtypes) {
    cout << "ERROR: get net input type failed!" << endl;
    exit(1);
  }

  if (BM_FLOAT32 == net_info_->input_dtypes[0]) {
    is_int8_ = false;
  } else {
    is_int8_ = true;
  }

  auto &input_shape = net_info_->stages[0].input_shapes[0];
  auto &output_shape = net_info_->stages[0].output_shapes[0];
  batch_size_ = input_shape.dims[0];
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  out_count_= bmrt_shape_count(&output_shape);
  class_num_ = output_shape.dims[1];

 // set input shape according to input bm images
  input_shape_ = {4, {batch_size_, 3, net_h_, net_w_}};

  // allocate output buffer
  output_ = new float[out_count_];
  // bm images for storing inference inputs
  bm_image_data_format_ext data_type;
  if (is_int8_) { // INT8
    data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  } else { // FP32
    data_type = DATA_TYPE_EXT_FLOAT32;
  }

  // initialize linear transform parameter
  // - mean value
  // - scale value (mainly for INT8 calibration)
  float input_scale = net_info_->input_scales[0];
  input_scale *= 0.017;
  linear_trans_param_.alpha_0 = input_scale;
  linear_trans_param_.beta_0 = -103.94 * input_scale;
  linear_trans_param_.alpha_1 = input_scale;
  linear_trans_param_.beta_1 = -116.78 * input_scale;
  linear_trans_param_.alpha_2 = input_scale;
  linear_trans_param_.beta_2 = -123.68 * input_scale;

  scaled_inputs_ = new bm_image[batch_size_];
  bm_status_t bm_ret = bm_image_create_batch (bm_handle_,
                               net_h_,
                               net_w_,
                               FORMAT_BGR_PLANAR,
                               data_type,
                               scaled_inputs_,
                               batch_size_);

  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }
}

ClassifyNet::~ClassifyNet() {
  // deinit bm images
  bm_image_destroy_batch (scaled_inputs_, batch_size_);
  if (scaled_inputs_) {
    delete []scaled_inputs_;
  }

  // free output buffer
  delete []output_;

  // deinit contxt handle
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void ClassifyNet::preprocess(bm_image& in, bm_image& out) {
  bm_image_create(bm_handle_, net_h_, net_w_,
             FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &out, NULL);
  bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
  bmcv_image_vpp_convert(bm_handle_, 1, in, &out, &crop_rect);
}

void ClassifyNet::preForward(std::vector<cv::Mat>& images) {
  vector<bm_image> processed_imgs;
  for (size_t i = 0; i < images.size(); i++) {
    bm_image bmimg;
    bm_image processed_img;
    bm_image_from_mat(bm_handle_, images[i], bmimg);
    preprocess(bmimg, processed_img);
    bm_image_destroy(bmimg);
    processed_imgs.push_back(processed_img);
  }
  bmcv_image_convert_to(bm_handle_, batch_size_,
             linear_trans_param_, &processed_imgs[0], scaled_inputs_);

  for (size_t i = 0; i < images.size(); i++) {
    bm_image_destroy(processed_imgs[i]);
  }
}

void ClassifyNet::forward() {
  auto res = bm_inference (p_bmrt_, scaled_inputs_,
              reinterpret_cast<void*>(output_),
              input_shape_, reinterpret_cast<const char*>(net_names_[0]));

  if (!res) {
    cout << "ERROR : inference failed!!"<< endl;
    exit(1);
  }
}

std::vector<st_ClassifyResult> ClassifyNet::postForward() {
  std::vector<st_ClassifyResult> results;
  for (int i = 0; i < batch_size_; i++) {
    st_ClassifyResult result;
    float* image_output = output_ + i * class_num_;
    vector<float> out_val;
    for (int j = 0; j < class_num_; j++) {
      out_val.push_back(image_output[j]);
    }
    auto max_item =  max_element(out_val.begin(), out_val.end());
    result.class_id = distance(out_val.begin(), max_item);
    result.score = *max_item;
    results.push_back(result);
  }
  return results;
}

void ClassifyNet::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}
int ClassifyNet::getBatchSize() {
  return batch_size_;
}


