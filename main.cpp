/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include "Classify.hpp"
#include "utils.hpp"

namespace fs = boost::filesystem;
using namespace std;

static void Classify(ClassifyNet &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
  ts->save("overall");
  ts->save("stage 1: pre-process");
  net.preForward(images);
  ts->save("stage 1: pre-process");
  ts->save("stage 2: forward  ");
  net.forward();
  ts->save("stage 2: forward  ");
  ts->save("stage 3:post-process");
  vector<st_ClassifyResult> results = net.postForward();
  ts->save("stage 3:post-process");
  ts->save("overall");
  for (size_t i = 0; i < results.size(); i++) {
    cout << names[i] << " class id : " <<
       results[i].class_id << " score : " << results[i].score << endl;
  }
}


int main(int argc, char **argv) {
  cout.setf(ios::fixed);

  if (argc < 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image list> <bmodel file> " << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel file> " << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0) {
    is_video = true;
  } else if (strcmp(argv[1], "image") == 0) {
    is_video = false;
  } else {
    cout << "Wrong input type, neither image nor video." << endl;
    exit(1);
  }

  string image_list = argv[2];
  if (!is_video && !fs::exists(image_list)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  int device_id = 0;
  ClassifyNet net(bmodel_file, device_id);
  int batch_size = net.getBatchSize();
  TimeStamp ts;
  net.enableProfile(&ts);
  char image_path[1024] = {0};
  ifstream fp_img_list(image_list);
  if (!is_video) {
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    while(fp_img_list.getline(image_path, 1024)) {
      ts.save("decode overall");
      ts.save("stage 0: decode");
      cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR, device_id);
      ts.save("stage 0: decode");
      if (img.empty()) {
         cout << "read image error!" << endl;
         exit(1);
      }
      ts.save("decode overall");
      fs::path fs_path(image_path);
      string img_name = fs_path.filename().string();
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if (static_cast<int>(batch_imgs.size()) == batch_size) {
        Classify(net, batch_imgs, batch_names, &ts);
        batch_imgs.clear();
        batch_names.clear();
      }
    }
  } else {
    vector <cv::VideoCapture> caps;
    vector <string> cap_srcs;
    while(fp_img_list.getline(image_path, 1024)) {
      cv::VideoCapture cap;
      cap.open(image_path, 0, device_id);
      cap.set(cv::CAP_PROP_OUTPUT_YUV, 1);
      caps.push_back(cap);
      cap_srcs.push_back(image_path);
    }

    if ((int)caps.size() != batch_size) {
      cout << "video num should equal model's batch size" << endl;
      exit(1);
    }

    uint32_t batch_id = 0;
    const uint32_t run_frame_no = 200;
    uint32_t frame_id = 0;
    while(1) {
      if (frame_id == run_frame_no) {
        break;
      }
      vector<cv::Mat> batch_imgs;
      vector<string> batch_names;
      ts.save("decode overall");
      ts.save("stage 0: decode");
      for (size_t i = 0; i < caps.size(); i++) {
         if (caps[i].isOpened()) {
           int w = int(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
           int h = int(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
           cv::Mat img;
           caps[i] >> img;
           if (img.rows != h || img.cols != w) {
             break;
           }
           batch_imgs.push_back(img);
           batch_names.push_back(to_string(batch_id) + "_" +
                            to_string(i) + "_video.jpg");
           batch_id++;
         }else{
           cout << "VideoCapture " << i << " "
                   << cap_srcs[i] << " open failed!" << endl;
         }
      }
      if ((int)batch_imgs.size() < batch_size) {
        break;
      }
      ts.save("stage 0: decode");
      ts.save("decode overall");
      Classify(net, batch_imgs, batch_names, &ts);
      batch_imgs.clear();
      batch_names.clear();
      frame_id += 1;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("classify");
  ts.show_summary("classify ");
  ts.clear();

  std::cout << std::endl;

  return 0;
}
