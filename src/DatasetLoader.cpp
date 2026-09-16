/**
AUTHOR: ROSSETTO LUCA
*/

#include "DatasetLoader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

bool DatasetLoader::parseGroundTruth(const std::string& txt_path, int img_w, int img_h,
                                     int& label, cv::Rect& bbox) {
    std::ifstream in_file(txt_path);
    if (!in_file.is_open()) {
        std::cerr << "[ERROR] Cannot open txt file: " << txt_path << std::endl;
        return false;
    }

    float xc, yc, w, h;
    if (in_file >> label >> xc >> yc >> w >> h) {
        float abs_xc;
        if (xc <= 1.0f) {
            abs_xc = xc * (float)img_w;
        } else {
            abs_xc = xc;
        }

        float abs_yc;
        if (yc <= 1.0f) {
            abs_yc = yc * (float)img_h;
        } else {
            abs_yc = yc;
        }

        float abs_w;
        if (w <= 1.0f) {
            abs_w = w * (float)img_w;
        } else {
            abs_w = w;
        }

        float abs_h;
        if (h <= 1.0f) {
            abs_h = h * (float)img_h;
        } else {
            abs_h = h;
        }

        int x = (int)(abs_xc - abs_w / 2.0f);
        int y = (int)(abs_yc - abs_h / 2.0f);
        int width = (int)abs_w;
        int height = (int)abs_h;

        cv::Rect raw_box(x, y, width, height);
        cv::Rect img_box(0, 0, img_w, img_h);

        bbox = raw_box & img_box;
        return true;
    }
    return false;
}

std::vector<SequenceData> DatasetLoader::loadDataset(const std::string& dataset_path) {
    std::vector<SequenceData> dataset;

    if (!fs::exists(dataset_path)) {
        std::cerr << "[ERROR] Folder does not exist: " << dataset_path << std::endl;
        return dataset;
    }

    for (const fs::directory_entry& class_dir : fs::directory_iterator(dataset_path)) {
        if (!class_dir.is_directory()) {
            continue;
        }

        for (const fs::directory_entry& seq_dir : fs::directory_iterator(class_dir.path())) {
            if (!seq_dir.is_directory()) {
                continue;
            }

            SequenceData seq;
            seq.sequence_name = seq_dir.path().filename().string();

            fs::path frames_path = seq_dir.path() / "data";
            fs::path labels_path = seq_dir.path() / "labels";

            if (!fs::exists(frames_path)) {
                frames_path = seq_dir.path();
            }
            if (!fs::exists(labels_path)) {
                labels_path = seq_dir.path();
            }

            std::vector<std::string> frame_files;
            if (fs::exists(frames_path) && fs::is_directory(frames_path)) {
                for (const fs::directory_entry& f : fs::directory_iterator(frames_path)) {
                    std::string ext = f.path().extension().string();
                    
                    // Conversione in minuscolo con ciclo classico
                    for (size_t i = 0; i < ext.length(); i++) {
                        ext[i] = (char)std::tolower(ext[i]);
                    }

                    if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                        frame_files.push_back(f.path().string());
                    }
                }
            }

            std::sort(frame_files.begin(), frame_files.end());

            for (size_t i = 0; i < frame_files.size(); i++) {
                std::string f_path = frame_files[i];
                cv::Mat frame_img = cv::imread(f_path);
                if (!frame_img.empty()) {
                    seq.frames.push_back(frame_img);
                }
            }

            std::string gt_file = "";
            if (fs::exists(labels_path) && fs::is_directory(labels_path)) {
                for (const fs::directory_entry& f : fs::directory_iterator(labels_path)) {
                    if (f.path().extension().string() == ".txt") {
                        gt_file = f.path().string();
                        break;
                    }
                }
            }

            if (seq.frames.empty() || gt_file.empty()) {
                continue;
            }

            int img_w = seq.frames[0].cols;
            int img_h = seq.frames[0].rows;
            if (parseGroundTruth(gt_file, img_w, img_h, seq.class_label, seq.median_bbox)) {
                dataset.push_back(seq);
            }
        }
    }

    return dataset;
}