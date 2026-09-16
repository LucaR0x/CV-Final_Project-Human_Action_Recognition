/**
AUTHOR: ROSSETTO LUCA
*/

#include "YoloTracker.hpp"
#include <iostream>

YoloTracker::YoloTracker(const std::string& model_path) {
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Could not load YOLO model: " << e.what() << std::endl;
    }
    reset();
}

void YoloTracker::reset() {
    first_frame = true;
    prev_bbox = cv::Rect(0, 0, 0, 0);
}

cv::Rect YoloTracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    if (frame.empty()) return prev_bbox;

    cv::Mat bgr_frame;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr_frame, cv::COLOR_GRAY2BGR);
    } else {
        bgr_frame = frame.clone();
    }

    int img_w = bgr_frame.cols;
    int img_h = bgr_frame.rows;

    cv::Mat blob;
    cv::dnn::blobFromImage(bgr_frame, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat out = outputs[0];
    
    int rows = out.size[0];
    int cols = out.size[1];
    if (out.dims == 3) {
        rows = out.size[1];
        cols = out.size[2];
    }

    cv::Mat predictions(rows, cols, CV_32F, out.ptr<float>());
    predictions = predictions.clone().t();

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> keypoints_list;

    float x_factor = (float)img_w / (float)INPUT_WIDTH;
    float y_factor = (float)img_h / (float)INPUT_HEIGHT;

    for (int i = 0; i < predictions.rows; ++i) {
        float* data = (float*)predictions.ptr<float>(i);
        float confidence = data[4]; 

        if (confidence > SCORE_THRESHOLD) {
            scores.push_back(confidence);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = (int)((cx - 0.5f * w) * x_factor);
            int top = (int)((cy - 0.5f * h) * y_factor);
            int width = (int)(w * x_factor);
            int height = (int)(h * y_factor);
            
            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<cv::Point2f> kpts;
            for (int k = 0; k < 17; ++k) {
                float kx = data[5 + k * 3];
                float ky = data[5 + k * 3 + 1];
                float k_conf = data[5 + k * 3 + 2];

                if (k_conf > 0.3f) { 
                    kpts.push_back(cv::Point2f(kx * x_factor, ky * y_factor));
                }
            }
            keypoints_list.push_back(kpts);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    out_mask = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    cv::Rect best_bbox = prev_bbox;

    if (!indices.empty()) {
        int idx = indices[0]; 
        best_bbox = boxes[idx];
        
        std::vector<cv::Point2f> kpts = keypoints_list[idx];
        if (kpts.size() >= 3) {
            std::vector<cv::Point> kpts_int;
            int min_x = img_w;
            int min_y = img_h;
            int max_x = 0;
            int max_y = 0;
            
            for (size_t p = 0; p < kpts.size(); ++p) {
                int px = (int)kpts[p].x;
                int py = (int)kpts[p].y;
                kpts_int.push_back(cv::Point(px, py));
                
                if (px < min_x) min_x = px;
                if (py < min_y) min_y = py;
                if (px > max_x) max_x = px;
                if (py > max_y) max_y = py;
            }
            
            // Expand bounding box to enclose all keypoints
            cv::Rect kpt_rect(min_x, min_y, max_x - min_x, max_y - min_y);
            int union_x1 = (best_bbox.x < kpt_rect.x) ? best_bbox.x : kpt_rect.x;
            int union_y1 = (best_bbox.y < kpt_rect.y) ? best_bbox.y : kpt_rect.y;
            int union_x2 = (best_bbox.x + best_bbox.width > kpt_rect.x + kpt_rect.width) ? (best_bbox.x + best_bbox.width) : (kpt_rect.x + kpt_rect.width);
            int union_y2 = (best_bbox.y + best_bbox.height > kpt_rect.y + kpt_rect.height) ? (best_bbox.y + best_bbox.height) : (kpt_rect.y + kpt_rect.height);
            best_bbox = cv::Rect(union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1);

            // Generate convex hull mask
            std::vector<cv::Point> hull;
            cv::convexHull(kpts_int, hull);
            cv::fillConvexPoly(out_mask, hull, cv::Scalar(255));
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
            cv::dilate(out_mask, out_mask, kernel);
        } else {
            cv::rectangle(out_mask, best_bbox, cv::Scalar(255), cv::FILLED);
        }

        // Apply dynamic padding
        int pad_x = (int)(best_bbox.width * 0.06);
        int pad_y = (int)(best_bbox.height * 0.02);
        
        best_bbox.x = best_bbox.x - pad_x;
        if (best_bbox.x < 0) {
            best_bbox.x = 0;
        }
        
        best_bbox.y = best_bbox.y - pad_y;
        if (best_bbox.y < 0) {
            best_bbox.y = 0;
        }
        
        int new_w = best_bbox.width + 2 * pad_x;
        int max_w = img_w - best_bbox.x;
        if (new_w < max_w) {
            best_bbox.width = new_w;
        } else {
            best_bbox.width = max_w;
        }

        int new_h = best_bbox.height + 2 * pad_y;
        int max_h = img_h - best_bbox.y;
        if (new_h < max_h) {
            best_bbox.height = new_h;
        } else {
            best_bbox.height = max_h;
        }

        int inter_x1 = (best_bbox.x > 0) ? best_bbox.x : 0;
        int inter_y1 = (best_bbox.y > 0) ? best_bbox.y : 0;
        int inter_x2 = (best_bbox.x + best_bbox.width < img_w) ? (best_bbox.x + best_bbox.width) : img_w;
        int inter_y2 = (best_bbox.y + best_bbox.height < img_h) ? (best_bbox.y + best_bbox.height) : img_h;
        
        int inter_w = inter_x2 - inter_x1;
        int inter_h = inter_y2 - inter_y1;
        if (inter_w < 0) inter_w = 0;
        if (inter_h < 0) inter_h = 0;
        best_bbox = cv::Rect(inter_x1, inter_y1, inter_w, inter_h);

        // Exponential moving average smoothing against jitter
        if (first_frame) {
            prev_bbox = best_bbox;
            first_frame = false;
        } else {
            float alpha = 0.80f; 
            int new_x = (int)std::round(best_bbox.x * alpha + prev_bbox.x * (1.0f - alpha));
            int new_y = (int)std::round(best_bbox.y * alpha + prev_bbox.y * (1.0f - alpha));
            int new_w = (int)std::round(best_bbox.width * alpha + prev_bbox.width * (1.0f - alpha));
            int new_h = (int)std::round(best_bbox.height * alpha + prev_bbox.height * (1.0f - alpha));

            best_bbox = cv::Rect(new_x, new_y, new_w, new_h);
            prev_bbox = best_bbox;
        }
    } else {
        // Fallback to previous bounding box if detection missed
        best_bbox = prev_bbox;
    }
    return best_bbox;
}