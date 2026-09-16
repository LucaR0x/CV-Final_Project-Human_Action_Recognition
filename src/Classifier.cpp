/**
AUTHOR: CORTESE ALESSANDRO
*/

#include "Classifier.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

ActionClassifier::ActionClassifier() {
    svm_model = cv::ml::SVM::create();
    svm_model->setType(cv::ml::SVM::C_SVC);
    svm_model->setKernel(cv::ml::SVM::RBF);
    svm_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5000, 1e-6));
}

void ActionClassifier::prepareMatrices(const std::vector<FeatureSample>& samples, cv::Mat& out_features, cv::Mat& out_labels) {
    int num_samples = (int)samples.size();
    int num_features = (int)samples[0].descriptors.size();

    out_features = cv::Mat(num_samples, num_features, CV_32F);
    out_labels = cv::Mat(num_samples, 1, CV_32S);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            out_features.at<float>(i, j) = samples[i].descriptors[j];
        }
        out_labels.at<int>(i, 0) = samples[i].label;
    }
}

void ActionClassifier::computeScalingParams(const cv::Mat& data) {
    int cols = data.cols;
    int rows = data.rows;
    trained_means.resize(cols);
    trained_stds.resize(cols);
    for (int j = 0; j < cols; ++j) {
        trained_means[j] = 0.0f;
        trained_stds[j] = 0.0f;
    }

    for (int j = 0; j < cols; ++j) {
        float sum_val = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum_val += data.at<float>(i, j);
        }
        trained_means[j] = sum_val / (float)rows;

        float sq_diff = 0.0f;
        for (int i = 0; i < rows; ++i) {
            float d = data.at<float>(i, j) - trained_means[j];
            sq_diff += d * d;
        }
        trained_stds[j] = std::sqrt(sq_diff / (float)rows) + 1e-6f;
    }
}

void ActionClassifier::applyScaling(cv::Mat& data) const {
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            data.at<float>(i, j) = (data.at<float>(i, j) - trained_means[j]) / trained_stds[j];
        }
    }
}

float ActionClassifier::evaluate(const std::vector<FeatureSample>& dataset, float) {
    if (dataset.empty()) return 0.0f;

    std::vector<FeatureSample> class_buckets[6];
    for (size_t i = 0; i < dataset.size(); ++i) {
        int lbl = dataset[i].label;
        if (lbl >= 1 && lbl <= 6) {
            class_buckets[lbl - 1].push_back(dataset[i]);
        }
    }

    std::srand(42);
    for (int c = 0; c < 6; ++c) {
        int n = (int)class_buckets[c].size();
        for (int i = n - 1; i > 0; --i) {
            int j = std::rand() % (i + 1);
            FeatureSample temp = class_buckets[c][i];
            class_buckets[c][i] = class_buckets[c][j];
            class_buckets[c][j] = temp;
        }
    }

    const int K_FOLDS = 6;
    int confusion_matrix[6][6];
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            confusion_matrix[r][c] = 0;
        }
    }
    int total_eval = 0;
    int total_correct = 0;

    const std::string class_names[6] = {"Boxing", "Clapping", "Waving", "Jogging", "Running", "Walking"};

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        std::vector<FeatureSample> train_fold;
        std::vector<FeatureSample> test_fold;

        for (int c = 0; c < 6; ++c) {
            int fold_size = (int)class_buckets[c].size() / K_FOLDS;
            int start_idx = fold * fold_size;
            int end_idx = start_idx + fold_size;

            for (int i = 0; i < (int)class_buckets[c].size(); ++i) {
                if (i >= start_idx && i < end_idx) {
                    test_fold.push_back(class_buckets[c][i]);
                } else {
                    train_fold.push_back(class_buckets[c][i]);
                }
            }
        }

        cv::Mat train_X, train_y, test_X, test_y;
        prepareMatrices(train_fold, train_X, train_y);
        prepareMatrices(test_fold, test_X, test_y);

        computeScalingParams(train_X);
        applyScaling(train_X);
        applyScaling(test_X);

        cv::Ptr<cv::ml::SVM> fold_svm = cv::ml::SVM::create();
        fold_svm->setType(cv::ml::SVM::C_SVC);
        fold_svm->setKernel(cv::ml::SVM::RBF);

        cv::ml::ParamGrid c_grid(0.1, 500, 2);
        cv::ml::ParamGrid gamma_grid(0.0001, 2.0, 2);

        fold_svm->trainAuto(cv::ml::TrainData::create(train_X, cv::ml::ROW_SAMPLE, train_y),
                             10, c_grid, gamma_grid,
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE),
                             true);

        for (int i = 0; i < test_X.rows; ++i) {
            int pred_label = (int)fold_svm->predict(test_X.row(i));
            int gt_label = test_y.at<int>(i, 0);

            if (gt_label >= 1 && gt_label <= 6 && pred_label >= 1 && pred_label <= 6) {
                confusion_matrix[gt_label - 1][pred_label - 1]++;
                if (pred_label == gt_label) {
                    total_correct++;
                } else {
                    std::cout << "   Fold " << fold << " misclass: " 
                              << test_fold[i].sequence_name << " (expected " 
                              << class_names[gt_label - 1] << ", got " 
                              << class_names[pred_label - 1] << ")\n";
                }
                total_eval++;
            }
        }
    }

    float accuracy = ((float)total_correct / (float)total_eval) * 100.0f;
    std::cout << "\n--- 6-Fold Cross-Validation ---\n";
    std::cout << "Evaluated sequences: " << total_eval << " / " << dataset.size() << "\n";
    std::cout << "Overall CV Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n\n";

    std::cout << "Confusion Matrix:\n";
    std::cout << "      [1]  [2]  [3]  [4]  [5]  [6]\n";
    for (int r = 0; r < 6; ++r) {
        std::cout << " [" << (r + 1) << "] ";
        for (int c = 0; c < 6; ++c) {
            std::cout << std::setw(4) << confusion_matrix[r][c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-class metrics:\n";
    for (int c = 0; c < 6; ++c) {
        int tp = confusion_matrix[c][c];
        int fn = 0;
        int fp = 0;

        for (int i = 0; i < 6; ++i) {
            if (i != c) {
                fn += confusion_matrix[c][i];
                fp += confusion_matrix[i][c];
            }
        }

        float prec = (tp + fp > 0) ? ((float)tp / (float)(tp + fp)) : 0.0f;
        float rec = (tp + fn > 0) ? ((float)tp / (float)(tp + fn)) : 0.0f;
        float f1 = (prec + rec > 0.0f) ? (2.0f * (prec * rec) / (prec + rec)) : 0.0f;

        std::cout << "  - " << std::left << std::setw(10) << class_names[c]
                  << " | Prec: " << std::fixed << std::setprecision(2) << prec
                  << " | Rec: " << rec
                  << " | F1: " << f1 << "\n";
    }
    std::cout << std::string(45, '-') << "\n\n";
    return accuracy;
}

void ActionClassifier::trainAndSave(const std::vector<FeatureSample>& dataset, const std::string& model_output_path) {
    cv::Mat all_X, all_y;
    prepareMatrices(dataset, all_X, all_y);

    computeScalingParams(all_X);
    applyScaling(all_X);

    cv::ml::ParamGrid c_grid(0.1, 500, 2);
    cv::ml::ParamGrid gamma_grid(0.0001, 2.0, 2);

    svm_model->trainAuto(cv::ml::TrainData::create(all_X, cv::ml::ROW_SAMPLE, all_y),
                         10, c_grid, gamma_grid,
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE),
                         true);

    svm_model->save(model_output_path);

    std::string scale_path = model_output_path + ".scale.yaml";
    cv::FileStorage fs(scale_path, cv::FileStorage::WRITE);
    fs << "means" << trained_means;
    fs << "stds" << trained_stds;
    fs.release();

    std::cout << "Saved SVM model and scaler: " << model_output_path << "\n";
}

int ActionClassifier::predict(const std::vector<float>& raw_feature_vector) const {
    if (trained_means.empty() || trained_stds.empty()) {
        std::cerr << "Error: Scaler parameters not initialized\n";
        return -1;
    }

    cv::Mat sample_mat(1, (int)raw_feature_vector.size(), CV_32F);
    for (size_t j = 0; j < raw_feature_vector.size(); ++j) {
        sample_mat.at<float>(0, (int)j) = (raw_feature_vector[j] - trained_means[j]) / trained_stds[j];
    }

    return (int)svm_model->predict(sample_mat);
}

bool ActionClassifier::loadModel(const std::string& model_input_path) {
    svm_model = cv::ml::SVM::load(model_input_path);

    std::string scale_path = model_input_path + ".scale.yaml";
    cv::FileStorage fs(scale_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Warning: Scaling file missing: " << scale_path << "\n";
        return !svm_model.empty();
    }

    fs["means"] >> trained_means;
    fs["stds"] >> trained_stds;
    fs.release();

    return !svm_model.empty();
}