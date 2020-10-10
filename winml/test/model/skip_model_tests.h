#pragma once
#include "common.h"

// TODO: Need to file bugs for failing tests and add to reason

// {"model test name", "reason for why it is happening and bug filed for it."}
std::unordered_map<std::string, std::string> disabledTests(
    {
      // Tier 3 models
     {"mxnet_arcface_opset8", "Model not working on CPU and GPU."},
     {"XGBoost_XGClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"XGBoost_XGClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"XGBoost_XGClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"XGBoost_XGClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_SVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_SVC_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_SVC_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_SVC_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_RandomForestClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_RandomForestClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_Nu_SVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_Nu_SVC_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_Nu_SVC_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_Nu_SVC_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_Normalizer_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_LogisticRegression_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_LogisticRegression_OpenML_31_credit_opset7", "Model not working on CPU and GPU."},
     {"scikit_LogisticRegression_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_LogisticRegression_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_LabelEncoder_OpenML_3_chess_opset7", "Model not working on CPU and GPU."},
     {"scikit_LabelEncoder_BikeSharing_opset7", "Model not working on CPU and GPU."},
     {"scikit_Imputer_LogisticRegression_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", "Model not working on CPU and GPU."},
     {"scikit_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_GradientBoostingClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_GradientBoostingClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_sklearn_load_Iris_missing_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_sklearn_load_digits_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_sklearn_load_diabetes_missing_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_OpenML_31_credit_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", "Model not working on CPU and GPU."},
     {"scikit_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", "Model not working on CPU and GPU."},
     {"scikit_DecisionTreeClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"scikit_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"scikit_DecisionTreeClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1492_plants_opset7", "Model not working on CPU and GPU."},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"libsvm_Nu_SVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"libsvm_Nu_SVC_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"libsvm_Nu_SVC_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"libsvm_Nu_SVC_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_VGG16_ImageNet_opset7", "Model not working on CPU and GPU."},
     {"coreml_SVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_SVC_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_SVC_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_SVC_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_SqueezeNet_ImageNet_opset7", "Model not working on CPU and GPU."},
     {"coreml_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_Resnet50_ImageNet_opset7", "Model not working on CPU and GPU."},
     {"coreml_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_RandomForestClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_RandomForestClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_RandomForestClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_Normalizer_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_LogisticRegression_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_LogisticRegression_OpenML_31_credit_opset7", "Model not working on CPU and GPU."},
     {"coreml_LogisticRegression_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_LogisticRegression_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_LinearSVC_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_LinearSVC_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_LinearSVC_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_Inceptionv3_ImageNet_opset7", "Model not working on CPU and GPU."},
     {"coreml_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", "Model not working on CPU and GPU."},
     {"coreml_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_Criteo_opset7", "Model not working on CPU and GPU."},
     {"coreml_GradientBoostingClassifier_BingClick_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_sklearn_load_Iris_missing_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_sklearn_load_digits_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_sklearn_load_diabetes_missing_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_OpenML_31_credit_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", "Model not working on CPU and GPU."},
     {"coreml_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", "Model not working on CPU and GPU."},
     {"coreml_DecisionTreeClassifier_sklearn_load_wine_opset7", "Model not working on CPU and GPU."},
     {"coreml_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", "Model not working on CPU and GPU."},
     {"coreml_DecisionTreeClassifier_OpenML_312_scene_opset7", "Model not working on CPU and GPU."},
     {"coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", "Model not working on CPU and GPU."},
     {"coreml_AgeNet_ImageNet_opset7", "Model not working on CPU and GPU."}});

std::unordered_map<std::string, std::string> disabledGpuTests(
    {{"LSTM_Seq_lens_unpacked_opset9", "Model not working on GPU."},
     {"fp16_inception_v1_opset8", "Model not working on GPU."},
     {"fp16_inception_v1_opset7", "Model not working on GPU."},
     {"mlperf_ssd_mobilenet_300_opset10", "Model not working on GPU."},
     {"mask_rcnn_opset10", "Model not working on GPU."},
     {"faster_rcnn_opset10", "Model not working on GPU."},
     {"BERT_Squad_opset10", "Model not working on GPU."}});
