# -*- coding: utf-8 -*-
"""Translation keys for the i18n system."""

from enum import Enum


class TranslationKey(str, Enum):
    """Strongly typed keys for translations."""
    
    # Header
    HEADER_TITLE = "header.title"
    HEADER_SUBTITLE = "header.subtitle"
    HEADER_LANGUAGE = "header.language"
    HEADER_BRANDING = "header.branding"
    
    # Tabs
    TAB_SINGLE_INFERENCE = "tabs.single_inference"
    TAB_COMPARISON = "tabs.comparison"
    TAB_GRADCAM = "tabs.gradcam"
    TAB_BATCH_EVAL = "tabs.batch_eval"
    
    # Video Input
    VIDEO_INPUT_TITLE = "video_input.section_title"
    VIDEO_INPUT_SOURCE = "video_input.source_label"
    VIDEO_INPUT_UPLOAD_CHOICE = "video_input.source_upload"
    VIDEO_INPUT_DATASET_CHOICE = "video_input.source_dataset"
    VIDEO_INPUT_UPLOAD = "video_input.upload_label"
    VIDEO_INPUT_CLASS = "video_input.class_label"
    VIDEO_INPUT_SPLIT = "video_input.split_label"
    VIDEO_INPUT_VIDEO = "video_input.video_label"
    VIDEO_INPUT_PREVIEW = "video_input.preview_label"
    
    # Dataset Splits
    DATASET_SPLIT_ALL = "dataset.split_all"
    DATASET_SPLIT_1 = "dataset.split_1"
    DATASET_SPLIT_2 = "dataset.split_2"
    DATASET_SPLIT_3 = "dataset.split_3"
    
    # Model config
    MODEL_TITLE = "model.section_title"
    MODEL_SHOW_ADVANCED = "model.show_advanced"
    MODEL_SELECT = "model.select_model"
    MODEL_CLASSIFY_BTN = "model.classify_btn"
    MODEL_COMPARE_BTN = "model.compare_btn"
    
    # Results
    RESULTS_TITLE = "results.section_title"
    RESULTS_STATUS_WAITING = "results.status_waiting"
    RESULTS_TOP5 = "results.top5_label"
    RESULTS_MODEL_INFO = "results.model_info_label"
    
    # Comparison
    COMP_DESC = "comparison.description"
    COMP_PLOT = "comparison.plot_label"
    COMP_STATUS_WAITING = "comparison.status_waiting"
    
    # Footer
    FOOTER_TEXT = "footer.text"
    
    # Errors
    ERR_UPLOAD_VIDEO = "errors.upload_video"
    ERR_NO_VIDEOS = "errors.no_videos_in_class"
    ERR_NO_PREVIEW = "errors.no_preview"
    ERR_INFERENCE = "errors.inference_failed"
    
    # Info
    INFO_INFERENCE_SUCCESS = "info.inference_success"
    INFO_COMPARISON_SUCCESS = "info.comparison_success"
    
    # Info labels
    INFO_PARAMS = "info.params"
    INFO_SIZE = "info.size"
    INFO_ACCURACY = "info.accuracy_test"
    INFO_TIME = "info.inference_time"
    
    # Status
    STATUS_LOADING = "status.loading"
    STATUS_NO_PRED = "status.no_prediction"
    STATUS_PRED = "status.prediction"
    STATUS_CORRECT = "status.correct"
    STATUS_INCORRECT = "status.incorrect"
    
    # Comparison Summary
    COMP_SUMMARY_TITLE = "comparison.summary_title"
    COMP_GROUND_TRUTH = "comparison.ground_truth"
    COMP_UNKNOWN_ERROR = "comparison.unknown_error"
    COMP_NO_PREDICTION = "comparison.no_prediction"
    COMP_BEST_PREDICTION = "comparison.best_prediction"
    COMP_ALL_INCORRECT = "comparison.all_incorrect"
    
    # Models Descriptions
    DESC_TEACHER = "models.teacher_desc"
    DESC_STUDENT = "models.student_desc"
    DESC_STUDENT_DISTILLED_T10 = "models.student_distilled_t10_desc"
    DESC_STUDENT_AT = "models.student_at_desc"
    
    # Advanced Models
    DESC_STUDENT_KD_T1 = "models.student_kd_t1"
    DESC_STUDENT_KD_T5 = "models.student_kd_t5"
    DESC_STUDENT_KD_T20 = "models.student_kd_t20"
    DESC_STUDENT_AT_SEED42 = "models.student_at_seed42"
    DESC_STUDENT_AT_BETA10 = "models.student_at_beta10"
    DESC_STUDENT_AT_BETA100 = "models.student_at_beta100"
    DESC_STUDENT_KD_T10_W1_5 = "models.student_kd_t10_w1_5"


    # Batch Evaluation Tab
    BATCH_DESC = "batch.description"
    BATCH_CONTROLS_TITLE = "batch.controls_title"
    BATCH_MODELS_LABEL = "batch.models_label"
    BATCH_RUN_BTN = "batch.run_btn"
    BATCH_STATUS_WAITING = "batch.status_waiting"
    BATCH_ACCURACY_LABEL = "batch.accuracy_label"
    BATCH_DETAIL_MODEL_LABEL = "batch.detail_model_label"
    BATCH_CONFIDENCE_LABEL = "batch.confidence_label"
    BATCH_CONFUSION_LABEL = "batch.confusion_label"
    BATCH_TABLE_LABEL = "batch.table_label"
    BATCH_COL_MODEL = "batch.columns.model"
    BATCH_COL_VIDEO = "batch.columns.video"
    BATCH_COL_PREDICTED = "batch.columns.predicted"
    BATCH_COL_CONFIDENCE = "batch.columns.confidence"
    BATCH_COL_TRUE_CONFIDENCE = "batch.columns.true_confidence"
    BATCH_COL_TOP5 = "batch.columns.top5"
    BATCH_COL_CORRECT = "batch.columns.correct"
    BATCH_SUMMARY_TITLE = "batch.summary_title"
    BATCH_SUMMARY_MODEL = "batch.summary_model"
    BATCH_PROGRESS_LABEL = "batch.progress_label"
    BATCH_ERR_NO_CLASS = "batch.err_no_class"
    BATCH_ERR_NO_MODELS = "batch.err_no_models"

    # Grad-CAM Tab
    GRADCAM_SETTINGS_TITLE = "gradcam.settings_title"
    GRADCAM_MODEL1_LABEL = "gradcam.model1_label"
    GRADCAM_LAYER1_LABEL = "gradcam.layer1_label"
    GRADCAM_MODEL2_LABEL = "gradcam.model2_label"
    GRADCAM_LAYER2_LABEL = "gradcam.layer2_label"
    GRADCAM_CLASS_MODE_LABEL = "gradcam.class_mode_label"
    GRADCAM_CLASS_MODE_GROUND_TRUTH = "gradcam.class_mode_ground_truth"
    GRADCAM_CLASS_MODE_PREDICTED = "gradcam.class_mode_predicted"
    GRADCAM_CLASS_MODE_MANUAL = "gradcam.class_mode_manual"
    GRADCAM_MANUAL_CLASS_LABEL = "gradcam.manual_class_label"
    GRADCAM_GENERATE_BTN = "gradcam.generate_btn"
    GRADCAM_OUTPUT1_LABEL = "gradcam.output1_label"
    GRADCAM_OUTPUT2_LABEL = "gradcam.output2_label"

    # Grad-CAM Info / Tooltips
    GRADCAM_CLASS_MODE_INFO = "gradcam.class_mode_info"
    GRADCAM_LAYER_INFO = "gradcam.layer_info"
    GRADCAM_MANUAL_CLASS_INFO = "gradcam.manual_class_info"

    # Grad-CAM Prediction display
    GRADCAM_PREDICTION_LABEL = "gradcam.prediction_label"

    # Grad-CAM Errors
    GRADCAM_ERR_SELECT_VIDEO = "gradcam.err_select_video"
    GRADCAM_ERR_UPLOAD_VIDEO = "gradcam.err_upload_video"
    GRADCAM_ERR_NO_GROUND_TRUTH = "gradcam.err_no_ground_truth"
    GRADCAM_ERR_SELECT_MANUAL_CLASS = "gradcam.err_select_manual_class"
    GRADCAM_ERR_MODEL1 = "gradcam.err_model1"
    GRADCAM_ERR_MODEL2 = "gradcam.err_model2"
