# -*- coding: utf-8 -*-
"""Translation keys for the i18n system."""

from enum import Enum


class TranslationKey(str, Enum):
    """Strongly typed keys for translations."""
    
    # Header
    HEADER_TITLE = "header.title"
    HEADER_SUBTITLE = "header.subtitle"
    HEADER_LANGUAGE = "header.language"
    
    # Tabs
    TAB_SINGLE_INFERENCE = "tabs.single_inference"
    TAB_COMPARISON = "tabs.comparison"
    
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
