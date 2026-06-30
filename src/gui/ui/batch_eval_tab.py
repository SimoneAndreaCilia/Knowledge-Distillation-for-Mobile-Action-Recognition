# -*- coding: utf-8 -*-
"""Folder-level batch evaluation tab layout and event wiring."""

from typing import Callable, Dict, List, Optional

import gradio as gr

from src.gui.callbacks.batch_eval_callbacks import BatchEvalCallbackHandler
from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator
from src.repositories.model_registry import ModelRegistry


class BatchEvalTab:
    """Builds Tab 4: Folder-Level Batch Evaluation."""

    def __init__(self, demo: gr.Blocks, batch_handler: BatchEvalCallbackHandler, registry: ModelRegistry, classes: List[str], default_class: Optional[str]) -> None:
        self.demo = demo
        self.batch_handler = batch_handler
        self.registry = registry
        self.classes = classes
        self.default_class = default_class
        self.tab = None
        self.desc_md = None
        self.controls_md = None
        self.class_dropdown = None
        self.split_dropdown = None
        self.show_advanced = None
        self.model_group = None
        self.run_btn = None
        self.summary_md = None
        self.accuracy_plot = None
        self.detail_model = None
        self.confidence_plot = None
        self.confusion_plot = None
        self.results_table = None
        self.results_state = None

    def build(self, lang_state: gr.State) -> None:
        """Construct the layout and wire events."""
        main_models = self.registry.keys(show_advanced=False)
        with gr.Tab(id="batch_eval") as self.tab:
            self.desc_md = gr.Markdown()
            self.results_state = gr.State([])
            with gr.Column(elem_classes="comparison-container"):
                self.controls_md = gr.Markdown()
                with gr.Row():
                    self.class_dropdown = gr.Dropdown(choices=self.classes, value=self.default_class, interactive=True, allow_custom_value=False, scale=2)
                    self.split_dropdown = gr.Dropdown(choices=[("All Videos", "all"), ("Test Split 1", "split1"), ("Test Split 2", "split2"), ("Test Split 3", "split3")], value="split1", interactive=True, allow_custom_value=False, scale=1)
                with gr.Row():
                    self.model_group = gr.CheckboxGroup(choices=main_models, value=main_models, interactive=True, scale=4)
                    self.show_advanced = gr.Checkbox(value=False, interactive=True, scale=1)
                self.run_btn = gr.Button(variant="primary", elem_classes="primary-btn")
                self.summary_md = gr.Markdown(elem_classes="result-card")
                self.accuracy_plot = gr.Plot()
                self.detail_model = gr.Dropdown(choices=[], value=None, interactive=True, allow_custom_value=False)
                with gr.Row():
                    self.confidence_plot = gr.Plot(scale=1)
                    self.confusion_plot = gr.Plot(scale=1)
                self.results_table = gr.Dataframe(interactive=False, wrap=True, elem_classes="result-card")

            self.show_advanced.change(fn=self.batch_handler.update_model_choices, inputs=[self.show_advanced], outputs=[self.model_group])
            self.run_btn.click(
                fn=self.batch_handler.run_batch,
                inputs=[self.class_dropdown, self.split_dropdown, self.model_group, lang_state],
                outputs=[self.results_table, self.accuracy_plot, self.confidence_plot, self.confusion_plot, self.summary_md, self.detail_model, self.results_state],
            )
            self.detail_model.change(fn=self.batch_handler.render_selected_model, inputs=[self.results_state, self.detail_model], outputs=[self.confidence_plot, self.confusion_plot])

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Return updater functions for live language switching."""
        def update_tab(lang):
            return gr.update(label=translator.t(TranslationKey.TAB_BATCH_EVAL, lang=lang))
        def update_desc(lang):
            return gr.update(value=translator.t(TranslationKey.BATCH_DESC, lang=lang))
        def update_controls(lang):
            return gr.update(value=translator.t(TranslationKey.BATCH_CONTROLS_TITLE, lang=lang))
        def update_class(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_CLASS, lang=lang))
        def update_split(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_SPLIT, lang=lang), choices=[(translator.t(TranslationKey.DATASET_SPLIT_ALL, lang=lang), "all"), (translator.t(TranslationKey.DATASET_SPLIT_1, lang=lang), "split1"), (translator.t(TranslationKey.DATASET_SPLIT_2, lang=lang), "split2"), (translator.t(TranslationKey.DATASET_SPLIT_3, lang=lang), "split3")])
        def update_models(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_MODELS_LABEL, lang=lang))
        def update_show_advanced(lang):
            return gr.update(label=translator.t(TranslationKey.MODEL_SHOW_ADVANCED, lang=lang))
        def update_run(lang):
            return gr.update(value=translator.t(TranslationKey.BATCH_RUN_BTN, lang=lang))
        def update_summary(lang):
            return gr.update(value=translator.t(TranslationKey.BATCH_STATUS_WAITING, lang=lang))
        def update_accuracy(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_ACCURACY_LABEL, lang=lang))
        def update_detail_model(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_DETAIL_MODEL_LABEL, lang=lang))
        def update_confidence(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_CONFIDENCE_LABEL, lang=lang))
        def update_confusion(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_CONFUSION_LABEL, lang=lang))
        def update_table(lang):
            return gr.update(label=translator.t(TranslationKey.BATCH_TABLE_LABEL, lang=lang))
        return {
            self.tab: update_tab,
            self.desc_md: update_desc,
            self.controls_md: update_controls,
            self.class_dropdown: update_class,
            self.split_dropdown: update_split,
            self.model_group: update_models,
            self.show_advanced: update_show_advanced,
            self.run_btn: update_run,
            self.summary_md: update_summary,
            self.accuracy_plot: update_accuracy,
            self.detail_model: update_detail_model,
            self.confidence_plot: update_confidence,
            self.confusion_plot: update_confusion,
            self.results_table: update_table,
        }
