# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Optional
import gradio as gr

from src.gui.callbacks.gradcam_callbacks import GradCamCallbackHandler
from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.ui.components import VideoInputSection
from src.repositories.model_registry import ModelRegistry
from src.i18n.translator import Translator

class GradCamTab:
    """Builds Tab 3: Visualizza Attenzione (Grad-CAM)."""

    def __init__(
        self,
        demo: gr.Blocks,
        gradcam_handler: GradCamCallbackHandler,
        dataset_handler: DatasetCallbackHandler,
        registry: ModelRegistry,
        classes: List[str],
        default_class: Optional[str],
    ) -> None:
        self.demo = demo
        self.gradcam_handler = gradcam_handler
        self.dataset_handler = dataset_handler
        self.registry = registry
        self.classes = classes
        self.default_class = default_class
        
        self.models_main = registry.keys(show_advanced=False)
        self.models_all = registry.keys(show_advanced=True)

        # Components
        self.tab: Optional[gr.Tab] = None
        self.video_section: Optional[VideoInputSection] = None
        self.model1_dropdown: Optional[gr.Dropdown] = None
        self.layer1_dropdown: Optional[gr.Dropdown] = None
        self.model2_dropdown: Optional[gr.Dropdown] = None
        self.layer2_dropdown: Optional[gr.Dropdown] = None
        
        self.class_mode_dropdown: Optional[gr.Radio] = None
        self.manual_class_dropdown: Optional[gr.Dropdown] = None
        
        self.generate_btn: Optional[gr.Button] = None
        self.video1_out: Optional[gr.Video] = None
        self.video2_out: Optional[gr.Video] = None

    def build(self, lang_state: gr.State) -> None:
        """Constructs the layout and wires events."""
        with gr.Tab(id="gradcam", label="Visualizza Attenzione") as self.tab:
            
            with gr.Column():
                # ---- Top: Video Input ----
                self.video_section = VideoInputSection(self.classes, self.default_class)
                self.video_section.build()
                
                # ---- Middle: Settings ----
                gr.Markdown("### Configurazione Modelli e Target Class")
                with gr.Row():
                    with gr.Column(scale=1):
                        self.model1_dropdown = gr.Dropdown(
                            label="Modello 1 (Sinistra)",
                            choices=self.models_all,
                            value=self.models_main[0] if self.models_main else None,
                            interactive=True
                        )
                        self.layer1_dropdown = gr.Dropdown(
                            label="Target Layer (Modello 1)",
                            choices=["layer4", "layer3", "stages[-1]", "stages[4]"],
                            value="layer4",
                            interactive=True,
                            allow_custom_value=True
                        )
                    
                    with gr.Column(scale=1):
                        self.model2_dropdown = gr.Dropdown(
                            label="Modello 2 (Destra)",
                            choices=self.models_all,
                            value=self.models_main[1] if len(self.models_main) > 1 else None,
                            interactive=True
                        )
                        self.layer2_dropdown = gr.Dropdown(
                            label="Target Layer (Modello 2)",
                            choices=["layer4", "layer3", "stages[-1]", "stages[4]"],
                            value="stages[-1]",
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                with gr.Row():
                    self.class_mode_dropdown = gr.Radio(
                        label="Classe Target (per Grad-CAM)",
                        choices=["ground_truth", "predicted", "manual"],
                        value="ground_truth",
                        interactive=True
                    )
                    self.manual_class_dropdown = gr.Dropdown(
                        label="Classe Manuale",
                        choices=self.classes,
                        value=self.default_class,
                        interactive=True,
                        visible=False
                    )

                self.generate_btn = gr.Button("Genera Grad-CAM", variant="primary")
                
                # ---- Bottom: Output ----
                with gr.Row():
                    with gr.Column(scale=1):
                        self.video1_out = gr.Video(label="Modello 1: Attenzione")
                    with gr.Column(scale=1):
                        self.video2_out = gr.Video(label="Modello 2: Attenzione")

            # ---- Event Wiring ----
            self.video_section.video_source.change(
                fn=self.dataset_handler.toggle_video_source,
                inputs=[self.video_section.video_source],
                outputs=[self.video_section.upload_section, self.video_section.dataset_section],
            )

            self.video_section.dataset_class.change(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )
            
            self.video_section.dataset_split.change(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

            self.video_section.dataset_video.change(
                fn=self.dataset_handler.get_preview_path,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_video],
                outputs=[self.video_section.video_preview],
            )

            def toggle_manual_class(mode):
                return gr.update(visible=(mode == "manual"))

            self.class_mode_dropdown.change(
                fn=toggle_manual_class,
                inputs=[self.class_mode_dropdown],
                outputs=[self.manual_class_dropdown]
            )

            self.generate_btn.click(
                fn=self.gradcam_handler.generate,
                inputs=[
                    self.video_section.uploaded_video,
                    self.video_section.dataset_class,
                    self.video_section.dataset_video,
                    self.video_section.video_source,
                    self.model1_dropdown,
                    self.layer1_dropdown,
                    self.model2_dropdown,
                    self.layer2_dropdown,
                    self.class_mode_dropdown,
                    self.manual_class_dropdown,
                    lang_state,
                ],
                outputs=[self.video1_out, self.video2_out],
            )

            self.demo.load(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        updates = self.video_section.get_language_updates(translator)
        
        # We can add translations later, for now we return just video_section updates
        def update_tab(lang):
            return gr.update(label="Visualizza Attenzione") # TODO: Add TranslationKey
            
        updates.update({
            self.tab: update_tab,
        })
        return updates
