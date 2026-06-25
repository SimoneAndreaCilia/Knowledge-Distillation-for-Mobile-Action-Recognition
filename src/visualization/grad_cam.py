import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM3D:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None
        
        self._register_hooks()

    def _get_target_layer(self):
        try:
            parts = self.target_layer_name.replace('[', '.').replace(']', '').split('.')
            curr = self.model
            for part in parts:
                if not part: continue
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                elif hasattr(curr, '__getitem__'):
                    curr = curr[int(part)]
                else:
                    return dict(self.model.named_modules()).get(self.target_layer_name)
            return curr
        except Exception:
            return dict(self.model.named_modules()).get(self.target_layer_name)

    def _register_hooks(self):
        target_layer = self._get_target_layer()
        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
            
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def generate(self, input_tensor, target_class=None):
        """
        Computes the Grad-CAM heatmap for a 3D tensor.
        input_tensor: (1, 3, T, H, W)
        target_class: int (optional)
        
        Returns: (cam_resized, target_class)
        where cam_resized is (T, H, W)
        """
        # Ensure model is in eval mode
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        score = logits[0, target_class]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Ensure the target layer was used in the forward pass.")
            
        gradients = self.gradients[0].cpu().data.numpy()  # (C, T', H', W')
        activations = self.activations[0].cpu().data.numpy()  # (C, T', H', W')
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2, 3))  # (C,)
        
        # Weighted combination
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (T', H', W')
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        # Interpolate to original tensor size (T, H, W)
        _, _, orig_T, orig_H, orig_W = input_tensor.shape
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0) # (1, 1, T', H', W')
        cam_resized_tensor = F.interpolate(cam_tensor, size=(orig_T, orig_H, orig_W), mode='trilinear', align_corners=False)
        cam_resized = cam_resized_tensor.squeeze().numpy() # (T, H, W)
        
        return cam_resized, target_class

def generate_gradcam_video(
    model: torch.nn.Module, 
    video_tensor: torch.Tensor, 
    raw_frames: np.ndarray, 
    target_layer_name: str, 
    output_path: str, 
    target_class: int = None,
    fps: int = 15
):
    """
    Generates a Grad-CAM video.
    
    video_tensor: (1, 3, T, H, W)
    raw_frames: (T, H, W, 3) RGB uint8
    """
    cam_extractor = GradCAM3D(model, target_layer_name)
    try:
        cam, pred_class = cam_extractor.generate(video_tensor, target_class)
    finally:
        cam_extractor.remove_hooks()
        model.zero_grad(set_to_none=True)
        
    T, H, W, _ = raw_frames.shape
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (W, H))
    
    for t in range(T):
        frame = raw_frames[t] # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        heatmap = cam[t]
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(frame_bgr, 0.5, heatmap_color, 0.5, 0)
        out.write(overlay)
        
    out.release()
    
    # Transcode to H.264 for web browser compatibility
    try:
        from src.services.video_converter import _get_ffmpeg_exe
        import subprocess
        ffmpeg_exe = _get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", temp_output_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_output_path)
    except Exception as e:
        import shutil
        import logging
        logging.getLogger(__name__).warning("ffmpeg conversion failed, keeping original mp4v format: %s", e)
        shutil.move(temp_output_path, output_path)

    return pred_class
