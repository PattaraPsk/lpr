"""
YOLO Fine-Tuning Engine.
Executes YOLO model fine-tuning with frozen layers, custom hyperparameters, and real-time metric streaming.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
import threading
import time
from PyQt6.QtCore import QObject, pyqtSignal

class TrainerEngine(QObject):
    """Worker object that manages YOLO model fine-tuning."""
    
    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, int)  # current, total
    metrics_signal = pyqtSignal(dict)    # {'epoch': e, 'loss': l, 'map50': m, etc.}
    finished_signal = pyqtSignal(str)   # best.pt path
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._is_running = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start_training(
        self,
        model_path: str,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        frozen_layers: Optional[List[int]] = None,
        lr0: float = 0.01,
        optimizer: str = "auto",
        device: str = "cpu",
        project_dir: str = "runs/detect",
        run_name: str = "lpr_finetuned"
    ):
        if self._is_running:
            return
            
        self._is_running = True
        self._stop_requested = False
        
        self._thread = threading.Thread(
            target=self._run_training,
            kwargs={
                'model_path': model_path,
                'data_yaml': data_yaml,
                'epochs': epochs,
                'batch_size': batch_size,
                'imgsz': imgsz,
                'frozen_layers': frozen_layers or [],
                'lr0': lr0,
                'optimizer': optimizer,
                'device': device,
                'project_dir': project_dir,
                'run_name': run_name
            },
            daemon=True
        )
        self._thread.start()

    def stop_training(self):
        self._stop_requested = True
        self.log_signal.emit("[TrainerEngine] Stop requested by user...")

    def _run_training(
        self,
        model_path: str,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        frozen_layers: List[int],
        lr0: float,
        optimizer: str,
        device: str,
        project_dir: str,
        run_name: str
    ):
        try:
            self.log_signal.emit(f"🚀 Initializing Fine-Tuning...")
            self.log_signal.emit(f"• Base Model: {model_path}")
            self.log_signal.emit(f"• Dataset YAML: {data_yaml}")
            self.log_signal.emit(f"• Epochs: {epochs} | Batch: {batch_size} | ImgSize: {imgsz}")
            self.log_signal.emit(f"• Frozen Layers: {frozen_layers if frozen_layers else 'None'}")
            self.log_signal.emit(f"• Optimizer: {optimizer} | LR: {lr0} | Device: {device}")
            
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Define Ultralytics callbacks for real-time UI streaming
            def on_train_epoch_end(trainer):
                if self._stop_requested:
                    trainer.stop = True
                    return
                current = trainer.epoch + 1
                total = trainer.epochs
                self.epoch_signal.emit(current, total)
                
                # Extract metrics
                metrics = {'epoch': current, 'total': total}
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        losses = trainer.loss_items.cpu().numpy()
                        if len(losses) >= 1:
                            metrics['box_loss'] = float(losses[0])
                        if len(losses) >= 2:
                            metrics['cls_loss'] = float(losses[1])
                        if len(losses) >= 3:
                            metrics['dfl_loss'] = float(losses[2])
                    except Exception:
                        pass
                if hasattr(trainer, 'metrics'):
                    m = trainer.metrics
                    if 'metrics/mAP50(B)' in m:
                        metrics['mAP50'] = float(m['metrics/mAP50(B)'])
                    if 'metrics/mAP50-95(B)' in m:
                        metrics['mAP50_95'] = float(m['metrics/mAP50-95(B)'])
                self.metrics_signal.emit(metrics)
                self.log_signal.emit(f"Epoch [{current}/{total}] Loss: {metrics.get('box_loss', 0):.4f} mAP50: {metrics.get('mAP50', 0):.3f}")

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            # Ultralytics train args
            train_args = {
                'data': data_yaml,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': imgsz,
                'lr0': lr0,
                'device': device,
                'project': project_dir,
                'name': run_name,
                'exist_ok': True,
                'verbose': True
            }
            if optimizer != "auto":
                train_args['optimizer'] = optimizer
            if frozen_layers:
                train_args['freeze'] = frozen_layers
                
            self.log_signal.emit("⚡ Starting training loop...")
            results = model.train(**train_args)
            
            save_dir = getattr(results, "save_dir", Path(project_dir) / run_name)
            best_pt = Path(save_dir) / "weights" / "best.pt"
            
            self.log_signal.emit(f"🎉 Training completed successfully!")
            self.log_signal.emit(f"Best weights saved at: {best_pt}")
            self.finished_signal.emit(str(best_pt))
            
        except Exception as e:
            err = f"Fine-tuning error: {str(e)}"
            self.log_signal.emit(f"❌ {err}")
            self.error_signal.emit(err)
        finally:
            self._is_running = False
