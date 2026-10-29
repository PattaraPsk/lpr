"""
Background worker threads for non-blocking edge_reader AI Reader inference.
Provides single-image asynchronous prediction and multi-image batch prediction.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal
from .plate_reader import PlateReader


class AISingleReaderWorker(QThread):
    """Worker thread to run edge_reader prediction on a single image without blocking the GUI."""
    prediction_ready = pyqtSignal(str, dict) # (image_path, result_dict)
    
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = str(image_path)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            return
            
        reader = PlateReader.get_instance()
        result = reader.predict(self.image_path)
        
        if not self._is_cancelled:
            self.prediction_ready.emit(self.image_path, result)


class AIBatchReaderWorker(QThread):
    """Worker thread to run batch edge_reader prediction across multiple images."""
    progress = pyqtSignal(int, int, str)    # (current, total, current_filename)
    item_predicted = pyqtSignal(int, dict)  # (item_index, result_dict)
    batch_finished = pyqtSignal(int)        # (total_completed)

    def __init__(self, items: List[Tuple[int, str]], parent=None):
        super().__init__(parent)
        self.items = items # List of (index, image_path)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        reader = PlateReader.get_instance()
        total = len(self.items)
        completed = 0
        
        for idx, img_path in self.items:
            if self._is_cancelled:
                break
                
            res = reader.predict(img_path)
            completed += 1
            
            if not self._is_cancelled:
                self.item_predicted.emit(idx, res)
                self.progress.emit(completed, total, Path(img_path).name)
                
        self.batch_finished.emit(completed)
