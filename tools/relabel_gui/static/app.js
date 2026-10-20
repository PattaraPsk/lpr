/**
 * Thai LPR Dataset Relabeling Studio - Unified Frontend Controller
 * Provides interactive YOLO Bounding Box Canvas & OCR Transcription Studio.
 */

// Application State
const state = {
  currentMode: 'detection', // 'detection' or 'ocr'
  currentDir: '',
  items: [],
  filteredItems: [],
  currentIndex: -1,
  filterStatus: 'all', // 'all', 'unlabeled', 'labeled'
  searchQuery: '',
  
  // Detection Mode State
  classes: ['license_plate'],
  selectedClassId: 0,
  boxes: [], // [{class_id, x_center, y_center, width, height}]
  selectedBoxIndex: -1,
  canvasImage: null,
  canvasScale: 1.0,
  canvasPanX: 0,
  canvasPanY: 0,
  isDrawing: false,
  isDraggingBox: false,
  isResizing: false,
  resizeHandle: null,
  isPanning: false,
  dragStartX: 0,
  dragStartY: 0,
  currentMouseX: 0,
  currentMouseY: 0,
  
  // OCR Mode State
  ocrFilters: {
    brightness: 100,
    contrast: 100,
    grayscale: false,
    invert: false
  },
  provinces: [],
  topProvinces: [],
  consonants: []
};

// Class Colors for Detection Canvas
const CLASS_COLORS = [
  '#10b981', // green
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ec4899', // pink
  '#8b5cf6', // purple
  '#06b6d4', // cyan
  '#f97316', // orange
  '#14b8a6'  // teal
];

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initCanvas();
  await loadProvincesAndVocab();
  await loadQuickDirectories();
  setupKeyboardShortcuts();
  setupOcrListeners();
  
  // Auto-load raw video crops or first available directory
  const select = document.getElementById('quick-dir-select');
  if (select && select.options.length > 1) {
    select.selectedIndex = 1;
    onQuickDirSelect(select.value);
  }
});

// ==========================================
// Mode Switching & Navigation
// ==========================================

function switchMode(mode) {
  state.currentMode = mode;
  
  // Update Tabs
  document.getElementById('tab-detection').classList.toggle('active', mode === 'detection');
  document.getElementById('tab-ocr').classList.toggle('active', mode === 'ocr');
  
  // Update Workspaces
  document.getElementById('detection-workspace').classList.toggle('active', mode === 'detection');
  document.getElementById('ocr-workspace').classList.toggle('active', mode === 'ocr');
  
  // Update Sidebar visibility
  document.getElementById('sidebar-detection').style.display = (mode === 'detection') ? 'flex' : 'none';

  // Reload current directory with mode perspective
  if (state.currentDir) {
    loadDirectory(state.currentDir);
  }
}

// ==========================================
// Directory & Dataset Loading
// ==========================================

async function loadQuickDirectories() {
  try {
    const res = await fetch('/api/quick_dirs');
    const data = await res.json();
    const select = document.getElementById('quick-dir-select');
    select.innerHTML = '<option value="">-- Select Project Dataset Folder --</option>';
    
    data.quick_dirs.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.path;
      const countStr = item.exists ? `(${item.image_count} imgs)` : `[not created]`;
      opt.textContent = `${item.name} ${countStr}`;
      opt.dataset.recommendedMode = item.recommended_mode;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn('Failed to load quick directories:', err);
  }
}

function onQuickDirSelect(dirPath) {
  if (!dirPath) return;
  document.getElementById('dir-input').value = dirPath;
  
  // Auto-switch mode based on recommendation if user hasn't explicitly chosen
  const select = document.getElementById('quick-dir-select');
  const selectedOpt = select.options[select.selectedIndex];
  if (selectedOpt && selectedOpt.dataset.recommendedMode) {
    const recMode = selectedOpt.dataset.recommendedMode;
    if (recMode !== state.currentMode) {
      switchMode(recMode);
      return;
    }
  }
  
  loadDirectory(dirPath);
}

async function loadDirectory(customDir) {
  const dirPath = customDir || document.getElementById('dir-input').value.trim();
  if (!dirPath) return;

  showToast('Loading dataset...');
  try {
    const res = await fetch(`/api/dataset_items?dir=${encodeURIComponent(dirPath)}&mode=${state.currentMode}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      showToast(data.error || 'Failed to open directory', 'error');
      return;
    }

    state.currentDir = data.folder;
    document.getElementById('dir-input').value = data.folder;
    state.items = data.items || [];
    
    if (data.classes && data.classes.length > 0) {
      state.classes = data.classes;
      renderClassList();
    }

    filterFiles();
    updateDatasetStats();

    if (state.items.length > 0) {
      selectImageIndex(0);
      showToast(`Loaded ${state.items.length} images`);
    } else {
      state.currentIndex = -1;
      clearCanvas();
      showToast('No image files found in folder', 'warning');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

function updateDatasetStats() {
  const total = state.items.length;
  const labeled = state.items.filter(i => i.has_label).length;
  const remaining = total - labeled;
  const percent = total > 0 ? Math.round((labeled / total) * 100) : 0;

  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-labeled').textContent = labeled;
  document.getElementById('stat-remaining').textContent = remaining;
  document.getElementById('progress-percent').textContent = `${percent}%`;
  document.getElementById('progress-bar-fill').style.width = `${percent}%`;
}

// ==========================================
// File Search & Filtering
// ==========================================

function setFileFilter(filter) {
  state.filterStatus = filter;
  document.querySelectorAll('.filter-pills .pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filter);
  });
  filterFiles();
}

function filterFiles() {
  state.searchQuery = document.getElementById('file-search').value.toLowerCase();
  
  state.filteredItems = state.items.filter(item => {
    // Search match
    const matchesSearch = item.name.toLowerCase().includes(state.searchQuery);
    if (!matchesSearch) return false;

    // Filter match
    if (state.filterStatus === 'labeled') return item.has_label;
    if (state.filterStatus === 'unlabeled') return !item.has_label;
    return true;
  });

  renderFileList();
}

function renderFileList() {
  const container = document.getElementById('file-list-container');
  if (state.filteredItems.length === 0) {
    container.innerHTML = '<div class="empty-hint">No matching images found.</div>';
    return;
  }

  container.innerHTML = '';
  state.filteredItems.forEach((item, fIdx) => {
    const el = document.createElement('div');
    el.className = 'file-item';
    if (item.path === (state.items[state.currentIndex] || {}).path) {
      el.classList.add('active');
    }

    el.innerHTML = `
      <span class="file-item-name" title="${item.name}">${item.name}</span>
      <span class="status-dot ${item.has_label ? 'labeled' : ''}" title="${item.has_label ? 'Labeled' : 'Unlabeled'}"></span>
    `;

    el.addEventListener('click', () => {
      const realIdx = state.items.findIndex(i => i.path === item.path);
      if (realIdx >= 0) selectImageIndex(realIdx);
    });

    container.appendChild(el);
  });
}

function selectImageIndex(index) {
  if (index < 0 || index >= state.items.length) return;
  state.currentIndex = index;
  const currentItem = state.items[index];

  // Update counter badges
  document.getElementById('current-file-name').textContent = currentItem.name;
  document.getElementById('current-file-counter').textContent = `(${index + 1} of ${state.items.length})`;

  // Highlight in file list
  document.querySelectorAll('.file-item').forEach((el, idx) => {
    const item = state.filteredItems[idx];
    if (item) el.classList.toggle('active', item.path === currentItem.path);
  });

  if (state.currentMode === 'detection') {
    loadDetectionImage(currentItem);
  } else {
    loadOcrImage(currentItem);
  }
}

function navigateImage(direction) {
  const newIdx = state.currentIndex + direction;
  if (newIdx >= 0 && newIdx < state.items.length) {
    selectImageIndex(newIdx);
  } else {
    showToast(direction > 0 ? 'Reached end of dataset' : 'At first image', 'info');
  }
}

async function discardCurrentImage() {
  if (state.currentIndex < 0) return;
  const item = state.items[state.currentIndex];
  if (!confirm(`Move "${item.name}" to _discarded folder?`)) return;

  try {
    const res = await fetch('/api/file/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: item.path })
    });
    const data = await res.json();
    if (data.success) {
      showToast('Image moved to trash', 'success');
      state.items.splice(state.currentIndex, 1);
      filterFiles();
      updateDatasetStats();
      if (state.currentIndex >= state.items.length) {
        state.currentIndex = state.items.length - 1;
      }
      selectImageIndex(state.currentIndex);
    } else {
      showToast(data.error || 'Failed to discard file', 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ==========================================
// Mode 1: YOLO Detection Canvas Editor
// ==========================================

let canvas, ctx;

function initCanvas() {
  canvas = document.getElementById('detection-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');
  
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Canvas Mouse & Interaction Events
  canvas.addEventListener('mousedown', onCanvasMouseDown);
  canvas.addEventListener('mousemove', onCanvasMouseMove);
  window.addEventListener('mouseup', onCanvasMouseUp);
  canvas.addEventListener('wheel', onCanvasWheel, { passive: false });
}

function resizeCanvas() {
  const wrapper = document.getElementById('canvas-wrapper');
  if (!wrapper || !canvas) return;
  canvas.width = wrapper.clientWidth;
  canvas.height = wrapper.clientHeight;
  renderCanvas();
}

function loadDetectionImage(item) {
  state.boxes = JSON.parse(JSON.stringify(item.boxes || []));
  state.selectedBoxIndex = -1;
  updateBoxesList();

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    state.canvasImage = img;
    resetCanvasTransform();
    renderCanvas();
  };
  img.src = `/api/image?path=${encodeURIComponent(item.path)}&t=${Date.now()}`;
}

function resetCanvasTransform() {
  if (!state.canvasImage) return;
  const cw = canvas.width;
  const ch = canvas.height;
  const iw = state.canvasImage.width;
  const ih = state.canvasImage.height;

  const scale = Math.min((cw - 40) / iw, (ch - 40) / ih, 1.5);
  state.canvasScale = scale;
  state.canvasPanX = (cw - iw * scale) / 2;
  state.canvasPanY = (ch - ih * scale) / 2;
  renderCanvas();
}

function canvasZoom(factor) {
  state.canvasScale = Math.max(0.1, Math.min(10.0, state.canvasScale * factor));
  renderCanvas();
}

function onCanvasWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
  const newScale = Math.max(0.1, Math.min(10.0, state.canvasScale * zoomFactor));

  // Zoom towards mouse pointer
  state.canvasPanX = mx - (mx - state.canvasPanX) * (newScale / state.canvasScale);
  state.canvasPanY = my - (my - state.canvasPanY) * (newScale / state.canvasScale);
  state.canvasScale = newScale;

  renderCanvas();
}

// Convert Canvas Window Coord -> Image Normalized Coords [0..1]
function windowToNormalized(wx, wy) {
  if (!state.canvasImage) return { x: 0, y: 0 };
  const imgX = (wx - state.canvasPanX) / state.canvasScale;
  const imgY = (wy - state.canvasPanY) / state.canvasScale;
  return {
    x: Math.max(0, Math.min(1, imgX / state.canvasImage.width)),
    y: Math.max(0, Math.min(1, imgY / state.canvasImage.height))
  };
}

// Convert Normalized [0..1] -> Canvas Pixel Coord
function normalizedToCanvas(normX, normY) {
  if (!state.canvasImage) return { x: 0, y: 0 };
  const imgX = normX * state.canvasImage.width;
  const imgY = normY * state.canvasImage.height;
  return {
    x: imgX * state.canvasScale + state.canvasPanX,
    y: imgY * state.canvasScale + state.canvasPanY
  };
}

// Convert YOLO (xc, yc, w, h) -> Box Corner Rect (x, y, w, h) in pixels
function yoloBoxToCanvasRect(box) {
  const imgW = state.canvasImage.width;
  const imgH = state.canvasImage.height;
  
  const boxW = box.width * imgW * state.canvasScale;
  const boxH = box.height * imgH * state.canvasScale;
  const boxX = (box.x_center * imgW - (box.width * imgW) / 2) * state.canvasScale + state.canvasPanX;
  const boxY = (box.y_center * imgH - (box.height * imgH) / 2) * state.canvasScale + state.canvasPanY;

  return { x: boxX, y: boxY, w: boxW, h: boxH };
}

function onCanvasMouseDown(e) {
  if (!state.canvasImage) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  // Space / Alt / Middle-click -> Pan
  if (e.spaceKey || e.altKey || e.button === 1 || e.buttons === 4) {
    state.isPanning = true;
    state.dragStartX = mx - state.canvasPanX;
    state.dragStartY = my - state.canvasPanY;
    return;
  }

  // Left Click
  if (e.button === 0) {
    // 1. Check if clicking on resize handle of selected box
    if (state.selectedBoxIndex >= 0) {
      const handle = getResizeHandleAtPoint(mx, my, state.boxes[state.selectedBoxIndex]);
      if (handle) {
        state.isResizing = true;
        state.resizeHandle = handle;
        state.dragStartX = mx;
        state.dragStartY = my;
        return;
      }
    }

    // 2. Check if clicking inside an existing box (selection / drag)
    for (let i = state.boxes.length - 1; i >= 0; i--) {
      const r = yoloBoxToCanvasRect(state.boxes[i]);
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
        state.selectedBoxIndex = i;
        state.isDraggingBox = true;
        state.dragStartX = mx;
        state.dragStartY = my;
        updateBoxesList();
        renderCanvas();
        return;
      }
    }

    // 3. Otherwise start drawing a new bounding box
    state.selectedBoxIndex = -1;
    state.isDrawing = true;
    state.dragStartX = mx;
    state.dragStartY = my;
    state.currentMouseX = mx;
    state.currentMouseY = my;
    updateBoxesList();
    renderCanvas();
  }
}

function onCanvasMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (state.isPanning) {
    state.canvasPanX = mx - state.dragStartX;
    state.canvasPanY = my - state.dragStartY;
    renderCanvas();
    return;
  }

  if (state.isResizing && state.selectedBoxIndex >= 0) {
    resizeSelectedBox(mx, my);
    renderCanvas();
    return;
  }

  if (state.isDraggingBox && state.selectedBoxIndex >= 0) {
    const dx = (mx - state.dragStartX) / (state.canvasImage.width * state.canvasScale);
    const dy = (my - state.dragStartY) / (state.canvasImage.height * state.canvasScale);
    
    const box = state.boxes[state.selectedBoxIndex];
    box.x_center = Math.max(box.width / 2, Math.min(1 - box.width / 2, box.x_center + dx));
    box.y_center = Math.max(box.height / 2, Math.min(1 - box.height / 2, box.y_center + dy));

    state.dragStartX = mx;
    state.dragStartY = my;
    renderCanvas();
    return;
  }

  if (state.isDrawing) {
    state.currentMouseX = mx;
    state.currentMouseY = my;
    renderCanvas();
    return;
  }

  // Update cursor if hovering over handles or boxes
  updateCursor(mx, my);
}

function onCanvasMouseUp(e) {
  if (state.isPanning) {
    state.isPanning = false;
    return;
  }

  if (state.isResizing) {
    state.isResizing = false;
    state.resizeHandle = null;
    updateBoxesList();
    return;
  }

  if (state.isDraggingBox) {
    state.isDraggingBox = false;
    updateBoxesList();
    return;
  }

  if (state.isDrawing) {
    state.isDrawing = false;
    // Calculate new box
    const p1 = windowToNormalized(state.dragStartX, state.dragStartY);
    const p2 = windowToNormalized(state.currentMouseX, state.currentMouseY);

    const xMin = Math.min(p1.x, p2.x);
    const xMax = Math.max(p1.x, p2.x);
    const yMin = Math.min(p1.y, p2.y);
    const yMax = Math.max(p1.y, p2.y);

    const w = xMax - xMin;
    const h = yMax - yMin;

    // Minimum size threshold (e.g. at least 10px equivalent)
    if (w * state.canvasImage.width > 8 && h * state.canvasImage.height > 8) {
      state.boxes.push({
        class_id: state.selectedClassId,
        x_center: xMin + w / 2,
        y_center: yMin + h / 2,
        width: w,
        height: h
      });
      state.selectedBoxIndex = state.boxes.length - 1;
      updateBoxesList();
    }
    renderCanvas();
  }
}

function getResizeHandleAtPoint(mx, my, box) {
  const r = yoloBoxToCanvasRect(box);
  const handleSize = 8;
  const handles = {
    nw: { x: r.x, y: r.y },
    ne: { x: r.x + r.w, y: r.y },
    se: { x: r.x + r.w, y: r.y + r.h },
    sw: { x: r.x, y: r.y + r.h }
  };

  for (const [key, pt] of Object.entries(handles)) {
    if (Math.abs(mx - pt.x) <= handleSize && Math.abs(my - pt.y) <= handleSize) {
      return key;
    }
  }
  return null;
}

function resizeSelectedBox(mx, my) {
  const box = state.boxes[state.selectedBoxIndex];
  const pNorm = windowToNormalized(mx, my);
  const imgW = state.canvasImage.width;
  const imgH = state.canvasImage.height;

  let x1 = box.x_center - box.width / 2;
  let y1 = box.y_center - box.height / 2;
  let x2 = box.x_center + box.width / 2;
  let y2 = box.y_center + box.height / 2;

  if (state.resizeHandle === 'nw') {
    x1 = Math.min(x2 - 0.01, pNorm.x);
    y1 = Math.min(y2 - 0.01, pNorm.y);
  } else if (state.resizeHandle === 'ne') {
    x2 = Math.max(x1 + 0.01, pNorm.x);
    y1 = Math.min(y2 - 0.01, pNorm.y);
  } else if (state.resizeHandle === 'se') {
    x2 = Math.max(x1 + 0.01, pNorm.x);
    y2 = Math.max(y1 + 0.01, pNorm.y);
  } else if (state.resizeHandle === 'sw') {
    x1 = Math.min(x2 - 0.01, pNorm.x);
    y2 = Math.max(y1 + 0.01, pNorm.y);
  }

  box.width = x2 - x1;
  box.height = y2 - y1;
  box.x_center = x1 + box.width / 2;
  box.y_center = y1 + box.height / 2;
}

function updateCursor(mx, my) {
  if (state.selectedBoxIndex >= 0) {
    const handle = getResizeHandleAtPoint(mx, my, state.boxes[state.selectedBoxIndex]);
    if (handle) {
      canvas.style.cursor = (handle === 'nw' || handle === 'se') ? 'nwse-resize' : 'nesw-resize';
      return;
    }
  }
  canvas.style.cursor = 'crosshair';
}

function renderCanvas() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.canvasImage) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No image selected', canvas.width / 2, canvas.height / 2);
    return;
  }

  // Draw Main Image
  const iw = state.canvasImage.width * state.canvasScale;
  const ih = state.canvasImage.height * state.canvasScale;
  ctx.drawImage(state.canvasImage, state.canvasPanX, state.canvasPanY, iw, ih);

  // Draw Existing Bounding Boxes
  state.boxes.forEach((box, idx) => {
    const r = yoloBoxToCanvasRect(box);
    const isSelected = (idx === state.selectedBoxIndex);
    const color = CLASS_COLORS[box.class_id % CLASS_COLORS.length] || '#10b981';

    // Box stroke
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeRect(r.x, r.y, r.w, r.h);

    // Box fill translucent
    ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.2)' : 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(r.x, r.y, r.w, r.h);

    // Class Tag Label
    const clsName = state.classes[box.class_id] || `Class ${box.class_id}`;
    const tagText = box.confidence ? `${clsName} ${(box.confidence * 100).toFixed(0)}%` : clsName;
    ctx.font = 'bold 11px Inter, sans-serif';
    const textWidth = ctx.measureText(tagText).width;

    ctx.fillStyle = color;
    ctx.fillRect(r.x, Math.max(0, r.y - 18), textWidth + 10, 18);

    ctx.fillStyle = '#fff';
    ctx.fillText(tagText, r.x + 5, Math.max(13, r.y - 5));

    // Draw resize handles if selected
    if (isSelected) {
      drawHandle(r.x, r.y);
      drawHandle(r.x + r.w, r.y);
      drawHandle(r.x + r.w, r.y + r.h);
      drawHandle(r.x, r.y + r.h);
    }
  });

  // Draw Active Drawing Box Preview
  if (state.isDrawing) {
    const x = Math.min(state.dragStartX, state.currentMouseX);
    const y = Math.min(state.dragStartY, state.currentMouseY);
    const w = Math.abs(state.currentMouseX - state.dragStartX);
    const h = Math.abs(state.currentMouseY - state.dragStartY);

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
    ctx.fillRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function drawHandle(x, y) {
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.fillRect(x - 4, y - 4, 8, 8);
  ctx.strokeRect(x - 4, y - 4, 8, 8);
}

function clearCanvas() {
  state.canvasImage = null;
  state.boxes = [];
  state.selectedBoxIndex = -1;
  renderCanvas();
  updateBoxesList();
}

function clearAllBoxes() {
  if (state.boxes.length === 0) return;
  state.boxes = [];
  state.selectedBoxIndex = -1;
  updateBoxesList();
  renderCanvas();
  showToast('Cleared all boxes for this image');
}

// Bounding Box List in Right Sidebar
function updateBoxesList() {
  const container = document.getElementById('boxes-list');
  const countBadge = document.getElementById('box-count-badge');
  if (!container) return;

  countBadge.textContent = state.boxes.length;
  if (state.boxes.length === 0) {
    container.innerHTML = '<div class="empty-hint">Click & drag on image to draw a bounding box.</div>';
    return;
  }

  container.innerHTML = '';
  state.boxes.forEach((box, idx) => {
    const el = document.createElement('div');
    el.className = `box-item ${idx === state.selectedBoxIndex ? 'selected' : ''}`;
    const clsName = state.classes[box.class_id] || `Class ${box.class_id}`;
    const color = CLASS_COLORS[box.class_id % CLASS_COLORS.length];

    el.innerHTML = `
      <div style="display: flex; align-items: center; gap: 6px;">
        <span class="class-color-indicator" style="background: ${color}"></span>
        <b>#${idx + 1} ${clsName}</b>
      </div>
      <button class="btn btn-xs btn-danger" onclick="deleteBox(${idx}, event)">✕</button>
    `;

    el.addEventListener('click', () => {
      state.selectedBoxIndex = idx;
      updateBoxesList();
      renderCanvas();
    });

    container.appendChild(el);
  });
}

function deleteBox(idx, e) {
  if (e) e.stopPropagation();
  state.boxes.splice(idx, 1);
  if (state.selectedBoxIndex === idx) {
    state.selectedBoxIndex = -1;
  } else if (state.selectedBoxIndex > idx) {
    state.selectedBoxIndex--;
  }
  updateBoxesList();
  renderCanvas();
}

// Class Selector
function renderClassList() {
  const list = document.getElementById('class-list');
  if (!list) return;
  list.innerHTML = '';

  state.classes.forEach((cls, idx) => {
    const badge = document.createElement('div');
    badge.className = `class-badge ${idx === state.selectedClassId ? 'active' : ''}`;
    const color = CLASS_COLORS[idx % CLASS_COLORS.length];
    badge.innerHTML = `
      <span><span class="class-color-indicator" style="background: ${color}"></span>${cls}</span>
      <span class="badge badge-info">${idx}</span>
    `;
    badge.addEventListener('click', () => {
      state.selectedClassId = idx;
      if (state.selectedBoxIndex >= 0) {
        state.boxes[state.selectedBoxIndex].class_id = idx;
        updateBoxesList();
        renderCanvas();
      }
      renderClassList();
    });
    list.appendChild(badge);
  });
}

function addClass() {
  const input = document.getElementById('new-class-input');
  const name = input.value.trim();
  if (!name) return;
  if (!state.classes.includes(name)) {
    state.classes.push(name);
    state.selectedClassId = state.classes.length - 1;
    renderClassList();
    input.value = '';
  }
}

async function saveDetectionLabels() {
  if (state.currentIndex < 0) return;
  const item = state.items[state.currentIndex];
  
  try {
    const res = await fetch('/api/detection/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: item.path,
        boxes: state.boxes
      })
    });
    const data = await res.json();
    if (data.success) {
      item.has_label = state.boxes.length > 0;
      item.boxes = JSON.parse(JSON.stringify(state.boxes));
      updateDatasetStats();
      filterFiles();
      showToast(data.message, 'success');
    } else {
      showToast(data.error || 'Failed to save labels', 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

async function runAiAutoDetect() {
  if (state.currentIndex < 0) return;
  const item = state.items[state.currentIndex];
  const conf = parseFloat(document.getElementById('yolo-conf').value) || 0.25;

  const btn = document.getElementById('btn-auto-detect');
  btn.textContent = '⏳ Detecting...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/detection/auto_detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: item.path,
        confidence: conf
      })
    });
    const data = await res.json();
    if (data.success) {
      if (data.boxes.length > 0) {
        state.boxes = data.boxes;
        state.selectedBoxIndex = 0;
        updateBoxesList();
        renderCanvas();
        showToast(`Auto-detected ${data.count} bounding boxes!`, 'success');
      } else {
        showToast('No plates detected at current confidence threshold', 'warning');
      }
    } else {
      showToast(data.error || 'Auto detect error', 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    btn.textContent = '🤖 Auto-Detect (YOLO)';
    btn.disabled = false;
  }
}

function updateConfLabel(val) {
  document.getElementById('conf-label').textContent = `Conf: ${val}`;
}

// ==========================================
// Mode 2: Thai License Plate OCR Editor
// ==========================================

async function loadProvincesAndVocab() {
  try {
    const res = await fetch('/api/provinces');
    const data = await res.json();
    state.provinces = data.provinces || [];
    state.topProvinces = data.top_provinces || [];
    state.consonants = data.consonants || [];

    renderTopProvincesChips();
    renderQuickPad();
    setupProvinceAutocomplete();
  } catch (err) {
    console.warn('Failed to load provinces:', err);
  }
}

function renderTopProvincesChips() {
  const container = document.getElementById('top-provinces-chips');
  if (!container) return;
  container.innerHTML = '';

  state.topProvinces.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'chip-btn';
    btn.textContent = p;
    btn.onclick = () => {
      document.getElementById('province-input').value = p;
      highlightActiveChip(p);
    };
    container.appendChild(btn);
  });
}

function highlightActiveChip(activeProv) {
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent === activeProv);
  });
}

function renderQuickPad() {
  const digitsContainer = document.getElementById('quick-pad-digits');
  const consonantsContainer = document.getElementById('quick-pad-consonants');
  if (!digitsContainer || !consonantsContainer) return;

  digitsContainer.innerHTML = '';
  consonantsContainer.innerHTML = '';

  // Digits 0-9
  for (let d = 0; d <= 9; d++) {
    const btn = document.createElement('button');
    btn.className = 'pad-btn';
    btn.textContent = d;
    btn.onclick = () => appendToPlateNumber(d.toString());
    digitsContainer.appendChild(btn);
  }

  // Thai Consonants
  state.consonants.forEach(ch => {
    const btn = document.createElement('button');
    btn.className = 'pad-btn';
    btn.textContent = ch;
    btn.onclick = () => appendToPlateNumber(ch);
    consonantsContainer.appendChild(btn);
  });
}

function appendToPlateNumber(char) {
  const input = document.getElementById('plate-number-input');
  input.value += char;
  input.focus();
}

function toggleQuickPad() {
  const pad = document.getElementById('quick-pad-body');
  pad.style.display = (pad.style.display === 'none') ? 'block' : 'none';
}

function clearInput(id) {
  const input = document.getElementById(id);
  if (input) {
    input.value = '';
    input.focus();
  }
}

function setupProvinceAutocomplete() {
  const input = document.getElementById('province-input');
  const suggestions = document.getElementById('province-suggestions');
  if (!input || !suggestions) return;

  input.addEventListener('input', () => {
    const val = input.value.trim().toLowerCase();
    if (!val) {
      suggestions.style.display = 'none';
      return;
    }

    const matches = state.provinces.filter(p => p.toLowerCase().includes(val));
    if (matches.length === 0) {
      suggestions.style.display = 'none';
      return;
    }

    suggestions.innerHTML = '';
    matches.slice(0, 8).forEach(prov => {
      const item = document.createElement('div');
      item.className = 'suggestion-item';
      item.textContent = prov;
      item.onclick = () => {
        input.value = prov;
        suggestions.style.display = 'none';
        highlightActiveChip(prov);
      };
      suggestions.appendChild(item);
    });
    suggestions.style.display = 'block';
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.autocomplete-wrapper')) {
      suggestions.style.display = 'none';
    }
  });
}

function setupOcrListeners() {
  // Category radio buttons visual toggle
  document.querySelectorAll('#category-selector .cat-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('#category-selector .cat-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
    });
  });

  // Enter key in plate input or province automatically saves and advances
  const plateInput = document.getElementById('plate-number-input');
  const provInput = document.getElementById('province-input');
  
  [plateInput, provInput].forEach(inp => {
    if (inp) {
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          saveOcrLabels(true);
        }
      });
    }
  });
}

function loadOcrImage(item) {
  const imgEl = document.getElementById('ocr-plate-img');
  imgEl.src = `/api/image?path=${encodeURIComponent(item.path)}&t=${Date.now()}`;
  
  // Fill inputs
  const plateInput = document.getElementById('plate-number-input');
  const provInput = document.getElementById('province-input');
  
  plateInput.value = item.plate_number || '';
  provInput.value = item.province || '';
  highlightActiveChip(item.province || '');

  // Set category radio
  const cat = item.category || 'real_physical';
  const radio = document.querySelector(`input[name="ocr-category"][value="${cat}"]`);
  if (radio) {
    radio.checked = true;
    document.querySelectorAll('#category-selector .cat-card').forEach(c => c.classList.remove('active'));
    radio.closest('.cat-card').classList.add('active');
  }

  // Apply visual filters
  applyOcrFilters();

  // Focus plate number input for lightning fast workflow
  setTimeout(() => plateInput.focus(), 50);
}

// Live Image Filters for OCR
function applyOcrFilters() {
  const imgEl = document.getElementById('ocr-plate-img');
  if (!imgEl) return;

  const bright = document.getElementById('filter-bright').value;
  const contrast = document.getElementById('filter-contrast').value;
  const isGray = state.ocrFilters.grayscale;
  const isInvert = state.ocrFilters.invert;

  const filterStrings = [
    `brightness(${bright}%)`,
    `contrast(${contrast}%)`,
    isGray ? 'grayscale(100%)' : '',
    isInvert ? 'invert(100%)' : ''
  ].filter(Boolean);

  imgEl.style.filter = filterStrings.join(' ');
}

function toggleOcrFilter(filterName) {
  state.ocrFilters[filterName] = !state.ocrFilters[filterName];
  const btn = document.getElementById(`btn-filter-${filterName === 'grayscale' ? 'gray' : 'invert'}`);
  if (btn) btn.classList.toggle('btn-primary', state.ocrFilters[filterName]);
  applyOcrFilters();
}

function resetOcrFilters() {
  state.ocrFilters.grayscale = false;
  state.ocrFilters.invert = false;
  document.getElementById('filter-bright').value = 100;
  document.getElementById('filter-contrast').value = 100;
  document.getElementById('btn-filter-gray').classList.remove('btn-primary');
  document.getElementById('btn-filter-invert').classList.remove('btn-primary');
  applyOcrFilters();
}

async function saveOcrLabels(advanceNext = true) {
  if (state.currentIndex < 0) return;
  const item = state.items[state.currentIndex];
  
  const plateNum = document.getElementById('plate-number-input').value.trim();
  const province = document.getElementById('province-input').value.trim();
  const selectedCatRadio = document.querySelector('input[name="ocr-category"]:checked');
  const category = selectedCatRadio ? selectedCatRadio.value : 'real_physical';

  const doRename = document.getElementById('opt-rename').checked;
  const doManifest = document.getElementById('opt-manifest').checked;
  const doMoveFolder = document.getElementById('opt-move-folder').checked;

  let saveMode = 'none';
  if (doRename && doManifest) saveMode = 'both';
  else if (doRename) saveMode = 'rename';
  else if (doManifest) saveMode = 'metadata';

  try {
    const res = await fetch('/api/ocr/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: item.path,
        plate_number: plateNum,
        province: province,
        category: category,
        save_mode: saveMode,
        move_to_subfolder: doMoveFolder
      })
    });
    const data = await res.json();
    if (data.success) {
      item.plate_number = plateNum;
      item.province = province;
      item.category = category;
      item.has_label = boolLabel(plateNum);
      if (data.new_path) {
        item.path = data.new_path;
        item.name = data.new_filename;
      }
      
      updateDatasetStats();
      filterFiles();
      showToast('Saved OCR labels!', 'success');

      if (advanceNext) {
        navigateImage(1);
      }
    } else {
      showToast(data.error || 'Failed to save OCR label', 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

function boolLabel(str) {
  return typeof str === 'string' && str.trim().length > 0;
}

// ==========================================
// Keyboard Navigation & Global Shortcuts
// ==========================================

function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    // Avoid triggering shortcuts while typing inside search or text inputs (except Enter which is handled above)
    const isInput = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);

    if (e.key === 'ArrowLeft' && !isInput) {
      e.preventDefault();
      navigateImage(-1);
    } else if (e.key === 'ArrowRight' && !isInput) {
      e.preventDefault();
      navigateImage(1);
    } else if ((e.key === 'Delete' || e.key === 'Backspace') && !isInput) {
      if (state.currentMode === 'detection' && state.selectedBoxIndex >= 0) {
        e.preventDefault();
        deleteBox(state.selectedBoxIndex);
      }
    } else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (state.currentMode === 'detection') saveDetectionLabels();
      else saveOcrLabels(false);
    } else if (e.key.toLowerCase() === 'a' && !isInput && state.currentMode === 'detection') {
      e.preventDefault();
      runAiAutoDetect();
    } else if (!isInput && e.key >= '0' && e.key <= '9' && state.currentMode === 'detection') {
      const clsIdx = parseInt(e.key);
      if (clsIdx < state.classes.length) {
        state.selectedClassId = clsIdx;
        if (state.selectedBoxIndex >= 0) {
          state.boxes[state.selectedBoxIndex].class_id = clsIdx;
          updateBoxesList();
          renderCanvas();
        }
        renderClassList();
      }
    }
  });
}

// ==========================================
// Toast Notifications
// ==========================================

let toastTimer = null;
function showToast(msg, type = 'info') {
  const toast = document.getElementById('toast');
  if (!toast) return;
  
  toast.textContent = msg;
  toast.className = 'toast show';
  
  if (type === 'error') {
    toast.style.borderColor = 'var(--accent-red)';
    toast.style.color = '#fca5a5';
  } else if (type === 'success') {
    toast.style.borderColor = 'var(--accent-emerald)';
    toast.style.color = '#6ee7b7';
  } else if (type === 'warning') {
    toast.style.borderColor = 'var(--accent-amber)';
    toast.style.color = '#fcd34d';
  } else {
    toast.style.borderColor = 'var(--accent-blue)';
    toast.style.color = '#fff';
  }

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
  }, 2400);
}
