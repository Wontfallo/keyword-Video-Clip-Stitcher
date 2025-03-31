# WontClip.py (Corrected and Enhanced)
import sys
import os
import subprocess
import whisper
import re
import uuid
import time
import threading
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, # Added QGridLayout
    QLineEdit, QLabel, QSlider, QCheckBox, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QStatusBar, QMessageBox, QPlainTextEdit,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor # For status colors

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_OUTPUT_FOLDER = 'output_WontClip' # Default relative output folder

# --- Global Whisper Model ---
whisper_model = None # Will be loaded by WhisperLoader

# --- Preset Definitions (FIXED) ---
cpuPresets = [
    {'value': "ultrafast", 'label': "Ultra Fast"}, {'value': "superfast", 'label': "Super Fast"},
    {'value': "veryfast", 'label': "Very Fast"}, {'value': "faster", 'label': "Faster"},
    {'value': "fast", 'label': "Fast"}, {'value': "medium", 'label': "Medium"},
    {'value': "slow", 'label': "Slow"}, {'value': "slower", 'label': "Slower"},
    {'value': "veryslow", 'label': "Very Slow"},
]
gpuPresets = [
    {'value': "p1", 'label': "P1 (Fastest)"}, {'value': "p2", 'label': "P2"},
    {'value': "p3", 'label': "P3"}, {'value': "p4", 'label': "P4 (Medium)"},
    {'value': "p5", 'label': "P5"}, {'value': "p6", 'label': "P6"},
    {'value': "p7", 'label': "P7 (Slowest, Best)"},
]

# --- Load Whisper Model ---
class WhisperLoader(QObject):
    # (WhisperLoader class remains the same)
    finished = pyqtSignal(object, str)
    def run(self):
        model = None; device = "cpu"
        try:
            logger.info("Loading Whisper model (tiny.en)...")
            model = whisper.load_model("tiny.en")
            device = str(model.device)
            logger.info(f"Whisper model loaded successfully. Device: {device}")
        except Exception as e: logger.error(f"Failed to load Whisper model: {e}")
        self.finished.emit(model, device)

# --- Processing Worker ---
class ProcessingWorker(QObject):
    job_updated = pyqtSignal(str, str, object)
    finished = pyqtSignal()

    # Pass the selected base output directory to the worker
    def __init__(self, jobs_to_process, base_output_dir):
        super().__init__()
        self.jobs = jobs_to_process
        self.base_output_dir = base_output_dir # Store the selected output directory
        self._is_running = True

    def stop(self): self._is_running = False

    def run(self):
        global whisper_model
        if not whisper_model:
            logger.critical("Whisper model not loaded, cannot start processing.")
            for job in self.jobs: self.job_updated.emit(job['id'], 'error', {'errorMsg': "Whisper model failed to load."})
            self.finished.emit(); return

        logger.info(f"Processing worker started for {len(self.jobs)} jobs. Output base: {self.base_output_dir}")
        # Ensure the base output directory exists
        try:
            os.makedirs(self.base_output_dir, exist_ok=True)
        except Exception as e:
             logger.error(f"Failed to create base output directory '{self.base_output_dir}': {e}")
             for job in self.jobs: self.job_updated.emit(job['id'], 'error', {'errorMsg': f"Failed to create output dir: {e}"})
             self.finished.emit(); return

        for job in self.jobs:
            if not self._is_running: logger.info(f"Worker stopping before job {job['id']}"); break
            job_id = job['id']; video_path = job['video_path']; keywords = job['keywords']; settings = job['settings']
            logger.info(f"Worker picked up job: {job_id} - {os.path.basename(video_path)}")
            self.job_updated.emit(job_id, 'processing', {})
            try:
                # Pass the base output dir to the processing function
                result = self._perform_video_processing(job_id, video_path, keywords, settings, self.base_output_dir)
                self.job_updated.emit(job_id, 'completed', {'result': result})
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=False)
                self.job_updated.emit(job_id, 'error', {'errorMsg': str(e)})
        logger.info("Processing worker finished."); self.finished.emit()

    # --- Core Processing Logic (Takes base_output_dir) ---
    def _perform_video_processing(self, job_id, video_path, keywords, settings, base_output_dir):
        start_time_total = time.perf_counter(); timings = {}
        # (Settings extraction remains the same)
        buffer_amount = settings.get('bufferAmount', 5)
        execution_provider = settings.get('executionProvider', 'CUDA')
        output_video_encoder = settings.get('outputVideoEncoder', 'hevc_nvenc')
        output_video_preset = settings.get('outputVideoPreset', 'p1')
        should_stitch_clips = settings.get('shouldStitchClips', False)
        custom_folder_name = settings.get('customFolderName')
        cleanup_output = settings.get('cleanupOutput', False)

        logger.info(f"Job {job_id}: Settings - Provider: {execution_provider}, Encoder: {output_video_encoder}, Preset: {output_video_preset}")

        # --- Generate output folder using the provided base directory ---
        if custom_folder_name:
            safe_custom_folder = self._sanitize_foldername(custom_folder_name)
            if not safe_custom_folder: raise ValueError("Invalid custom folder name provided.")
            unique_folder_name = safe_custom_folder
        else:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            unique_folder_name = f"{base_name}_{job_id[:8]}"

        # Use the selected base output directory
        output_dir = os.path.abspath(os.path.join(base_output_dir, unique_folder_name))
        # Basic check to prevent writing outside intended area (though base_output_dir is user-selected)
        # if not output_dir.startswith(os.path.abspath(base_output_dir)):
        #      raise ValueError("Invalid output directory path generated.")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
             logger.error(f"Failed to create job output directory '{output_dir}': {e}")
             raise RuntimeError(f"Failed to create output dir: {e}") from e

        logger.info(f"Job {job_id}: Processing '{os.path.basename(video_path)}' in output folder '{output_dir}'")

        # --- Transcribe ---
        start_time_transcribe = time.perf_counter()
        logger.info(f"Job {job_id}: Starting transcription (Device: {whisper_model.device})...")
        extracted_clips_relative_paths = []
        stitched_video_relative_path = None
        try:
            result = whisper_model.transcribe(video_path) # Use direct path
            timings['transcription_duration'] = time.perf_counter() - start_time_transcribe
            logger.info(f"Job {job_id}: Transcription complete ({timings['transcription_duration']:.2f}s).")
        except Exception as e:
            timings['transcription_duration'] = time.perf_counter() - start_time_transcribe
            logger.error(f"Job {job_id}: Whisper transcription failed: {e}")
            raise RuntimeError(f'Transcription failed: {e}') from e

        # --- Find timestamps ---
        clips_timestamps = self._find_keyword_timestamps(result.get('segments', []), keywords, buffer_amount)
        if not clips_timestamps:
            logger.info(f"Job {job_id}: No keywords found.")
            timings['total_duration'] = time.perf_counter() - start_time_total
            return {'extractedClips': [], 'stitchedVideo': None, 'message': 'No keywords found in the video.', 'timings': timings}

        # --- Extract clips ---
        start_time_extract = time.perf_counter()
        logger.info(f"Job {job_id}: Extracting {len(clips_timestamps)} clips...")
        try:
            # Pass absolute output_dir to extract_clips
            extracted_clips_paths = self._extract_clips(
                video_path, clips_timestamps, output_video_encoder,
                output_video_preset, execution_provider, output_dir, job_id
            )
            timings['extraction_duration'] = time.perf_counter() - start_time_extract
            # Store paths relative to the SELECTED base_output_dir
            extracted_clips_relative_paths = [os.path.relpath(p, base_output_dir).replace('\\', '/') for p in extracted_clips_paths]
            logger.info(f"Job {job_id}: Clips extracted ({timings['extraction_duration']:.2f}s)")
        except Exception as e:
            timings['extraction_duration'] = time.perf_counter() - start_time_extract
            raise e

        # --- Stitch clips if requested ---
        if should_stitch_clips:
            start_time_stitch = time.perf_counter()
            logger.info(f"Job {job_id}: Stitching clips...")
            try:
                # Pass absolute paths to stitch_clips
                stitched_video_path = self._stitch_clips(
                    extracted_clips_paths, output_video_encoder,
                    output_video_preset, execution_provider, output_dir, job_id
                )
                if stitched_video_path:
                    timings['stitching_duration'] = time.perf_counter() - start_time_stitch
                    # Store path relative to the SELECTED base_output_dir
                    stitched_video_relative_path = os.path.relpath(stitched_video_path, base_output_dir).replace('\\', '/')
                    logger.info(f"Job {job_id}: Clips stitched ({timings.get('stitching_duration', 0):.2f}s)")
                    if cleanup_output:
                        logger.info(f"Job {job_id}: Cleaning up individual clips...")
                        for clip_path in extracted_clips_paths:
                            try:
                                if os.path.exists(clip_path): os.remove(clip_path)
                            except Exception as e: logger.error(f"Job {job_id}: Error removing clip {clip_path}: {e}")
            except Exception as e:
                timings['stitching_duration'] = time.perf_counter() - start_time_stitch
                raise e

        timings['total_duration'] = time.perf_counter() - start_time_total
        logger.info(f"Job {job_id}: Processing finished ({timings['total_duration']:.2f}s).")
        return {
            'extractedClips': extracted_clips_relative_paths, # Return relative paths
            'stitchedVideo': stitched_video_relative_path, # Return relative path
            'timings': timings
        }

    # --- (Sanitize, Find Timestamps, Extract, Stitch methods remain the same internally) ---
    def _sanitize_foldername(self, name):
        name = re.sub(r'[^\w\-_\. ]', '_', name); return name.strip()
    def _find_keyword_timestamps(self, segments, keywords, buffer_amount):
        clips = []; processed_indices = set()
        for i, segment in enumerate(segments):
            segment_text_lower = segment.get('text', '').lower()
            segment_start, segment_end = segment.get('start'), segment.get('end')
            if segment_start is None or segment_end is None: continue
            for keyword in keywords:
                clean_keyword = keyword.lower().strip()
                if not clean_keyword: continue
                if re.search(r'\b' + re.escape(clean_keyword) + r'\b', segment_text_lower):
                    if i not in processed_indices:
                        start = max(0, segment_start - buffer_amount); end = segment_end + buffer_amount
                        clips.append({'start': start, 'end': end}); processed_indices.add(i)
                    break
        if not clips: return []
        clips.sort(key=lambda x: x['start']); merged_clips = []
        if not clips: return []
        current_clip = clips[0]
        for next_clip in clips[1:]:
            if next_clip['start'] <= current_clip['end']: current_clip['end'] = max(current_clip['end'], next_clip['end'])
            else: merged_clips.append(current_clip); current_clip = next_clip
        merged_clips.append(current_clip); return merged_clips
    def _extract_clips(self, video_path, clips_timestamps, encoder, preset, execution_provider, output_dir, job_id="N/A"):
        extracted_clips_paths = []
        for i, clip in enumerate(clips_timestamps):
            start, end = clip['start'], clip['end']
            clip_filename = f'clip_{i+1:03d}_{int(start)}_{int(end)}.mp4'
            output_path = os.path.join(output_dir, clip_filename)
            ffmpeg_command = ['ffmpeg', '-y']
            is_nvenc = encoder in ['h264_nvenc', 'hevc_nvenc']
            current_preset = preset
            if is_nvenc and preset in ['ultrafast', 'superfast', 'veryfast']: current_preset = 'p1'
            elif is_nvenc and preset in ['veryslow', 'slower', 'slow']: current_preset = 'p7'
            elif is_nvenc and preset == 'medium': current_preset = 'p5'
            if execution_provider == 'CUDA' and is_nvenc:
                logger.info(f"Job {job_id}: Enabling NVDEC+NVENC. Encoder: {encoder}, Preset: {current_preset}")
                ffmpeg_command.extend(['-hwaccel', 'nvdec'])
            elif execution_provider == 'CUDA' and not is_nvenc: logger.warning(f"Job {job_id}: CUDA selected but CPU encoder ({encoder}) chosen. Using CPU.")
            ffmpeg_command.extend(['-i', video_path,'-ss', str(start),'-to', str(end),'-map', '0:v:0?','-map', '0:a:0?'])
            ffmpeg_command.extend(['-c:v', encoder,'-preset', current_preset,'-c:a', 'aac','-b:a', '192k']) # Use potentially adjusted preset
            ffmpeg_command.extend(['-avoid_negative_ts', 'make_zero'])
            ffmpeg_command.append(output_path)
            logger.debug(f"Job {job_id}: Running FFmpeg command: {' '.join(ffmpeg_command)}")
            result = subprocess.run(ffmpeg_command, check=False, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                 stderr_snippet = result.stderr[-500:] # Get last part of stderr
                 logger.error(f"Job {job_id}: FFmpeg failed (extract). Code: {result.returncode}. Stderr tail: {stderr_snippet}")
                 raise subprocess.CalledProcessError(result.returncode, ffmpeg_command, output=result.stdout, stderr=result.stderr)
            extracted_clips_paths.append(output_path) # Append absolute path
        return extracted_clips_paths
    def _stitch_clips(self, extracted_clips_paths, encoder, preset, execution_provider, output_dir, job_id="N/A"):
        concat_filename, stitched_filename = 'concat.txt', 'stitched_video.mp4'
        concat_filepath, stitched_filepath = os.path.join(output_dir, concat_filename), os.path.join(output_dir, stitched_filename) # Absolute paths
        if not extracted_clips_paths: return None
        try:
            with open(concat_filepath, 'w', encoding='utf-8') as f:
                for clip_path in extracted_clips_paths: # Use absolute paths
                    if not os.path.exists(clip_path): raise FileNotFoundError(f"Job {job_id}: Required clip file not found for concat list: {clip_path}")
                    # Need filename relative to output_dir for concat file when running ffmpeg in output_dir
                    clip_filename_only = os.path.basename(clip_path)
                    safe_clip_filename = clip_filename_only.replace('\\', '/')
                    f.write(f"file '{safe_clip_filename}'\n")
            ffmpeg_command = ['ffmpeg', '-y']
            is_nvenc = encoder in ['h264_nvenc', 'hevc_nvenc']
            current_preset = preset
            if is_nvenc and preset in ['ultrafast', 'superfast', 'veryfast']: current_preset = 'p1'
            elif is_nvenc and preset in ['veryslow', 'slower', 'slow']: current_preset = 'p7'
            elif is_nvenc and preset == 'medium': current_preset = 'p5'
            if execution_provider == 'CUDA' and is_nvenc:
                logger.info(f"Job {job_id}: Enabling NVDEC+NVENC for stitching. Encoder: {encoder}, Preset: {current_preset}")
                ffmpeg_command.extend(['-hwaccel', 'nvdec'])
            elif execution_provider == 'CUDA' and not is_nvenc: logger.warning(f"Job {job_id}: CUDA selected but CPU encoder ({encoder}) chosen for stitching. Using CPU.")
            ffmpeg_command.extend(['-f', 'concat','-safe', '0','-i', concat_filename,'-map', '0:v:0?','-map', '0:a:0?'])
            ffmpeg_command.extend(['-c:v', encoder,'-preset', current_preset,'-c:a', 'aac','-b:a', '192k']) # Use adjusted preset
            ffmpeg_command.extend(['-avoid_negative_ts', 'make_zero'])
            ffmpeg_command.append(stitched_filename)
            logger.debug(f"Job {job_id}: Running FFmpeg stitch command in '{output_dir}': {' '.join(ffmpeg_command)}")
            result = subprocess.run(ffmpeg_command, check=False, cwd=output_dir, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                 stderr_snippet = result.stderr[-500:]
                 logger.error(f"Job {job_id}: FFmpeg failed (stitch). Code: {result.returncode}. Stderr tail: {stderr_snippet}")
                 raise subprocess.CalledProcessError(result.returncode, ffmpeg_command, output=result.stdout, stderr=result.stderr)
            if os.path.exists(stitched_filepath):
                logger.info(f"Job {job_id}: Stitching successful: {stitched_filepath}")
                return stitched_filepath # Return absolute path
            else: raise RuntimeError("FFmpeg command completed but output file is missing.")
        finally:
            if os.path.exists(concat_filepath):
                try: os.remove(concat_filepath)
                except Exception as e: logger.error(f"Job {job_id}: Error removing concat file {concat_filepath}: {e}")
        return None # Return None if stitching failed before completion


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WontClip - Keyword Video Extractor")
        self.setGeometry(100, 100, 1100, 750) # Increased size slightly

        self.jobs = {}
        self.processing_thread = None
        self.whisper_thread = None
        # self.whisper_model = None # Use global whisper_model
        self.whisper_device = "cpu"
        self.selected_files_paths = []
        # Store current output directory path
        self.current_output_directory = os.path.abspath(DEFAULT_OUTPUT_FOLDER)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_input_ui()
        self.setup_options_ui() # Options UI now includes output dir
        self.setup_queue_ui()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Loading Whisper model...")
        self.start_whisper_load()

    def start_whisper_load(self):
        # (Same as before)
        self.whisper_thread = QThread()
        self.whisper_loader = WhisperLoader()
        self.whisper_loader.moveToThread(self.whisper_thread)
        self.whisper_loader.finished.connect(self.on_whisper_loaded)
        self.whisper_thread.started.connect(self.whisper_loader.run)
        self.whisper_thread.finished.connect(self.whisper_thread.deleteLater)
        self.whisper_thread.start()

    def on_whisper_loaded(self, model, device):
        # (Same as before)
        global whisper_model
        whisper_model = model
        self.whisper_device = device
        if model: self.status_bar.showMessage(f"Ready. Whisper model loaded ({device}).", 5000)
        else:
            self.status_bar.showMessage("Error loading Whisper model! Processing will fail.", 0)
            QMessageBox.critical(self, "Model Load Error", "Failed to load the Whisper speech recognition model.")
        if hasattr(self, 'whisper_loader'): self.whisper_loader.deleteLater()
        if self.whisper_thread: self.whisper_thread.quit()

    def setup_input_ui(self):
        # (Same as before)
        input_layout = QHBoxLayout()
        file_group_layout = QVBoxLayout()
        file_label = QLabel("1. Select Video File(s):")
        self.file_button = QPushButton("Browse Files...")
        self.file_button.clicked.connect(self.browse_files)
        self.selected_files_label = QLabel("No files selected.")
        self.selected_files_label.setWordWrap(True)
        file_group_layout.addWidget(file_label); file_group_layout.addWidget(self.file_button); file_group_layout.addWidget(self.selected_files_label)
        input_layout.addLayout(file_group_layout, 1)
        keyword_group_layout = QVBoxLayout()
        keyword_label = QLabel("2. Keywords (comma-separated):")
        self.keyword_input = QLineEdit(); self.keyword_input.setPlaceholderText("e.g., keyword1, another phrase")
        keyword_group_layout.addWidget(keyword_label); keyword_group_layout.addWidget(self.keyword_input)
        input_layout.addLayout(keyword_group_layout, 2)
        self.main_layout.addLayout(input_layout)

    def setup_options_ui(self):
        options_label = QLabel("3. Processing Options:")
        options_label.setStyleSheet("font-weight: bold;")
        self.main_layout.addWidget(options_label)

        # Use a grid layout for better alignment
        options_grid = QGridLayout()
        options_grid.setSpacing(10)

        # Row 0: Buffer & Provider
        self.buffer_label = QLabel() # Label updated dynamically
        self.buffer_slider = QSlider(Qt.Orientation.Horizontal); self.buffer_slider.setRange(0, 300); self.buffer_slider.setValue(50)
        self.buffer_slider.valueChanged.connect(self.update_buffer_label); self.update_buffer_label(50) # Initial update
        options_grid.addWidget(self.buffer_label, 0, 0); options_grid.addWidget(self.buffer_slider, 0, 1)

        provider_label = QLabel("Provider:"); self.provider_combo = QComboBox(); self.provider_combo.addItems(["CUDA", "CPU"]); self.provider_combo.setCurrentText("CUDA")
        options_grid.addWidget(provider_label, 0, 2); options_grid.addWidget(self.provider_combo, 0, 3)

        # Row 1: Encoder & Preset
        encoder_label = QLabel("Encoder:"); self.encoder_combo = QComboBox()
        self.encoder_combo.addItem('hevc_nvenc (GPU H.265)', 'hevc_nvenc'); self.encoder_combo.addItem('h264_nvenc (GPU H.264)', 'h264_nvenc')
        self.encoder_combo.addItem('libx265 (CPU H.265)', 'libx265'); self.encoder_combo.addItem('libx264 (CPU H.264)', 'libx264')
        self.encoder_combo.setCurrentIndex(0); self.encoder_combo.currentTextChanged.connect(self.update_preset_options_from_text) # Use text changed signal
        options_grid.addWidget(encoder_label, 1, 0); options_grid.addWidget(self.encoder_combo, 1, 1)

        preset_label = QLabel("Preset:"); self.preset_combo = QComboBox()
        self.update_preset_options(self.encoder_combo.currentData()) # Initial population using data
        options_grid.addWidget(preset_label, 1, 2); options_grid.addWidget(self.preset_combo, 1, 3)

        # Row 2: Output Directory
        out_dir_label = QLabel("Output Directory:")
        self.output_dir_input = QLineEdit(self.current_output_directory)
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_directory)
        options_grid.addWidget(out_dir_label, 2, 0)
        options_grid.addWidget(self.output_dir_input, 2, 1, 1, 2) # Span 2 columns
        options_grid.addWidget(self.output_dir_button, 2, 3)

        # Row 3: Custom Subfolder & Checkboxes
        folder_label = QLabel("Output Subfolder (opt.):"); self.folder_input = QLineEdit(); self.folder_input.setPlaceholderText("Defaults to video name")
        options_grid.addWidget(folder_label, 3, 0); options_grid.addWidget(self.folder_input, 3, 1)

        self.stitch_checkbox = QCheckBox("Stitch clips"); self.cleanup_checkbox = QCheckBox("Delete clips after stitching")
        self.cleanup_checkbox.setEnabled(False); self.stitch_checkbox.toggled.connect(self.cleanup_checkbox.setEnabled)
        options_grid.addWidget(self.stitch_checkbox, 3, 2); options_grid.addWidget(self.cleanup_checkbox, 3, 3)

        # Set column stretch factors for responsiveness
        options_grid.setColumnStretch(1, 1)
        options_grid.setColumnStretch(3, 1)

        self.main_layout.addLayout(options_grid)

    def update_buffer_label(self, value): self.buffer_label.setText(f"Buffer: {value/10.0:.1f}s")

    def update_preset_options_from_text(self, text): self.update_preset_options(self.encoder_combo.currentData())

    def update_preset_options(self, encoder_value):
        # (Same logic as before)
        is_nvenc = encoder_value and 'nvenc' in encoder_value
        presets = gpuPresets if is_nvenc else cpuPresets
        default_preset = 'p1' if is_nvenc else 'ultrafast'
        current_preset_value = self.preset_combo.currentData()
        self.preset_combo.clear()
        for preset in presets: self.preset_combo.addItem(preset['label'], preset['value'])
        new_index = self.preset_combo.findData(current_preset_value)
        if new_index != -1: self.preset_combo.setCurrentIndex(new_index)
        else:
            default_index = self.preset_combo.findData(default_preset)
            if default_index != -1: self.preset_combo.setCurrentIndex(default_index)
            elif self.preset_combo.count() > 0: self.preset_combo.setCurrentIndex(0)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.current_output_directory)
        if directory:
            self.current_output_directory = directory
            self.output_dir_input.setText(directory)

    def setup_queue_ui(self):
        # (Same as before, maybe adjust column widths)
        queue_label = QLabel("4. Processing Queue:"); queue_label.setStyleSheet("font-weight: bold;")
        self.main_layout.addWidget(queue_label)
        self.queue_table = QTableWidget(); self.queue_table.setColumnCount(6)
        self.queue_table.setHorizontalHeaderLabels(["File", "Keywords", "Status", "Timings", "Actions"])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch) # Stretch Timings
        self.queue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.setColumnHidden(5, True); self.queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.main_layout.addWidget(self.queue_table)
        queue_controls_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Selected File(s) to Queue"); self.add_button.clicked.connect(self.add_to_queue)
        self.process_button = QPushButton("Process Pending Queue"); self.process_button.clicked.connect(self.start_processing)
        self.clear_button = QPushButton("Clear Completed/Errored"); self.clear_button.clicked.connect(self.clear_finished_jobs)
        queue_controls_layout.addWidget(self.add_button); queue_controls_layout.addStretch()
        queue_controls_layout.addWidget(self.clear_button); queue_controls_layout.addWidget(self.process_button)
        self.main_layout.addLayout(queue_controls_layout)

    def browse_files(self):
        # (Same as before)
        files, _ = QFileDialog.getOpenFileNames( self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)" )
        if files: self.selected_files_paths = files; self.selected_files_label.setText(f"{len(files)} file(s) selected: {', '.join(os.path.basename(f) for f in files)}")
        else: self.selected_files_paths = []; self.selected_files_label.setText("No files selected.")

    def add_to_queue(self):
        # (Same as before)
        if not hasattr(self, 'selected_files_paths') or not self.selected_files_paths: QMessageBox.warning(self, "No Files", "Please select video files first."); return
        keywords_text = self.keyword_input.text().strip()
        if not keywords_text: QMessageBox.warning(self, "No Keywords", "Please enter keywords."); return
        keywords_list = [k.strip() for k in keywords_text.split(',') if k.strip()]
        settings = { 'bufferAmount': self.buffer_slider.value() / 10.0, 'executionProvider': self.provider_combo.currentText(), 'outputVideoEncoder': self.encoder_combo.currentData(), 'outputVideoPreset': self.preset_combo.currentData(), 'shouldStitchClips': self.stitch_checkbox.isChecked(), 'customFolderName': self.folder_input.text().strip(), 'cleanupOutput': self.cleanup_checkbox.isChecked(), }
        added_count = 0
        for video_path in self.selected_files_paths:
            job_id = uuid.uuid4().hex
            if job_id not in self.jobs:
                 job_data = { 'id': job_id, 'video_path': video_path, 'filename': os.path.basename(video_path), 'keywords': keywords_list, 'settings': settings, 'status': 'pending', 'result': None, 'errorMsg': None, 'message': None }
                 self.jobs[job_id] = job_data; self.add_job_to_table(job_data); added_count += 1
        self.selected_files_paths = []; self.selected_files_label.setText("No files selected.")
        if added_count > 0: self.status_bar.showMessage(f"{added_count} job(s) added to queue.", 3000)

    def add_job_to_table(self, job_data):
        # (Same as before)
        row_position = self.queue_table.rowCount(); self.queue_table.insertRow(row_position)
        self.queue_table.setItem(row_position, 0, QTableWidgetItem(job_data['filename']))
        self.queue_table.setItem(row_position, 1, QTableWidgetItem(", ".join(job_data['keywords'])))
        self.queue_table.setItem(row_position, 2, QTableWidgetItem(job_data['status']))
        self.queue_table.setItem(row_position, 3, QTableWidgetItem(""))
        action_widget = QWidget(); action_layout = QHBoxLayout(action_widget); action_layout.setContentsMargins(0,0,0,0)
        remove_button = QPushButton("Remove"); remove_button.clicked.connect(lambda _, r=row_position, j=job_data['id']: self.remove_job(r, j))
        action_layout.addWidget(remove_button)
        self.queue_table.setCellWidget(row_position, 4, action_widget)
        self.queue_table.setItem(row_position, 5, QTableWidgetItem(job_data['id']))

    def update_job_in_table(self, job_id, status, data):
        # (Same as before)
        job_data = self.jobs.get(job_id);
        if not job_data: return
        job_data['status'] = status
        if 'errorMsg' in data: job_data['errorMsg'] = data['errorMsg']
        if 'result' in data: job_data['result'] = data['result']
        if 'message' in data: job_data['message'] = data['message']
        for row in range(self.queue_table.rowCount()):
            id_item = self.queue_table.item(row, 5)
            if id_item and id_item.text() == job_id:
                status_item = QTableWidgetItem(status); color = QColor("black")
                if status == 'error': color = QColor("red")
                elif status == 'completed': color = QColor("darkGreen")
                elif status == 'processing': color = QColor("blue")
                elif status == 'queued': color = QColor("orange")
                status_item.setForeground(color); self.queue_table.setItem(row, 2, status_item)
                timings_text = ""
                if status == 'completed':
                    if job_data.get('result', {}).get('timings'):
                        timings = job_data['result']['timings']; parts = []
                        if timings.get('transcription_duration') is not None: parts.append(f"T:{timings['transcription_duration']:.1f}s")
                        if timings.get('extraction_duration') is not None: parts.append(f"E:{timings['extraction_duration']:.1f}s")
                        if timings.get('stitching_duration') is not None: parts.append(f"S:{timings['stitching_duration']:.1f}s")
                        if timings.get('total_duration') is not None: parts.append(f"Tot:{timings['total_duration']:.1f}s")
                        timings_text = " | ".join(parts)
                    elif job_data.get('message'): timings_text = job_data['message']
                elif status == 'error' and job_data.get('errorMsg'): timings_text = f"Error: {job_data['errorMsg'][:100]}..."
                self.queue_table.setItem(row, 3, QTableWidgetItem(timings_text))
                action_widget = self.queue_table.cellWidget(row, 4)
                if action_widget:
                     remove_button = action_widget.findChild(QPushButton)
                     if remove_button: remove_button.setEnabled(status in ['pending', 'completed', 'error'])
                break

    def remove_job(self, row, job_id):
        # (Same as before)
        current_row = -1
        for r in range(self.queue_table.rowCount()):
             id_item = self.queue_table.item(r, 5)
             if id_item and id_item.text() == job_id: current_row = r; break
        if current_row != -1 and job_id in self.jobs:
            status = self.jobs[job_id]['status']
            if status in ['pending', 'completed', 'error']:
                del self.jobs[job_id]; self.queue_table.removeRow(current_row)
                self.status_bar.showMessage(f"Job {job_id[:8]} removed.", 2000)
            else: QMessageBox.warning(self, "Cannot Remove", f"Cannot remove job in '{status}' state.")
        elif job_id in self.jobs: logger.warning(f"Could not find row for job {job_id} to remove."); del self.jobs[job_id]

    def clear_finished_jobs(self):
        # (Same as before)
        rows_to_remove = []; ids_to_remove = []
        for row in range(self.queue_table.rowCount() -1, -1, -1):
            id_item = self.queue_table.item(row, 5)
            if id_item:
                job_id = id_item.text()
                if job_id in self.jobs and self.jobs[job_id]['status'] in ['completed', 'error']:
                    rows_to_remove.append(row); ids_to_remove.append(job_id)
        for row in rows_to_remove: self.queue_table.removeRow(row)
        for job_id in ids_to_remove:
            if job_id in self.jobs: del self.jobs[job_id]
        self.status_bar.showMessage(f"Cleared {len(ids_to_remove)} finished/errored jobs.", 2000)

    def start_processing(self):
        # Use the currently selected output directory
        self.current_output_directory = self.output_dir_input.text().strip()
        if not self.current_output_directory:
             QMessageBox.warning(self, "Output Directory", "Please select a valid output directory.")
             return
        if not os.path.isdir(self.current_output_directory):
             reply = QMessageBox.question(self, "Create Directory?", f"Output directory does not exist:\n{self.current_output_directory}\n\nCreate it?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes:
                 try:
                     os.makedirs(self.current_output_directory, exist_ok=True)
                     logger.info(f"Created output directory: {self.current_output_directory}")
                 except Exception as e:
                     QMessageBox.critical(self, "Error", f"Could not create output directory:\n{e}")
                     return
             else:
                 return # User chose not to create

        if self.processing_thread and self.processing_thread.isRunning(): QMessageBox.information(self, "Processing", "Processing is already running."); return
        if not whisper_model: QMessageBox.critical(self, "Model Error", "Whisper model not loaded."); return
        jobs_to_process = [job for job in self.jobs.values() if job['status'] == 'pending']
        if not jobs_to_process: QMessageBox.information(self, "No Jobs", "No pending jobs in the queue."); return

        for job_data in jobs_to_process: job_data['status'] = 'queued'; self.update_job_in_table(job_data['id'], 'queued', {})
        self.set_ui_enabled(False); self.status_bar.showMessage(f"Starting processing for {len(jobs_to_process)} jobs...")
        self.processing_thread = QThread()
        # Pass the selected output directory to the worker
        self.worker = ProcessingWorker(jobs_to_process, self.current_output_directory)
        self.worker.moveToThread(self.processing_thread)
        self.worker.job_updated.connect(self.update_job_in_table)
        self.worker.finished.connect(self.on_processing_finished)
        self.processing_thread.started.connect(self.worker.run)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)
        self.processing_thread.start()

    def on_processing_finished(self):
        # (Same as before)
        self.status_bar.showMessage("Processing queue finished.", 5000)
        self.set_ui_enabled(True); self.processing_thread = None
        failed_jobs = [job['filename'] for job in self.jobs.values() if job['status'] == 'error']
        if failed_jobs: QMessageBox.warning(self, "Processing Finished", f"Queue finished, but {len(failed_jobs)} job(s) failed:\n" + "\n".join(failed_jobs[:5]) + ("..." if len(failed_jobs)>5 else ""))

    def set_ui_enabled(self, enabled):
        # (Same as before)
        self.file_button.setEnabled(enabled); self.keyword_input.setEnabled(enabled)
        self.buffer_slider.setEnabled(enabled); self.provider_combo.setEnabled(enabled)
        self.encoder_combo.setEnabled(enabled); self.preset_combo.setEnabled(enabled)
        self.folder_input.setEnabled(enabled); self.stitch_checkbox.setEnabled(enabled)
        self.output_dir_input.setEnabled(enabled); self.output_dir_button.setEnabled(enabled) # Enable/disable output dir selection
        self.cleanup_checkbox.setEnabled(enabled and self.stitch_checkbox.isChecked())
        self.add_button.setEnabled(enabled); self.process_button.setEnabled(enabled); self.clear_button.setEnabled(enabled)
        for row in range(self.queue_table.rowCount()):
             widget = self.queue_table.cellWidget(row, 4)
             if widget:
                 remove_button = widget.findChild(QPushButton)
                 if remove_button:
                     job_id_item = self.queue_table.item(row, 5)
                     if job_id_item:
                         job_id = job_id_item.text()
                         if job_id in self.jobs: widget.setEnabled(enabled or self.jobs[job_id]['status'] in ['pending', 'completed', 'error'])

    def closeEvent(self, event):
        # (Same as before)
        logger.info("Close event triggered.")
        if self.whisper_thread and self.whisper_thread.isRunning(): logger.info("Waiting for whisper thread..."); self.whisper_thread.quit(); self.whisper_thread.wait(1000)
        if self.processing_thread and self.processing_thread.isRunning():
            logger.info("Attempting to stop processing worker...");
            if hasattr(self, 'worker'): self.worker.stop()
            self.processing_thread.quit(); logger.info("Waiting for processing thread..."); self.processing_thread.wait(3000)
        logger.info("Exiting application."); event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    # Default output dir relative to script location if run directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_full_output_path = os.path.join(script_dir, DEFAULT_OUTPUT_FOLDER)
    os.makedirs(default_full_output_path, exist_ok=True) # Ensure default exists

    app = QApplication(sys.argv)
    # Optional: Apply a style
    # app.setStyle('Fusion')
    main_window = MainWindow()
    # Set default output path in UI after window creation
    main_window.current_output_directory = default_full_output_path
    main_window.output_dir_input.setText(default_full_output_path)
    main_window.show()
    sys.exit(app.exec())