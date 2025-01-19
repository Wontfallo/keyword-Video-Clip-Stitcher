import sys
import os
import whisper
import torch
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QProgressBar, QFileDialog, QSpinBox, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal

def check_nvidia_gpu():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False

def get_output_folder(video_path):
    video_name = Path(video_path).stem
    output_folder = f"{video_name}_clips"
    return os.path.abspath(output_folder)

def get_video_files(directory):
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    video_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    return video_files

class TranscriptionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    batch_progress = pyqtSignal(str)

    def __init__(self, video_path, is_batch=False):
        super().__init__()
        self.video_path = video_path
        self.is_batch = is_batch

    def run(self):
        self.progress.emit("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base").to(device)
        
        if self.is_batch:
            self.batch_progress.emit(f"Transcribing: {os.path.basename(self.video_path)}")
        else:
            self.progress.emit(f"Transcribing video using {device.upper()}...")
            
        result = model.transcribe(self.video_path, fp16=torch.cuda.is_available())
        self.finished.emit(result)

class VideoSlicer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Keyword Slicer")
        self.setMinimumWidth(600)
        self.transcription = None
        self.segments = []
        self.has_nvidia = check_nvidia_gpu()
        self.output_folder = None
        self.batch_videos = []
        self.current_batch_index = 0
        self.current_video_path = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # GPU Status
        cuda_status = "GPU Available (CUDA)" if torch.cuda.is_available() else "Using CPU"
        ffmpeg_status = "NVIDIA encoder available" if self.has_nvidia else "Using CPU encoder"
        gpu_label = QLabel(f"Whisper: {cuda_status}\nFFmpeg: {ffmpeg_status}")
        layout.addWidget(gpu_label)

        # Batch mode checkbox
        self.batch_mode = QCheckBox("Batch Process (Select Directory)")
        layout.addWidget(self.batch_mode)
        self.batch_mode.stateChanged.connect(self.toggle_batch_mode)

        # Video file/directory selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select video file or directory...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Keywords input
        keywords_layout = QHBoxLayout()
        keywords_layout.addWidget(QLabel("Keywords (comma-separated):"))
        self.keywords_input = QLineEdit()
        keywords_layout.addWidget(self.keywords_input)
        layout.addLayout(keywords_layout)

        # Buffer seconds
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer seconds:"))
        self.buffer_seconds = QSpinBox()
        self.buffer_seconds.setValue(5)
        self.buffer_seconds.setRange(0, 60)
        buffer_layout.addWidget(self.buffer_seconds)
        buffer_layout.addStretch()
        layout.addLayout(buffer_layout)

        # Concatenate option
        self.concatenate_check = QCheckBox("Concatenate clips into single video")
        self.concatenate_check.setChecked(True)
        layout.addWidget(self.concatenate_check)

        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Batch progress
        self.batch_label = QLabel()
        self.batch_label.hide()
        layout.addWidget(self.batch_label)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        layout.addWidget(self.process_btn)

    def toggle_batch_mode(self):
        if self.batch_mode.isChecked():
            self.file_path.setPlaceholderText("Select directory containing videos...")
        else:
            self.file_path.setPlaceholderText("Select video file...")
        self.file_path.clear()

    def browse_file(self):
        if self.batch_mode.isChecked():
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Directory Containing Videos"
            )
            if directory:
                self.file_path.setText(directory)
                self.batch_videos = get_video_files(directory)
                self.update_status(f"Found {len(self.batch_videos)} video files")
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
            )
            if file_name:
                self.file_path.setText(file_name)
                self.output_folder = get_output_folder(file_name)

    def update_status(self, message):
        self.status_label.setText(message)
        self.progress_bar.setFormat(message)

    def process_video(self):
        if not self.file_path.text():
            self.update_status("Please select a video file or directory!")
            return

        if not self.keywords_input.text():
            self.update_status("Please enter at least one keyword!")
            return

        if self.batch_mode.isChecked():
            if not self.batch_videos:
                self.update_status("No video files found in selected directory!")
                return
            self.current_batch_index = 0
            self.batch_label.show()
            self.process_batch_video()
        else:
            self.process_single_video(self.file_path.text())

    def process_batch_video(self):
        if self.current_batch_index >= len(self.batch_videos):
            self.update_status("Batch processing complete!")
            self.batch_label.hide()
            self.process_btn.setEnabled(True)
            return

        current_video = self.batch_videos[self.current_batch_index]
        self.current_video_path = current_video  # Store current video path
        self.batch_label.setText(f"Processing video {self.current_batch_index + 1}/{len(self.batch_videos)}: {os.path.basename(current_video)}")
        self.process_single_video(current_video, is_batch=True)

    def process_single_video(self, video_path, is_batch=False):
        self.current_video_path = video_path  # Store current video path
        self.output_folder = get_output_folder(video_path)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.process_btn.setEnabled(False)
        self.progress_bar.show()
        
        self.worker = TranscriptionWorker(video_path, is_batch)
        self.worker.progress.connect(self.update_status)
        self.worker.batch_progress.connect(lambda msg: self.batch_label.setText(msg))
        self.worker.finished.connect(lambda result: self.handle_transcription(result, is_batch))
        self.worker.start()

    def handle_transcription(self, transcription, is_batch=False):
        self.transcription = transcription
        self.update_status("Finding keywords...")
        
        self.segments = self.find_keyword_timestamps(
            self.keywords_input.text(),
            self.buffer_seconds.value()
        )

        if not self.segments:
            self.update_status("No keywords found in the video!")
            if is_batch:
                self.current_batch_index += 1
                self.process_batch_video()
            else:
                self.process_btn.setEnabled(True)
            return

        self.update_status("Extracting clips...")
        self.extract_clips()

        if self.concatenate_check.isChecked():
            self.update_status("Concatenating clips...")
            self.concatenate_clips()

        if is_batch:
            self.current_batch_index += 1
            self.process_batch_video()
        else:
            self.update_status(f"Processing complete! Output saved to: {self.output_folder}")
            self.process_btn.setEnabled(True)

    def find_keyword_timestamps(self, keywords, buffer_seconds):
        segments = []
        keywords = [k.strip().lower() for k in keywords.split(',')]
        
        for segment in self.transcription['segments']:
            text = segment['text'].lower()
            if any(keyword in text for keyword in keywords):
                start = max(0, segment['start'] - buffer_seconds)
                end = segment['end'] + buffer_seconds
                segments.append((start, end))
        
        return segments

    def extract_clips(self):
        clips_folder = os.path.join(self.output_folder, "clips")
        if not os.path.exists(clips_folder):
            os.makedirs(clips_folder)
        
        for i, (start, end) in enumerate(self.segments):
            output_path = os.path.join(clips_folder, f"clip_{i}.mp4")
            duration = end - start
            
            cmd = [
                'ffmpeg', '-y',
                '-i', self.current_video_path,  # Use current_video_path instead of file_path
                '-ss', str(start),
                '-t', str(duration)
            ]
            
            if self.has_nvidia:
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-b:v', '5M'
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'fast'
                ])
            
            cmd.extend([
                '-c:a', 'aac',
                output_path
            ])
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                self.update_status(f"Created clip {i+1}/{len(self.segments)}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing clip {i}: {e.stderr.decode()}")
                self.update_status(f"Error creating clip {i+1}")

    def concatenate_clips(self):
        if not self.segments:
            return
            
        list_path = os.path.join(self.output_folder, 'files.txt')
        clips_folder = os.path.join(self.output_folder, "clips")
        
        with open(list_path, 'w') as f:
            for i in range(len(self.segments)):
                clip_path = os.path.abspath(os.path.join(clips_folder, f"clip_{i}.mp4"))
                f.write(f"file '{clip_path}'\n")
        
        try:
            output_path = os.path.join(self.output_folder, "combined_video.mp4")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_path
            ]
            
            if self.has_nvidia:
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-b:v', '5M'
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'fast'
                ])
            
            cmd.extend([
                '-c:a', 'aac',
                output_path
            ])
            
            print("Executing command:", ' '.join(cmd))  # Debug output
            subprocess.run(cmd, check=True, capture_output=True)
            self.update_status("Created final compilation")
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating clips: {e.stderr.decode()}")
            self.update_status("Error creating final compilation")
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)

def main():
    app = QApplication(sys.argv)
    window = VideoSlicer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
