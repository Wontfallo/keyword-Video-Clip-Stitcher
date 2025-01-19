from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QLineEdit, QSpinBox, QCheckBox,
                           QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from video_processor import VideoProcessor

class VideoProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Keyword Extractor")
        self.setGeometry(100, 100, 600, 400)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # File selection
        self.file_label = QLabel("No file selected")
        self.select_button = QPushButton("Select Video File")
        self.select_button.clicked.connect(self.select_file)

        # Keywords input
        self.keywords_label = QLabel("Enter keywords (comma-separated):")
        self.keywords_input = QLineEdit()

        # Buffer time
        self.buffer_label = QLabel("Buffer time (seconds):")
        self.buffer_input = QSpinBox()
        self.buffer_input.setValue(5)
        self.buffer_input.setRange(1, 30)

        # Stitch option
        self.stitch_checkbox = QCheckBox("Stitch video chunks together")
        self.stitch_checkbox.setChecked(True)

        # Process button
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("")

        # Add widgets to layout
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.keywords_label)
        layout.addWidget(self.keywords_input)
        layout.addWidget(self.buffer_label)
        layout.addWidget(self.buffer_input)
        layout.addWidget(self.stitch_checkbox)
        layout.addWidget(self.process_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.video_path = None
        self.worker = None

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        if file_name:
            self.video_path = file_name
            self.file_label.setText(f"Selected: {file_name}")
            self.process_button.setEnabled(True)

    def process_video(self):
        if not self.video_path or not self.keywords_input.text():
            self.status_label.setText("Please select a video and enter keywords")
            return

        self.worker = ProcessingWorker(
            self.video_path,
            self.keywords_input.text(),
            self.buffer_input.value(),
            self.stitch_checkbox.isChecked()
        )
        
        self.worker.progress_update.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.processing_finished)

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def processing_finished(self):
        self.process_button.setEnabled(True)
        self.status_label.setText("Processing completed!")

class ProcessingWorker(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, video_path, keywords, buffer_time, stitch):
        super().__init__()
        self.video_path = video_path
        self.keywords = [k.strip() for k in keywords.split(',')]
        self.buffer_time = buffer_time
        self.stitch = stitch
        self.processor = VideoProcessor()

    def run(self):
        try:
            self.status_update.emit("Transcribing video...")
            self.processor.transcribe_video(self.video_path, self.progress_update)
            
            self.status_update.emit("Processing keywords...")
            num_chunks = self.processor.process_keywords(
                self.keywords, 
                self.buffer_time, 
                self.stitch,
                self.progress_update  # Pass progress callback
            )
            
            if num_chunks == 0:
                self.status_update.emit("No matching segments found!")
            else:
                self.status_update.emit(f"Created {num_chunks} video chunks")
            
            self.progress_update.emit(100)
            self.finished.emit()
        except Exception as e:
            print(f"Error: {str(e)}")  # Print to terminal for debugging
            self.status_update.emit(f"Error: {str(e)}")
            self.finished.emit()