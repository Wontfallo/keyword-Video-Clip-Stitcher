import customtkinter as ctk
import torch
import whisper
import ffmpeg
import os
import time
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import shutil

class Logger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.start_time = None

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}\n"
        self.queue.put(formatted_message)

    def update_text_widget(self):
        while not self.queue.empty():
            message = self.queue.get()
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", message)
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")

class VideoProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.stop_processing = False

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base", device=device)
        self.logger.log(f"Loaded Whisper model on {device}")

    def process_videos(self, input_dir, keywords, buffer_time, progress_callback):
        if not self.model:
            self.load_model()

        input_path = Path(input_dir)
        clips_dir = input_path / "clips"
        clips_dir.mkdir(exist_ok=True)

        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mkv"))
        clip_paths = []
        
        for idx, video_file in enumerate(video_files):
            if self.stop_processing:
                break

            self.logger.log(f"Processing {video_file.name}")
            
            # Transcribe video
            result = self.model.transcribe(str(video_file))
            
            # Process each keyword
            for keyword in keywords:
                keyword = keyword.strip().lower()
                
                # Find timestamps for keyword
                timestamps = self._find_keyword_timestamps(result, keyword)
                
                for i, (start_time, end_time) in enumerate(timestamps):
                    # Add buffer
                    clip_start = max(0, start_time - buffer_time)
                    clip_end = end_time + buffer_time
                    
                    output_path = clips_dir / f"{video_file.stem}_{keyword}_{i}.mp4"
                    
                    # Extract clip using FFmpeg
                    self._extract_clip(
                        str(video_file),
                        str(output_path),
                        clip_start,
                        clip_end
                    )
                    
                    clip_paths.append(output_path)
            
            progress = (idx + 1) / len(video_files) * 100
            progress_callback(progress)

        if clip_paths:
            # Combine all clips
            final_output = clips_dir / "final_combined.mp4"
            self._combine_clips(clip_paths, str(final_output))
            return str(final_output)
        return None

    def _find_keyword_timestamps(self, result, keyword):
        timestamps = []
        for segment in result["segments"]:
            if keyword in segment["text"].lower():
                timestamps.append((segment["start"], segment["end"]))
        return timestamps

    def _extract_clip(self, input_path, output_path, start_time, end_time):
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-c:a', 'aac',
                output_path
            ], check=True, capture_output=True)
            self.logger.log(f"Extracted clip: {output_path}")
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Error extracting clip: {e}", "ERROR")

    def _combine_clips(self, clip_paths, output_path):
        # Create file list
        with open("files.txt", "w") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")

        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', 'files.txt',
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-c:a', 'aac',
                output_path
            ], check=True, capture_output=True)
            self.logger.log(f"Created combined video: {output_path}")
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Error combining clips: {e}", "ERROR")
        finally:
            if os.path.exists("files.txt"):
                os.remove("files.txt")

class VideoProcessorGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Video Keyword Clipper")
        self.root.geometry("800x600")
        ctk.set_appearance_mode("dark")
        
        self.processor = VideoProcessor(None)  # Will be initialized after GUI
        self.setup_gui()
        self.processor.logger = Logger(self.log_text)
        
        self.processing_thread = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Input directory selection
        self.dir_frame = ctk.CTkFrame(self.root)
        self.dir_frame.pack(fill="x", padx=20, pady=10)
        
        self.dir_label = ctk.CTkLabel(self.dir_frame, text="Input Directory:")
        self.dir_label.pack(side="left", padx=5)
        
        self.dir_entry = ctk.CTkEntry(self.dir_frame, width=400)
        self.dir_entry.pack(side="left", padx=5)
        
        self.dir_button = ctk.CTkButton(
            self.dir_frame,
            text="Browse",
            command=self.browse_directory
        )
        self.dir_button.pack(side="left", padx=5)

        # Keywords input
        self.keywords_frame = ctk.CTkFrame(self.root)
        self.keywords_frame.pack(fill="x", padx=20, pady=10)
        
        self.keywords_label = ctk.CTkLabel(
            self.keywords_frame,
            text="Keywords (comma-separated):"
        )
        self.keywords_label.pack(side="left", padx=5)
        
        self.keywords_entry = ctk.CTkEntry(self.keywords_frame, width=400)
        self.keywords_entry.pack(side="left", padx=5)

        # Buffer time input
        self.buffer_frame = ctk.CTkFrame(self.root)
        self.buffer_frame.pack(fill="x", padx=20, pady=10)
        
        self.buffer_label = ctk.CTkLabel(
            self.buffer_frame,
            text="Buffer Time (seconds):"
        )
        self.buffer_label.pack(side="left", padx=5)
        
        self.buffer_entry = ctk.CTkEntry(self.buffer_frame, width=100)
        self.buffer_entry.pack(side="left", padx=5)
        self.buffer_entry.insert(0, "3")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.root)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.set(0)

        # Log text area
        self.log_text = ctk.CTkTextbox(self.root, height=200)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)
        self.log_text.configure(state="disabled")

        # Buttons frame
        self.buttons_frame = ctk.CTkFrame(self.root)
        self.buttons_frame.pack(fill="x", padx=20, pady=10)

        self.start_button = ctk.CTkButton(
            self.buttons_frame,
            text="Start Processing",
            command=self.start_processing
        )
        self.start_button.pack(side="left", padx=5)

        self.open_output_button = ctk.CTkButton(
            self.buttons_frame,
            text="Open Output Directory",
            command=self.open_output_directory
        )
        self.open_output_button.pack(side="left", padx=5)

    def browse_directory(self):
        directory = ctk.filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, directory)

    def update_progress(self, value):
        self.progress_bar.set(value / 100)
        self.processor.logger.update_text_widget()
        self.root.update_idletasks()

    def start_processing(self):
        input_dir = self.dir_entry.get()
        keywords = [k.strip() for k in self.keywords_entry.get().split(",")]
        buffer_time = float(self.buffer_entry.get())

        if not input_dir or not keywords:
            self.processor.logger.log("Please fill in all required fields", "ERROR")
            return

        self.start_button.configure(state="disabled")
        self.processing_thread = threading.Thread(
            target=self.process_videos_thread,
            args=(input_dir, keywords, buffer_time)
        )
        self.processing_thread.start()

        # Start progress updates
        self.root.after(100, self.check_progress)

    def process_videos_thread(self, input_dir, keywords, buffer_time):
        try:
            output_path = self.processor.process_videos(
                input_dir, keywords, buffer_time, self.update_progress
            )
            if output_path:
                self.processor.logger.log(f"Processing complete! Output: {output_path}")
            else:
                self.processor.logger.log("No clips were generated", "WARNING")
        except Exception as e:
            self.processor.logger.log(f"Error during processing: {str(e)}", "ERROR")
        finally:
            self.root.after(0, self.processing_complete)

    def processing_complete(self):
        self.start_button.configure(state="normal")
        self.processor.logger.update_text_widget()

    def check_progress(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.processor.logger.update_text_widget()
            self.root.after(100, self.check_progress)

    def open_output_directory(self):
        input_dir = self.dir_entry.get()
        if input_dir:
            clips_dir = os.path.join(input_dir, "clips")
            if os.path.exists(clips_dir):
                os.startfile(clips_dir)
            else:
                self.processor.logger.log("Output directory does not exist yet", "WARNING")

    def on_closing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.processor.stop_processing = True
            self.processing_thread.join()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoProcessorGUI()
    app.run()
