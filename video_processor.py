import os
import time
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
import whisper

class VideoProcessor:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.video_path = None
        self.segments = None

    def transcribe_video(self, video_path, progress_callback):
        self.video_path = video_path
        result = self.model.transcribe(video_path)
        self.segments = result["segments"]
        progress_callback.emit(25)

    def process_keywords(self, keywords, buffer_time, stitch, progress_callback):
        if not self.video_path or not self.segments:
            raise ValueError("No video has been transcribed yet")

        try:
            start_time = time.time()
            print(f"\nStarting processing at: {time.strftime('%H:%M:%S')}")
            
            # Create output directory based on video name
            video_name = Path(self.video_path).stem
            output_dir = os.path.join(os.getcwd(), f"processed_{video_name}")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

            print(f"Opening video file: {self.video_path}")
            video = VideoFileClip(self.video_path)
            chunks = []

            for i, segment in enumerate(self.segments):
                text = segment["text"].lower()
                if any(keyword.lower() in text for keyword in keywords):
                    try:
                        print(f"\nProcessing segment {i+1}/{len(self.segments)}:")
                        print(f"- Text: {text}")
                        print(f"- Time: {segment['start']:.2f} to {segment['end']:.2f}")
                        
                        start_time_clip = max(0, segment["start"] - buffer_time)
                        end_time_clip = min(video.duration, segment["end"] + buffer_time)
                        print(f"- Adjusted time: {start_time_clip:.2f} to {end_time_clip:.2f}")
                        
                        chunk = video.subclipped(start_time_clip, end_time_clip)
                        chunks.append(chunk)
                        
                        chunk_path = os.path.join(output_dir, f"chunk_{len(chunks):03d}.mp4")
                        print(f"Saving chunk to: {chunk_path}")
                        
                        # GPU-accelerated chunk encoding
                        chunk.write_videofile(chunk_path,
                                           codec='hevc_nvenc',
                                           threads=24,
                                           bitrate='20M',
                                           preset='fast',
                                           audio_codec='aac',
                                           logger=None)
                        
                        progress_callback.emit(25 + (65 * (i + 1) // len(self.segments)))
                        
                    except Exception as e:
                        print(f"Error processing segment {i+1}: {str(e)}")
                        raise

            if stitch and chunks:
                try:
                    # Put stitched video in same directory as chunks
                    final_path = os.path.join(output_dir, f"{video_name}_stitched.mp4")
                    print(f"\nStitching final video to: {final_path}")
                    print(f"Start time: {time.strftime('%H:%M:%S')}")
                    
                    final_video = concatenate_videoclips(chunks)
                    final_video.write_videofile(final_path,
                                             codec='hevc_nvenc',
                                             threads=24,
                                             bitrate='20M',
                                             preset='fast',
                                             audio_codec='aac',
                                             logger=None)
                    
                except Exception as e:
                    print(f"Error during video stitching: {str(e)}")
                    raise

            # Cleanup
            video.close()
            for chunk in chunks:
                chunk.close()

            end_time = time.time()
            duration = end_time - start_time
            print(f"\nProcessing completed at: {time.strftime('%H:%M:%S')}")
            print(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            return len(chunks)

        except Exception as e:
            print(f"\nFATAL ERROR in process_keywords: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise