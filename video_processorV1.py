import os
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
            print(f"Opening video file: {self.video_path}")
            video = VideoFileClip(self.video_path)

            # Debug: let's ee what methods are available for video
            '''
            print(f"Available methods for video: {dir(video)}")
            print([method for method in dir(video) if not method.startswith('_')])
            print(f"Available properties for video: {video.__dict__.keys()}")
            print(f"Available attributes for video: {video.__dict__.keys()}")
            print(f"Available methods for video: {dir(video)}")
            print(f"Available properties for video: {video.__dict__.keys()}")
            print(f"Available attributes for video: {video.__dict__.keys()}")
            print(f"Available methods for video: {dir(video)}")
            print(f"Available properties for video: {video.__dict__.keys()}")
            print(f"Available attributes for video: {video.__dict__.keys()}")
            '''
            
            chunks = []
            output_dir = os.path.join(os.getcwd(), "output_chunks")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

            for i, segment in enumerate(self.segments):
                text = segment["text"].lower()
                if any(keyword.lower() in text for keyword in keywords):
                    try:
                        print(f"\nProcessing segment {i+1}:")
                        print(f"- Text: {text}")
                        print(f"- Time: {segment['start']} to {segment['end']}")
                        
                        start_time = max(0, segment["start"] - buffer_time)
                        end_time = min(video.duration, segment["end"] + buffer_time)
                        print(f"- Adjusted time: {start_time} to {end_time}")
                        
                        chunk = video.subclipped(start_time, end_time)
                        chunks.append(chunk)
                        
                        chunk_path = os.path.join(output_dir, f"chunk_{len(chunks)}.mp4")
                        print(f"Saving chunk to: {chunk_path}")
                        chunk.write_videofile(chunk_path, logger=None)
                        progress_callback.emit(25 + (65 * (i + 1) // len(self.segments)))
                    except Exception as e:
                        print(f"Error processing segment {i+1}: {str(e)}")
                        raise

            if stitch and chunks:
                try:
                    final_path = os.path.join(os.getcwd(), "final_stitched_video.mp4")
                    print(f"Stitching final video to: {final_path}")
                    final_video = concatenate_videoclips(chunks)
                    final_video.write_videofile(final_path, logger=None)
                except Exception as e:
                    print(f"Error during video stitching: {str(e)}")
                    raise

            video.close()
            for chunk in chunks:
                chunk.close()
            
            return len(chunks)

        except Exception as e:
            print(f"\nFATAL ERROR in process_keywords: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise