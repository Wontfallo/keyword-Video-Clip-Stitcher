import sys
import os
from PyQt6.QtWidgets import QApplication
from video_gui import VideoProcessorGUI

def main():
    print(f"Working directory: {os.getcwd()}")  # Add this line
    app = QApplication(sys.argv)
    window = VideoProcessorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()