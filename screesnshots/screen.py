import cv2
import keyboard
import os

# Define a function to capture video and take screenshots
def capture_screenshots_from_video(video_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    screenshot_count = 0

    while True:
        # Read the frame from the video
        ret, frame = cap.read()

        # If no frame is returned, break the loop (end of video)
        if not ret:
            print("Video ended.")
            break

        # Display the frame
        cv2.imshow("Video", frame)

        # Check for key press 'c' to capture a screenshot
        if keyboard.is_pressed('c'):
            screenshot_name = os.path.join(output_folder, f"screenshot_{screenshot_count}.png")
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved as {screenshot_name}")
            screenshot_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "input.mp4"  # Replace with your video file path
output_folder = "outputs"          # Folder where screenshots will be saved
capture_screenshots_from_video(video_path, output_folder)
