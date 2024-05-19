import cv2
import numpy as np
import glob

# Path to the folder containing keyframes
keyframes_folder = 'path/to/keyframes_folder/*.jpg'  # Change this to your folder

# Load keyframes
keyframes = [cv2.imread(file) for file in sorted(glob.glob(keyframes_folder))]

# Ensure we have keyframes
if len(keyframes) == 0:
    raise ValueError("No keyframes found in the specified folder")

# Assuming all keyframes have the same size, get the dimensions
height, width, _ = keyframes[0].shape

# Create a base image (background) from the first keyframe (or a separate background image)
background = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize the action shot with the background
action_shot = background.copy()

# Iterate through each keyframe and overlay it onto the action shot
for i, frame in enumerate(keyframes):
    # Optionally, you can use mask to blend keyframes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Extract the region of the keyframe to overlay
    frame_fg = cv2.bitwise_and(frame, frame, mask=mask)
    action_shot_bg = cv2.bitwise_and(action_shot, action_shot, mask=mask_inv)

    # Combine the background and the foreground
    action_shot = cv2.add(action_shot_bg, frame_fg)

# Save the resulting action shot
cv2.imwrite('action_shot.jpg', action_shot)

# Display the result
cv2.imshow('Action Shot', action_shot)
cv2.waitKey(0)
cv2.destroyAllWindows()
