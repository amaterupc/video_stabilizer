import cv2
import numpy as np
import argparse
from moviepy.editor import VideoFileClip

# Function to stabilize the video using optical flow
def stabilize_video(input_path, temp_output_path, radius, scaling_factor):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    # Read the first frame
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # Transform accumulator
    transforms = np.zeros((n_frames-1, 3), np.float32)
    
    for i in range(n_frames-1):
        ret, curr = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        
        # Extract translation and rotation
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        
        # Store transformation
        transforms[i] = [dx, dy, da]
        
        # Move to the next frame
        prev_gray = curr_gray
    
    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)
    
    # Compute smooth trajectory
    smooth_trajectory = smooth(trajectory, radius)
    
    # Compute difference between smoothed and original trajectory
    difference = (smooth_trajectory - trajectory) * scaling_factor
    transforms_smooth = transforms + difference
    
    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(n_frames-1):
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract transformation
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]
        
        # Reconstruct transformation matrix
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        
        # Write the frame to the output video
        out.write(frame_stabilized)
    
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to smooth trajectory using a sliding window
def smooth(trajectory, radius=5):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(radius, len(trajectory)-radius):
        smoothed_trajectory[i] = np.mean(trajectory[i-radius:i+radius], axis=0)
    return smoothed_trajectory

# Function to combine stabilized video with original audio
def merge_audio(input_video, stabilized_video, output_video):
    # Load the original video with audio using moviepy
    original_clip = VideoFileClip(input_video)
    
    # Load the stabilized video (without audio)
    stabilized_clip = VideoFileClip(stabilized_video).without_audio()
    
    # Combine the original audio with the stabilized video
    final_clip = stabilized_clip.set_audio(original_clip.audio)
    
    # Write the final output video with audio
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

# Main function
if __name__ == "__main__":
    # Argument parser for command-line input
    parser = argparse.ArgumentParser(description="Video Stabilization Script")
    parser.add_argument('input', type=str, help="Input video file path")
    parser.add_argument('output', type=str, help="Output video file path")
    parser.add_argument('-r', '--radius', type=int, default=5, help="Smoothing radius for trajectory (default: 5)")
    parser.add_argument('-s', '--scaling_factor', type=float, default=1.0, help="Scaling factor for stabilization (default: 1.0)")
    
    args = parser.parse_args()
    
    # Temporary output file for video stabilization (without audio)
    temp_output = "temp_stabilized.mp4"
    
    # Run the stabilization function
    stabilize_video(args.input, temp_output, args.radius, args.scaling_factor)
    
    # Merge the original audio with the stabilized video
    merge_audio(args.input, temp_output, args.output)

