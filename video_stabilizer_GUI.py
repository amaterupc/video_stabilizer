import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES  # TkinterDnD2のインポート
from moviepy.editor import VideoFileClip

# Function to stabilize the video using optical flow
def stabilize_video(input_path, temp_output_path, radius, scaling_factor, zoom_factor, progress_bar):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    progress_bar['maximum'] = n_frames - 1

    for i in range(n_frames-1):
        ret, curr = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        
        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

        progress_bar['value'] = i
        progress_bar.update_idletasks()

    trajectory = np.cumsum(transforms, axis=0)
    smooth_trajectory = smooth(trajectory, radius)
    difference = (smooth_trajectory - trajectory) * scaling_factor
    transforms_smooth = transforms + difference
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(n_frames-1):
        ret, frame = cap.read()
        if not ret:
            break
        
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized_zoomed = cv2.resize(frame_stabilized, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        
        center_x, center_y = frame_stabilized_zoomed.shape[1] // 2, frame_stabilized_zoomed.shape[0] // 2
        cropped_frame = frame_stabilized_zoomed[
            center_y - height // 2:center_y + height // 2,
            center_x - width // 2:center_x + width // 2
        ]
        
        out.write(cropped_frame)

        progress_bar['value'] = i
        progress_bar.update_idletasks()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def smooth(trajectory, radius=5):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(radius, len(trajectory)-radius):
        smoothed_trajectory[i] = np.mean(trajectory[i-radius:i+radius], axis=0)
    return smoothed_trajectory

def merge_audio(input_video, stabilized_video, output_video):
    original_clip = VideoFileClip(input_video)
    stabilized_clip = VideoFileClip(stabilized_video).without_audio()
    final_clip = stabilized_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

# GUI setup with drag-and-drop support
def run_gui():
    def select_input_file():
        filename = filedialog.askopenfilename(title="Select Input Video", filetypes=[("Video Files", "*.mp4")])
        input_entry.delete(0, tk.END)
        input_entry.insert(0, filename)

    def select_output_file():
        filename = filedialog.asksaveasfilename(title="Save Output Video", defaultextension=".mp4", filetypes=[("MP4 Files", "*.mp4")])
        output_entry.delete(0, tk.END)
        output_entry.insert(0, filename)

    def on_drop(event):
        input_entry.delete(0, tk.END)
        input_entry.insert(0, event.data)

    def start_stabilization():
        input_file = input_entry.get()
        output_file = output_entry.get()
        radius = int(radius_entry.get())
        scaling_factor = float(scaling_entry.get())
        zoom_factor = float(zoom_entry.get())

        if not input_file or not output_file:
            messagebox.showwarning("Error", "Please select input and output files")
            return
        
        try:
            temp_output = "temp_stabilized.mp4"
            stabilize_video(input_file, temp_output, radius, scaling_factor, zoom_factor, progress_bar)
            merge_audio(input_file, temp_output, output_file)
            messagebox.showinfo("Success", "Video stabilization completed!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Create root window with drag-and-drop support
    root = TkinterDnD.Tk()  # Use TkinterDnD.Tk for drag-and-drop support
    root.title("Video Stabilizer")

    tk.Label(root, text="Input Video:").grid(row=0, column=0, padx=10, pady=5)
    input_entry = tk.Entry(root, width=50)
    input_entry.grid(row=0, column=1, padx=10, pady=5)
    input_entry.drop_target_register(DND_FILES)  # Enable drag-and-drop for input_entry
    input_entry.dnd_bind('<<Drop>>', on_drop)

    tk.Button(root, text="Browse", command=select_input_file).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(root, text="Output Video:").grid(row=1, column=0, padx=10, pady=5)
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=select_output_file).grid(row=1, column=2, padx=10, pady=5)

    tk.Label(root, text="Radius:").grid(row=2, column=0, padx=10, pady=5)
    radius_entry = tk.Entry(root)
    radius_entry.insert(0, "5")
    radius_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Scaling Factor:").grid(row=3, column=0, padx=10, pady=5)
    scaling_entry = tk.Entry(root)
    scaling_entry.insert(0, "1.0")
    scaling_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="Zoom Factor:").grid(row=4, column=0, padx=10, pady=5)
    zoom_entry = tk.Entry(root)
    zoom_entry.insert(0, "1.1")
    zoom_entry.grid(row=4, column=1, padx=10, pady=5)

    # Progress bar
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.grid(row=6, column=1, padx=10, pady=10)

    tk.Button(root, text="Start Stabilization", command=start_stabilization).grid(row=5, column=1, padx=10, pady=20)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    run_gui()

