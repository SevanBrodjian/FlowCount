import numpy as np
import cv2
import random
import os
from tensorflow.keras import models, layers

def generate_drifting_dots_video(
    output_path='drifting_dots_X.mp4',
    frame_size=(128, 128),
    num_frames=1024,
    fps=24,
    entry_probability=0.05,
    dot_size=2,
    dot_color=(255, 255, 255),  # White dots
    min_velocity=1.0,  # Minimum velocity (pixels per frame)
    max_velocity=4.0,  # Maximum velocity (pixels per frame)
    vertical_irregularity=0.3,  # Amount of vertical irregularity
    min_brightness=0.1,  # Minimum brightness for dots
    max_brightness=1.0,  # Maximum brightness for dots
):
    """
    Generate a video of dots drifting from left to right with some irregularity.
    
    Parameters:
    -----------
    output_path : str
        Path where the video will be saved
    frame_size : tuple
        Size of each frame (height, width)
    num_frames : int
        Number of frames in the video
    fps : int
        Frames per second
    entry_probability : float
        Probability of a new dot entering from the left in each frame
    dot_size : int
        Size of each dot in pixels
    dot_color : tuple
        Color of the dots in BGR format
    min_velocity : float
        Minimum velocity in pixels per frame
    max_velocity : float
        Maximum velocity in pixels per frame
    vertical_irregularity : float
        Amount of vertical irregularity in the motion
    min_brightness : float
        Minimum brightness for dots
    max_brightness : float
        Maximum brightness for dots
    """
    try:
        height, width = frame_size
        print(f"Creating video with size {width}x{height}")
        
        # Initialize video writer with H.264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception(f"Failed to create video writer for {output_path}")
        
        # List to store active dots: each dot is [id, x, y, velocity, contrast]
        dots = []
        dot_id_counter = 0  # Unique ID for each dot
        
        # List to store frame pair information
        frame_pairs = []
        
        # Store previous frame's x positions and their indices
        prev_dots = {}  # Initialize prev_dots as empty for frame 0
        
        # Keep track of total completed traversals
        total_completed_traversals = 0
        
        for frame in range(num_frames):
            # Create black frame
            frame_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Store current frame's dots and their positions
            current_dots = {}  # {dot_id: x}
            
            # Check if new dot should enter
            if random.random() < entry_probability:
                # Random y position for new dot
                margin = dot_size * 3
                y = random.randint(margin, height - margin)
                # Random velocity between min_velocity and max_velocity
                velocity = np.round(random.uniform(min_velocity, max_velocity), decimals=1)
                # Random contrast between 0.2 and 1.0
                contrast = random.uniform(0.2, 1.0)
                # Random brightness between 0.1 and 0.9
                brightness = random.uniform(0.1, 0.9)
                dot_id = dot_id_counter
                dot_id_counter += 1
                dots.append([dot_id, 0, y, velocity, contrast, brightness])
                print(f"Frame {frame}: New dot {dot_id} created at y={y} with velocity={velocity}, contrast={contrast}, and brightness={brightness}")
            
            # Update and draw existing dots
            dots_to_remove = []
            completed_traversals_this_frame = 0
            
            for i, dot in enumerate(dots):
                dot_id, x, y, velocity, contrast, brightness = dot
                
                # Add slight vertical irregularity to the motion
                y += random.uniform(-vertical_irregularity, vertical_irregularity)
                # Maintain consistent horizontal velocity
                x += velocity
                
                # Keep y within bounds
                y = max(dot_size, min(height - dot_size, y))
                
                # Draw dot if it's still in frame
                if x < width:
                    # Apply contrast and brightness to dot color
                    adjusted_color = tuple(int(c * contrast * brightness) for c in dot_color)
                    cv2.circle(frame_img, (int(x), int(y)), dot_size, adjusted_color, -1)
                    dots[i] = [dot_id, x, y, velocity, contrast, brightness]
                    current_dots[dot_id] = x
                    print(f"Frame {frame}: Drawing dot {dot_id} at ({int(x)}, {int(y)}) with contrast={contrast} and brightness={brightness}")
                else:
                    dots_to_remove.append(i)
                    completed_traversals_this_frame += 1
                    print(f"Frame {frame}: Dot {dot_id} exited at x={x}")
            
            # Calculate total x-displacement
            total_displacement = 0
            
            # Handle dots that were in previous frame
            for dot_id, prev_x in prev_dots.items():
                if dot_id in current_dots:
                    # Dot continued through frame
                    displacement = current_dots[dot_id] - prev_x
                    total_displacement += displacement
                else:
                    # Dot exited frame
                    displacement = (width + 1) - prev_x
                    total_displacement += displacement
            
            # Handle new dots that entered this frame
            for dot_id, curr_x in current_dots.items():
                if dot_id not in prev_dots:
                    # New dot entered frame
                    displacement = curr_x - (-1)
                    total_displacement += displacement
            
            # Store frame pair information
            frame_pairs.append({
                'frame1': frame - 1,  # Previous frame
                'frame2': frame,      # Current frame
                'dots_frame1': len(prev_dots),
                'dots_frame2': len(current_dots),
                'total_displacement': np.round(total_displacement, decimals=1),  # Round to nearest 0.1 using numpy
                'completed_traversals': total_completed_traversals
            })
            
            # Update previous frame's dots
            prev_dots = current_dots.copy()
            
            # Remove dots that have exited the frame
            for i in sorted(dots_to_remove, reverse=True):
                dots.pop(i)
                total_completed_traversals += 1  # Increment counter when dot is removed
                print(f"Frame {frame}: Dot completed journey, total completed: {total_completed_traversals}")
            
            # Write frame to video
            out.write(frame_img)
            print(f"Frame {frame}: Active dots: {len(dots)}")
        
        # Release video writer
        out.release()
        print(f"Video saved to {output_path}")
        
        # Write frame pair information to file
        with open(output_path.replace('_X.mp4', '_Y.txt'), 'w') as f:
            for pair in frame_pairs:
                f.write(f"{pair['frame1']} {pair['frame2']} {pair['dots_frame1']} {pair['dots_frame2']} {pair['total_displacement']} {pair['completed_traversals']}\n")
        print(f"Frame pair information saved to {output_path.replace('_X.mp4', '_Y.txt')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up video writer if it exists
        if 'out' in locals() and out is not None:
            out.release()
        # Remove partial video file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def load_data(video_path, label_path):
    """
    Load video frames and corresponding labels for training.

    Parameters:
    -----------
    video_path : str
        Path to the video file.
    label_path : str
        Path to the label file.

    Returns:
    --------
    frame_pairs : numpy.ndarray
        Array of consecutive frame pairs.
    labels : numpy.ndarray
        Array of total displacement labels.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)

    # Create pairs of consecutive frames
    frame_pairs = np.array([frames[i:i+2] for i in range(len(frames)-1)])

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(float(parts[4]))  # Assuming the fifth column is the total displacement
    labels = np.array(labels)

    return frame_pairs, labels

def create_model(input_shape):
    """
    Create a deep learning model for displacement estimation.

    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data.

    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_counting_model(data_path, model_save_path):
    """
    Train a counting model using the provided data.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing the training data.
    model_save_path : str
        Path where the trained model will be saved.
    """
    video_path = os.path.join(data_path, 'drifting_dots_X.mp4')
    label_path = os.path.join(data_path, 'drifting_dots_Y.txt')

    frame_pairs, labels = load_data(video_path, label_path)
    input_shape = frame_pairs[0].shape

    model = create_model(input_shape)
    model.fit(frame_pairs, labels, epochs=10, batch_size=32, validation_split=0.2)

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Generate video with default parameters
    generate_drifting_dots_video() 