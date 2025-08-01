import cv2
import numpy as np
import random
from tqdm import tqdm

def generate_drifting_dots_video(
    output_file_base_name='drifting_dots',
    frame_size=(128, 128),
    num_frames=1024,
    fps=24,
    entry_probability=0.05,
    dot_size=(2, 5),
    dot_color=((64, 255), (64, 255), (64, 255)),
    dot_velocity=(1.0, 4.0),
    vertical_irregularity=0.3,
    seed=42  # default seed for reproducibility
):
    random.seed(seed)
    np.random.seed(seed)

    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_X = cv2.VideoWriter(f'{output_file_base_name}_X.mp4', fourcc, fps, (width, height))

    class Dot:
        def __init__(self):
            self.x = 0
            self.y = random.uniform(0, height)
            self.size = random.randint(dot_size[0], dot_size[1])
            self.color = (
                random.randint(dot_color[0][0], dot_color[0][1]),
                random.randint(dot_color[1][0], dot_color[1][1]),
                random.randint(dot_color[2][0], dot_color[2][1]),
            )
            self.velocity = random.uniform(dot_velocity[0], dot_velocity[1])

        def update(self):
            self.x += self.velocity
            self.y += random.uniform(-vertical_irregularity, vertical_irregularity)

        def is_out(self):
            return self.x > width

    dots = []
    cross_count = 0
    previous_frame_dots = []
    y_data = []

    for t in tqdm(range(num_frames), desc="Generating video frames"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if random.random() < entry_probability:
            dots.append(Dot())

        current_frame_dots = []

        for dot in dots:
            dot.update()
            if 0 <= int(dot.x) < width and 0 <= int(dot.y) < height:
                cv2.circle(frame, (int(dot.x), int(dot.y)), dot.size, dot.color, -1)
                current_frame_dots.append(dot.x)
            elif dot.is_out():
                cross_count += 1

        dots = [d for d in dots if not d.is_out()]
        out_X.write(frame)

        if t > 0:
            prev_count = len(previous_frame_dots)
            curr_count = len(current_frame_dots)
            displacement = sum([curr - prev for curr, prev in zip(current_frame_dots, previous_frame_dots[:len(current_frame_dots)])])
            y_data.append((t - 1, t, prev_count, curr_count, round(displacement, 2), cross_count))

        previous_frame_dots = current_frame_dots

    out_X.release()

    with open(f'{output_file_base_name}_Y.txt', 'w') as f:
        for row in y_data:
            f.write(' '.join(map(str, row)) + '\n')

generate_drifting_dots_video()