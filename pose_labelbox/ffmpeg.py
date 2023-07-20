import ffmpeg
import pathlib


def extract_video_frames(
    video_path,
    frame_directory_path,
    ffmpeg_frame_identifier,
    frames_per_second=10,
):
    if not pathlib.Path(video_path).is_file():
        raise ValueError(f'Video file {video_path} does not exist')
    pathlib.Path(frame_directory_path).mkdir(parents=True, exist_ok=True)
    frames_input_argument = str(video_path)
    frames_output_argument = str(frame_directory_path / ffmpeg_frame_identifier)
    stdout, stderr = (
        ffmpeg
        .input(frames_input_argument)
        .output(frames_output_argument, r=frames_per_second)
        .run(quiet=True)
    )