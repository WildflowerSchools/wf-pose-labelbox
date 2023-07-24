import ffmpeg
import subprocess
import pathlib
import logging

logger = logging.getLogger(__name__)


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

def generate_bounding_box_overlay_videos(
    inference_id,
    bounding_box_overlay_parent_directory='/data/bounding_box_overlays',
    overlay_image_extension='png',
    bounding_box_overlay_video_parent_directory='/data/bounding_box_overlay_videos',
    overlay_video_extension='mp4',
    overlay_video_codec='libx264',
    overlay_video_pixel_format='yuv420p',
    frames_per_second=10,
):
    inference_directory_path = (
        pathlib.Path(bounding_box_overlay_parent_directory) /
        inference_id
    )
    for camera_directory_path in inference_directory_path.iterdir():
        camera_id = camera_directory_path.name
        logger.info(f'Generating bounding box overlay videos for camera {camera_id}')
        for pose_track_directory_path in camera_directory_path.iterdir():
            pose_track_label = pose_track_directory_path.name
            image_list_path = pose_track_directory_path / 'image_list.txt'
            output_path = (
                pathlib.Path(bounding_box_overlay_video_parent_directory) /
                inference_id /
                camera_id /
                f'{pose_track_label}.{overlay_video_extension}'
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_file():
                logger.info(f'Bounding box overlay video {output_path} already exists. Skipping')
                continue
            logger.info(f'Generating bounding box overlay video for camera {camera_id} for pose track {pose_track_label}')
            # TODO: Use the python-ffmpeg API instead of subprocess
            arguments = [
                'ffmpeg',
                '-safe',
                '0',
                '-f',
                'concat',
                '-i',
                str(image_list_path),
                '-c:v',
                overlay_video_codec,
                '-r',
                str(frames_per_second),
                '-pix_fmt',
                overlay_video_pixel_format,
                '-y',
                str(output_path),
            ]
            logger.info(f"Executing: {' '.join(arguments)}")
            subprocess.run(arguments)
            
