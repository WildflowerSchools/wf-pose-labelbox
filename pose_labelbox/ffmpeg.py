import ffmpeg
import subprocess
import datetime
import re
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
    frame_period=datetime.timedelta(milliseconds=100),
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
            image_paths = sorted(pose_track_directory_path.glob(f'*.{overlay_image_extension}'))
            pose_track_start = extract_bounding_box_overlay_timestamp(image_paths[0].stem)
            pose_track_end = extract_bounding_box_overlay_timestamp(image_paths[-1].stem) + frame_period
            output_path = generate_bounding_box_overlay_video_path(
                inference_id=inference_id,
                camera_id=camera_id,
                pose_track_label=pose_track_label,
                pose_track_start=pose_track_start,
                pose_track_end=pose_track_end,
                bounding_box_overlay_video_parent_directory=bounding_box_overlay_video_parent_directory,
                overlay_video_extension=overlay_video_extension,
            )
            if output_path.is_file():
                logger.info(f'Bounding box overlay video {output_path} already exists. Skipping')
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_list_path = pose_track_directory_path / 'image_list.txt'
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
            
def generate_bounding_box_overlay_video_path(
    inference_id,
    camera_id,
    pose_track_label,
    pose_track_start,
    pose_track_end,
    bounding_box_overlay_video_parent_directory='/data/bounding_box_overlay_videos',
    overlay_video_extension='mp4',
):
    pose_track_start_string = pose_track_start.strftime('%Y%m%d_%H%M%S_%f')
    pose_track_end_string = pose_track_end.strftime('%Y%m%d_%H%M%S_%f')
    generate_bounding_box_overlay_video_path = (
        pathlib.Path(bounding_box_overlay_video_parent_directory) /
        inference_id /
        camera_id /
        f'{pose_track_label}_{pose_track_start_string}_{pose_track_end_string}.{overlay_video_extension}'
    )
    return generate_bounding_box_overlay_video_path

bounding_box_overlay_filename_re = re.compile(r'pose_track_overlay_(?P<year_string>[0-9]{4})(?P<month_string>[0-9]{2})(?P<day_string>[0-9]{2})_(?P<hour_string>[0-9]{2})(?P<minute_string>[0-9]{2})(?P<second_string>[0-9]{2})_(?P<microsecond_string>[0-9]{6})')
def extract_bounding_box_overlay_timestamp(filename_stem):
    m = bounding_box_overlay_filename_re.match(filename_stem)
    if not m:
        raise ValueError(f'Failed to parse filename \'{filename_stem}\'')
    timestamp = datetime.datetime(
        year=int(m.group('year_string')),
        month=int(m.group('month_string')),
        day=int(m.group('day_string')),
        hour=int(m.group('hour_string')),
        minute=int(m.group('minute_string')),
        second=int(m.group('second_string')),
        microsecond=int(m.group('microsecond_string')),
        tzinfo=datetime.timezone.utc
    )
    return timestamp
