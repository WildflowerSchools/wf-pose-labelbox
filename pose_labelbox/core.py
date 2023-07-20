import pose_labelbox.ffmpeg
import video_io
import honeycomb_io
import pandas as pd
import tqdm
import tqdm.notebook
import datetime
import itertools
import pathlib
import logging

logger = logging.getLogger(__name__)

def download_videos(
    start,
    end,
    environment_id=None,
    environment_name=None,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    names=None,
    video_duration=datetime.timedelta(seconds=10),
    client=None,
    uri=video_io.config.HONEYCOMB_URI,
    token_uri=video_io.config.HONEYCOMB_TOKEN_URI,
    audience=video_io.config.HONEYCOMB_AUDIENCE,
    client_id=video_io.config.HONEYCOMB_CLIENT_ID,
    client_secret=video_io.config.HONEYCOMB_CLIENT_SECRET,
    video_storage_url=video_io.config.VIDEO_STORAGE_URL,
    video_storage_auth_domain=video_io.config.VIDEO_STORAGE_AUTH_DOMAIN,
    video_storage_audience=video_io.config.VIDEO_STORAGE_AUDIENCE,
    video_storage_client_id=video_io.config.VIDEO_STORAGE_CLIENT_ID,
    video_storage_client_secret=video_io.config.VIDEO_STORAGE_CLIENT_SECRET,
    video_client: video_io.client.VideoStorageClient=None,
    local_video_directory="/data/videos",
    video_filename_extension=None,
    max_workers=video_io.config.MAX_DOWNLOAD_WORKERS,
    overwrite: bool = False,
):
    target_camera_ids = generate_target_camera_ids(
        start=start,
        end=end,
        environment_id=environment_id,
        environment_name=environment_name,
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        names=names,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,      
    )
    target_video_starts = generate_target_video_starts(
        start=start,
        end=end,
        video_duration=video_duration,
    )
    target_videos = list(itertools.product(target_camera_ids, target_video_starts))
    logger.info(f'Searching video service for {len(target_videos)} target videos')
    video_metadata = video_io.fetch_video_metadata(
        start=start,
        end=end,
        video_timestamps=None,
        camera_assignment_ids=None,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_types=None,
        camera_device_ids=device_ids,
        camera_part_numbers=part_numbers,
        camera_names=names,
        camera_serial_numbers=serial_numbers,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        video_storage_url=video_storage_url,
        video_storage_auth_domain=video_storage_auth_domain,
        video_storage_audience=video_storage_audience,
        video_storage_client_id=video_storage_client_id,
        video_storage_client_secret=video_storage_client_secret,
        video_client=video_client,
    )
    found_videos = list()
    for video_metadatum in video_metadata:
        found_videos.append((
            video_metadatum['device_id'],
            convert_to_datetime_utc(video_metadatum['video_timestamp'])
        ))
    logger.info(f'{len(found_videos)} videos found in video service')
    missing_videos = set(target_videos).difference(found_videos)
    if len(missing_videos) > 0:
        raise ValueError(f'{len(missing_videos)} videos not found: {missing_videos}')
    extra_videos = set(found_videos).difference(target_videos)
    if len(extra_videos) > 0:
        raise ValueError(f'{len(extra_videos)} videos found that are not in target video set: {extra_videos}')
    logger.info(f'Downloading {len(video_metadata)} videos')
    video_metadata_with_local_paths = video_io.download_video_files(
        video_metadata=video_metadata,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension,
        max_workers=max_workers,
        video_storage_url=video_storage_url,
        video_storage_auth_domain=video_storage_auth_domain,
        video_storage_audience=video_storage_audience,
        video_storage_client_id=video_storage_client_id,
        video_storage_client_secret=video_storage_client_secret,
        overwrite=overwrite,
        video_client=video_client,
    )
    video_metadata_df = (
        pd.DataFrame(video_metadata_with_local_paths)
        .rename(columns={
            'video_timestamp': 'video_start',
            'device_id': 'camera_id',
        })
        .reindex(columns=[
            'data_id',
            'environment_id',
            'camera_id',
            'video_start',
            'fps',
            'duration_seconds',
            'frame_offsets',
            'video_local_path',            
        ])
        .sort_values([
            'environment_id',
            'camera_id',
            'video_start',
        ])
        .set_index('data_id')
    )
    return video_metadata_df

def extract_frames(
    start,
    end,
    environment_id=None,
    environment_name=None,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    names=None,
    video_duration=datetime.timedelta(seconds=10),
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    frames_per_video=100,
    frames_per_second=10,
    local_video_directory="/data/videos",
    local_frames_directory="/data/frames",
    video_filename_extension='mp4',
    frame_filename_extension='png',
    overwrite=False,
    progress_bar=False,
    notebook=False,
):
    target_camera_ids = generate_target_camera_ids(
        start=start,
        end=end,
        environment_id=environment_id,
        environment_name=environment_name,
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        names=names,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,      
    )
    target_video_starts = generate_target_video_starts(
        start=start,
        end=end,
        video_duration=video_duration,
    )
    environment_id = honeycomb_io.fetch_environment_id(
        environment_id=environment_id,
        environment_name=environment_name,
    )
    if progress_bar:
        if notebook:
            video_iterator = tqdm.notebook.tqdm(list(itertools.product(target_camera_ids, target_video_starts)))
        else:
            video_iterator = tqdm.tqdm(list(itertools.product(target_camera_ids, target_video_starts)))
    else:
        video_iterator = list(itertools.product(target_camera_ids, target_video_starts))
    for camera_id, video_start in video_iterator:
        video_path = generate_video_path(
            environment_id=environment_id,
            camera_id=camera_id,
            video_start=video_start,
            local_video_directory=local_video_directory,
            video_filename_extension=video_filename_extension,
        )
        frame_directory_path = generate_frame_directory_path(
            environment_id=environment_id,
            camera_id=camera_id,
            video_start=video_start,
            local_frames_directory=local_frames_directory,
        )
        frame_filenames = generate_frame_filenames(
            environment_id=environment_id,
            camera_id=camera_id,
            video_start=video_start,
            frames_per_video=frames_per_video,
            frame_filename_extension=frame_filename_extension,
        )
        if frame_directory_path.is_dir():
            existing_filenames = {path.name for path in frame_directory_path.iterdir()}
            if set(frame_filenames).issubset(existing_filenames) and not overwrite:
                logger.info(f'Frames for {video_path} already extracted.')
                continue
        ffmpeg_frame_identifier = generate_ffmpeg_frame_identifier(
            environment_id=environment_id,
            camera_id=camera_id,
            video_start=video_start,
        )
        pose_labelbox.ffmpeg.extract_video_frames(
            video_path=video_path,
            frame_directory_path=frame_directory_path,
            ffmpeg_frame_identifier=ffmpeg_frame_identifier,
            frames_per_second=frames_per_second,
        )


def generate_target_video_starts(
    start,
    end,
    video_duration=datetime.timedelta(seconds=10),
):
    start_utc = convert_to_datetime_utc(start)
    end_utc = convert_to_datetime_utc(end)
    first_video_start_utc = pd.Timestamp(start_utc).floor(video_duration)
    last_video_start_utc = pd.Timestamp(end_utc).ceil(video_duration) - video_duration
    target_video_starts = (
        pd.date_range(
            start=first_video_start_utc,
            end=last_video_start_utc,
            freq=video_duration
        )
        .to_pydatetime()
        .tolist()
    )
    logger.info(f'{len(target_video_starts)} video start times are consistent with the specified start and end times')
    return target_video_starts

def generate_target_camera_ids(
    start,
    end,
    environment_id=None,
    environment_name=None,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    names=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
):
    if environment_id is None and environment_name is None:
        raise ValueError('Must specify either environment ID or environment name')
    target_camera_ids = honeycomb_io.fetch_device_ids(
        device_types=honeycomb_io.DEFAULT_CAMERA_DEVICE_TYPES,
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        tag_ids=None,
        names=names,
        environment_id=environment_id,
        environment_name=environment_name,
        start=start,
        end=end,
        chunk_size=100,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
    )
    logger.info(f'{len(target_camera_ids)} cameras are consistent with the specified search criteria')
    return target_camera_ids

def generate_video_path(
    environment_id,
    camera_id,
    video_start,
    local_video_directory='/data/videos',
    video_filename_extension='mp4',
):
    video_path = (
        pathlib.Path(local_video_directory) /
        environment_id /
        camera_id /
        f'{video_start.year}' /
        f'{video_start.month:02d}' /
        f'{video_start.day:02d}' /
        f'{video_start.hour:02d}' /
        f'{video_start.minute:02d}-{video_start.second:02d}.{video_filename_extension}'
    )
    return video_path

def generate_frame_directory_path(
    environment_id,
    camera_id,
    video_start,
    local_frames_directory='/data/frames'
):
    frame_directory_path = (
        pathlib.Path(local_frames_directory) /
        environment_id /
        camera_id /
        f'{video_start.year}' /
        f'{video_start.month:02d}' /
        f'{video_start.day:02d}' /
        f'{video_start.hour:02d}' /
        f'{video_start.minute:02d}-{video_start.second:02d}'
    )
    return frame_directory_path

def generate_frame_filenames(
    environment_id,
    camera_id,
    video_start,
    frames_per_video=100,
    frame_filename_extension='png',
):
    frame_filenames = list()
    for frame_index in range(1, frames_per_video + 1):
        frame_filename = generate_frame_filename(
            environment_id=environment_id,
            camera_id=camera_id,
            video_start=video_start,
            frame_index=frame_index,
            frame_filename_extension='png',
        )
        frame_filenames.append(frame_filename)
    return frame_filenames

def generate_frame_filename(
    environment_id,
    camera_id,
    video_start,
    frame_index,
    frame_filename_extension='png',
):
    frame_filename = '{}_{}_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}_{:03d}.{}'.format(
        environment_id,
        camera_id,
        video_start.year,
        video_start.month,
        video_start.day,
        video_start.hour,
        video_start.minute,
        video_start.second,
        frame_index,
        frame_filename_extension,
    )
    return frame_filename

def generate_ffmpeg_frame_identifier(
    environment_id,
    camera_id,
    video_start,
    frame_filename_extension='png',
):
    ffmpeg_frame_identifier = '{}_{}_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}_%03d.{}'.format(
        environment_id,
        camera_id,
        video_start.year,
        video_start.month,
        video_start.day,
        video_start.hour,
        video_start.minute,
        video_start.second,
        frame_filename_extension,
    )
    return ffmpeg_frame_identifier

def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
