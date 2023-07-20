import video_io
import honeycomb_io
import pandas as pd
import datetime
import itertools
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
        camera_device_ids=None,
        camera_part_numbers=None,
        camera_names=None,
        camera_serial_numbers=None,
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
    return video_metadata_with_local_paths


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

def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
