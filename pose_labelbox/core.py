import honeycomb_io
import pandas as pd
import datetime

def generate_target_video_starts(
    start,
    end,
    video_duration=datetime.timedelta(seconds=10),
    video_duration_string='10S'
):
    start_utc = convert_to_datetime_utc(start)
    end_utc = convert_to_datetime_utc(end)
    first_video_start_utc = pd.Timestamp(start_utc).floor(video_duration)
    last_video_start_utc = pd.Timestamp(end_utc).ceil(video_duration_string) - video_duration
    target_video_starts = (
        pd.date_range(
            start=first_video_start_utc,
            end=last_video_start_utc,
            freq=video_duration
        )
        .to_pydatetime()
        .tolist()
    )
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
    client_secret=None      
):
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
        client_secret=client_secret
    )
    return target_camera_ids

def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
