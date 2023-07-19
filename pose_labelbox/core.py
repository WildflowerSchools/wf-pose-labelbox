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


def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
