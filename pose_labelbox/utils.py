import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)

def generate_output_period(
    start,
    end,
    video_duration=datetime.timedelta(seconds=10),
):
    start_utc = convert_to_datetime_utc(start)
    end_utc = convert_to_datetime_utc(end)
    output_start = pd.Timestamp(start_utc).floor(video_duration).to_pydatetime()
    output_end = pd.Timestamp(end_utc).ceil(video_duration).to_pydatetime()
    return output_start, output_end

def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
