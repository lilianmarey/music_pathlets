"""
Defines time periods for a dataset along with relevant configuration parameters.
"""

from processing.helpers import datetime_to_timestamp

XXXX_time_periods = dict(
    [
        (0, (datetime_to_timestamp(1, 1, 2022), datetime_to_timestamp(1, 2, 2022))),
        (1, (datetime_to_timestamp(1, 2, 2022), datetime_to_timestamp(1, 3, 2022))),
        (2, (datetime_to_timestamp(1, 3, 2022), datetime_to_timestamp(1, 4, 2022))),
        (3, (datetime_to_timestamp(1, 4, 2022), datetime_to_timestamp(1, 5, 2022))),
        (4, (datetime_to_timestamp(1, 5, 2022), datetime_to_timestamp(1, 6, 2022))),
        (5, (datetime_to_timestamp(1, 6, 2022), datetime_to_timestamp(1, 7, 2022))),
        (6, (datetime_to_timestamp(1, 7, 2022), datetime_to_timestamp(1, 8, 2022))),
        (7, (datetime_to_timestamp(1, 8, 2022), datetime_to_timestamp(1, 9, 2022))),
        (8, (datetime_to_timestamp(1, 9, 2022), datetime_to_timestamp(1, 10, 2022))),
        (9, (datetime_to_timestamp(1, 10, 2022), datetime_to_timestamp(1, 11, 2022))),
        (10, (datetime_to_timestamp(1, 11, 2022), datetime_to_timestamp(1, 12, 2022))),
        (11, (datetime_to_timestamp(1, 12, 2022), datetime_to_timestamp(1, 1, 2023))),
        (12, (datetime_to_timestamp(1, 1, 2023), datetime_to_timestamp(1, 2, 2023))),
        (13, (datetime_to_timestamp(1, 2, 2023), datetime_to_timestamp(1, 3, 2023))),
        (14, (datetime_to_timestamp(1, 3, 2023), datetime_to_timestamp(1, 4, 2023))),
        (15, (datetime_to_timestamp(1, 4, 2023), datetime_to_timestamp(1, 5, 2023))),
        (16, (datetime_to_timestamp(1, 5, 2023), datetime_to_timestamp(1, 6, 2023))),
    ]
)

XXXX_K = max(XXXX_time_periods.keys())
XXXX_stream_threshold = 15
XXXX_processed_path = "processed_data/XXXX/"
XXXX_results_path = "results/XXXX/"
