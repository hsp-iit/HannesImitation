import os
import numpy as np
import cv2 
import zarr

def filter_demonstrations(demonstration_dir):
    """
    Filters bad demonstrations from demonstration_dir (e.g., those not correctly saved)
    returns demonstration_names: dict(list) with retained and discarded file names.
    """
    demonstration_names = os.listdir(demonstration_dir)

    retained_demonstration_names = []
    discarded_demonstration_names = []

    for demo_name in demonstration_names:
        try:
            store = zarr.open(demonstration_dir + demo_name)
        
            # it raises an exception if 'time' is not in the store
            # generally it means that the saved demonstration had some problems
            if len(store['time']) == 0 or len(list(store.group_keys())) == 0:
                raise Exception
            retained_demonstration_names.append(demo_name)
        except Exception:
            discarded_demonstration_names.append(demo_name)

    demonstration_names = {'retained': retained_demonstration_names, 
                           'discarded': discarded_demonstration_names}

    return demonstration_names

def index_alignment(x_collection):
    """
    Performs a temporal alignment of different data sources (e.g., hand, camera) following a nearest time-stamp approach.
    
    arg:
    - x_collection: dict(list) holds the time-stamps of different time series data.

    returns:
    - selected_timestamp_indeced: dict(list) holding for each timeseries the indeces to select to make the data sources aligned.
    """
    # find the time series with the least amount of samples (the source with lowest sampling frequency)
    series_lenths = [len(x_collection[key]) for key in x_collection]
    slower_series_idx = np.argmin(series_lenths)
    slowe_series_len = np.min(series_lenths)
    slower_series_name = list(x_collection.keys())[slower_series_idx]
    slower_series = x_collection[slower_series_name]

    # remove slower time series from the processed ones
    iterating_keys = list(x_collection.keys())
    iterating_keys.remove(slower_series_name)

    # save required indeces in this dictionary
    selected_timestamp_indeces = {key: [] for key in iterating_keys}

    for timestamp in slower_series:
        for key in iterating_keys:
            preprocessing_series = x_collection[key]
            time_diffs = np.abs(timestamp - preprocessing_series)
            
            closest_idx = np.argmin(time_diffs)
            selected_timestamp_indeces[key].append(closest_idx)     

    # for the slower series, just add all indeces
    selected_timestamp_indeces[slower_series_name] = np.arange(slowe_series_len)

    return selected_timestamp_indeces


def resize_image(img, scaling_factor):
    """
    Resizes an image by a scaling factor.
    If scaling_factor > 1, the image is upsampled;
    If scaling_facror < 1, the image is subsampled.
    """
    height, width, channels = img.shape

    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)

    interpolation = cv2.INTER_LINEAR if scaling_factor > 1 else cv2.INTER_AREA 

    new_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    return new_img


def store_merged_data_to_zarr(merged_data, path):
    """
    store dictionary containing all episodes into a store. It assumes the dictionary is structed as
    
    merged_data = {
        'data': 
            {'timeseries_1': list,
             ... 
             'timeseries_N': list}, 
        'meta': {'episode_ends': list}}
    
    """
    assert('data' in merged_data)
    assert('meta' in merged_data)
    assert('episode_ends' in merged_data['meta'])

    store = zarr.open(path, mode='w')
    store.create_groups('data', 'meta')
    
    # store all data
    for key, data in merged_data['data'].items():
        store['data'][key] = data

    # store episode ends
    store['meta']['episode_ends'] = merged_data['meta']['episode_ends']

    return store