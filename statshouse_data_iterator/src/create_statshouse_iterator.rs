use two_states_frame::TwoStatesFrame;
use two_states_frame_iterator::TwoStatesFrameIteraror;

use crate::{
    data_record::DataRecord,
    statshouse_data_iterator::StatsHouseDataIterator,
    streams::{csv_float_stream, csv_time_stream, csv_weekday_stream},
};

pub enum StatshouseStream {
    Float(String, usize, f32, f32),
    Time(String, usize),
    Weekday(String),
}

pub fn create_statshouse_iterator(streams_config: Vec<StatshouseStream>) -> StatsHouseDataIterator {
    let streams: Vec<Box<dyn TwoStatesFrame<Box<dyn DataRecord>>>> = streams_config
        .into_iter()
        .map(|stream| {
            Box::new(match stream {
                StatshouseStream::Float(file_path, capacity, min_value, max_value) => {
                    TwoStatesFrameIteraror::new(Box::new(csv_float_stream(
                        file_path, capacity, min_value, max_value,
                    )))
                }
                StatshouseStream::Time(file_path, capacity) => {
                    TwoStatesFrameIteraror::new(Box::new(csv_time_stream(file_path, capacity)))
                }
                StatshouseStream::Weekday(file_path) => {
                    TwoStatesFrameIteraror::new(Box::new(csv_weekday_stream(file_path)))
                }
            }) as Box<dyn TwoStatesFrame<Box<dyn DataRecord>>>
        })
        .collect();

    StatsHouseDataIterator::new(streams)
}
