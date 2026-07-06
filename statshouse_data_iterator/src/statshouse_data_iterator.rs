use chrono::NaiveDateTime;
use data_multiple_timelines::MultipleTimelinesValue;
use two_states_frame::TwoStatesFrame;

use crate::data_record::DataRecord;

pub struct StatsHouseDataIterator {
    streams: Vec<Box<dyn TwoStatesFrame<Box<dyn DataRecord>>>>,
    has_next: bool,
}

impl StatsHouseDataIterator {
    pub fn new(streams: Vec<Box<dyn TwoStatesFrame<Box<dyn DataRecord>>>>) -> Self {
        Self {
            streams,
            has_next: true,
        }
    }

    fn get_min_next_time(&self) -> Option<NaiveDateTime> {
        let mut result = None;

        for stream in self.streams.iter() {
            if let Some(next_date_ref) = stream.get_next() {
                let next_date = *next_date_ref.get_time();

                if let Some(current_result) = result {
                    if current_result > next_date {
                        result = Some(next_date);
                    }
                } else {
                    result = Some(next_date);
                }
            }
        }

        result
    }

    fn get_value(&self) -> MultipleTimelinesValue {
        let mut result = Vec::new();

        for stream in self.streams.iter() {
            match stream.get_current() {
                Some(data) => {
                    result.push(data.get_value());
                }
                _ => {
                    panic!("Empty stream");
                }
            }
        }

        result
    }

    fn shift(&mut self) -> bool {
        let mut has_shifted = false;

        if let Some(next_time) = self.get_min_next_time() {
            for stream in self.streams.iter_mut() {
                if let Some(stream_next_time) = stream.get_next() {
                    if *(stream_next_time.get_time()) <= next_time {
                        stream.shift();
                        has_shifted = true;
                    }
                }
            }
        }

        has_shifted
    }
}

impl Iterator for StatsHouseDataIterator {
    type Item = MultipleTimelinesValue;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next {
            return None;
        }

        let result = self.get_value();

        let has_shifted = self.shift();

        self.has_next = has_shifted;

        Some(result)
    }
}
