use chrono::NaiveDateTime;
use data_multiple_timelines::MultipleTimelinesValue;

use crate::statshouse_stream::StatsHouseStream;

pub struct StatsHouseDataIterator {
    streams: Vec<Box<dyn StatsHouseStream>>,
    has_next: bool,
}

impl StatsHouseDataIterator {
    pub fn new(streams: Vec<Box<dyn StatsHouseStream>>) -> Self {
        Self {
            streams,
            has_next: true,
        }
    }

    fn get_min_next_time(&self) -> Option<NaiveDateTime> {
        let mut result = None;

        for stream in self.streams.iter() {
            let next_date_option = stream.get_next_time();

            if let Some(next_date_ref) = next_date_option {
                let next_date = *next_date_ref;

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
            let value = stream.retrieve();

            result.push(value);
        }

        result
    }

    fn shift(&mut self) -> bool {
        let mut has_shifted = false;

        if let Some(next_time) = self.get_min_next_time() {
            for stream in self.streams.iter_mut() {
                if let Some(stream_next_time) = stream.get_next_time() {
                    if *stream_next_time <= next_time {
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
