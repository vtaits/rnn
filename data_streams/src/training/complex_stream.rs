use chrono::NaiveDateTime;
use timeline_helpers::ComplexTimelineValue;

use super::TrainingStream;

pub struct ComplexStream {
    streams: Vec<Box<dyn TrainingStream>>,
}

impl ComplexStream {
    pub fn new(streams: Vec<Box<dyn TrainingStream>>) -> Self {
        ComplexStream { streams }
    }

    pub fn get_value(&self) -> Vec<ComplexTimelineValue> {
        let mut result = Vec::new();

        for stream in self.streams.iter() {
            let value = stream.get_value();

            result.push(value)
        }

        result
    }

    pub fn is_finish(&self) -> bool {
        for stream in self.streams.iter() {
            if !stream.is_finish() {
                return false;
            }
        }

        true
    }

    fn get_min_next_date(&self) -> Option<NaiveDateTime> {
        let mut result = None;

        for stream in self.streams.iter() {
            let next_date_option = stream.get_next_date();

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

    pub fn step(&mut self) {
        let next_date_option = self.get_min_next_date();

        if let Some(next_date) = next_date_option {
            for stream in self.streams.iter_mut() {
                stream.set_date(next_date);
            }
        }
    }
}
