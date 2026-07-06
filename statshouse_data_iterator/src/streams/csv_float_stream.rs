use std::path::Path;

use chrono::NaiveDateTime;

use crate::data_record::DataRecord;
use crate::data_record_impl::FloatDataRecord;
use crate::streams::csv_stream::csv_stream;

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

pub fn csv_float_stream<P: AsRef<Path>>(
    file_path: P,
    capacity: usize,
    min_value: f32,
    max_value: f32,
) -> impl Iterator<Item = Box<dyn DataRecord>> {
    csv_stream(file_path).map(move |item| {
        Box::new(FloatDataRecord::new(
            NaiveDateTime::parse_from_str(&item.date, DEFAULT_FORMAT).unwrap(),
            item.value,
            capacity,
            min_value,
            max_value,
        )) as Box<dyn DataRecord>
    })
}
