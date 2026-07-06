use std::path::Path;

use chrono::NaiveDateTime;

use crate::data_record::DataRecord;
use crate::data_record_impl::TimeDataRecord;
use crate::streams::csv_stream::csv_stream;

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

pub fn csv_time_stream<P: AsRef<Path>>(
    file_path: P,
    capacity: usize,
) -> impl Iterator<Item = Box<dyn DataRecord>> {
    csv_stream(file_path).map(move |item| {
        Box::new(TimeDataRecord::new(
            NaiveDateTime::parse_from_str(&item.date, DEFAULT_FORMAT).unwrap(),
            capacity,
        )) as Box<dyn DataRecord>
    })
}
