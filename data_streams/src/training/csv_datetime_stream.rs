extern crate chrono;

use chrono::{NaiveDateTime, ParseResult};
use csv::{Error, Reader};
use serde_derive::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use timeline_helpers::ComplexTimelineValue;

use super::structures::TrainingStream;

#[derive(Debug, Deserialize)]
struct Record {
    Date: String,
}

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

fn parse_date(date_str: &str) -> ParseResult<NaiveDateTime> {
    NaiveDateTime::parse_from_str(date_str, DEFAULT_FORMAT)
}

pub struct CsvDateTimeStream {
    current_date: Option<NaiveDateTime>,
    next_date: Option<NaiveDateTime>,
    reader: Reader<BufReader<File>>,
}

#[derive(Deserialize)]
pub struct CsvDateTimeStreamConfig {
    path: String,
}

impl CsvDateTimeStream {
    fn get_date_format(&self) -> &str {
        DEFAULT_FORMAT
    }

    pub fn from_config(config: &CsvDateTimeStreamConfig) -> Self {
        CsvDateTimeStream::new(&config.path).unwrap()
    }

    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<CsvDateTimeStream, Error> {
        let file = File::open(file_path)?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b';') // Устанавливаем разделитель на ';'
            .from_reader(BufReader::new(file));

        let first = reader.deserialize().next();
        let second = reader.deserialize().next();

        if let Some(result) = first {
            let record: Record = result?;

            let current_date_result = parse_date(&record.Date);

            if let Ok(current_date) = current_date_result {
                if let Some(second_result) = second {
                    let second_record: Record = second_result?;
                    let second_date_result = parse_date(&second_record.Date);

                    if let Ok(second_date) = second_date_result {
                        return Ok(CsvDateTimeStream {
                            current_date: Some(current_date),
                            next_date: Some(second_date),
                            reader,
                        });
                    }
                }

                return Ok(CsvDateTimeStream {
                    current_date: Some(current_date),
                    next_date: None,
                    reader,
                });
            }

            return Ok(CsvDateTimeStream {
                current_date: None,
                next_date: None,
                reader,
            });
        }

        Ok(CsvDateTimeStream {
            current_date: None,
            next_date: None,
            reader,
        })
    }

    fn step(&mut self) {
        if let Some(next_date) = self.next_date {
            self.current_date = Some(next_date);

            let next = self.reader.deserialize().next();

            if let Some(second_result) = next {
                if let Ok(next_record) = second_result {
                    let next_record: Record = next_record;
                    let next_date_result = parse_date(&next_record.Date);

                    if let Ok(next_date) = next_date_result {
                        self.next_date = Some(next_date);
                    } else {
                        self.next_date = None;
                    }
                }
            } else {
                self.next_date = None;
            }
        }
    }

    fn is_date_in_interval(&self, date: NaiveDateTime) -> bool {
        match self.next_date {
            Some(next_date) => next_date > date,
            None => true,
        }
    }
}

impl TrainingStream for CsvDateTimeStream {
    fn get_value(&self) -> ComplexTimelineValue {
        let format = self.get_date_format();

        let datetime = self.current_date.unwrap();

        ComplexTimelineValue::Datetime(format!("{}", datetime.format(format)))
    }

    fn get_date(&self) -> &Option<NaiveDateTime> {
        &self.current_date
    }

    fn get_next_date(&self) -> &Option<NaiveDateTime> {
        &self.next_date
    }

    fn set_date(&mut self, date: NaiveDateTime) {
        if self.is_finish() {
            return;
        }

        while !self.is_date_in_interval(date) {
            self.step();
        }
    }

    fn is_finish(&self) -> bool {
        self.next_date.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_file() -> Result<(), Error> {
        let mut csv_datetime_stream = CsvDateTimeStream::new("./src/training/csv_stream_test.csv")?;

        assert!(!csv_datetime_stream.is_finish());

        match csv_datetime_stream.get_date() {
            Some(date) => {
                assert_eq!(
                    format!("{}", date.format(DEFAULT_FORMAT)),
                    "2024-05-04 23:00:00"
                );
            }
            _ => {
                panic!("Date is not setted type");
            }
        }

        match csv_datetime_stream.get_next_date() {
            Some(date) => {
                assert_eq!(
                    format!("{}", date.format(DEFAULT_FORMAT)),
                    "2024-05-05 00:00:00"
                );
            }
            _ => {
                panic!("Date is not setted type");
            }
        }

        match csv_datetime_stream.get_value() {
            ComplexTimelineValue::Datetime(value) => {
                assert_eq!(value, "2024-05-04 23:00:00");
            }
            _ => {
                panic!("Wrong result type");
            }
        }

        csv_datetime_stream.set_date(parse_date("2024-05-04 23:30:00").unwrap());

        assert!(!csv_datetime_stream.is_finish());

        match csv_datetime_stream.get_value() {
            ComplexTimelineValue::Datetime(value) => {
                assert_eq!(value, "2024-05-04 23:30:00");
            }
            _ => {
                panic!("Wrong result type");
            }
        }

        csv_datetime_stream.set_date(parse_date("2024-05-05 00:00:00").unwrap());

        assert!(!csv_datetime_stream.is_finish());

        match csv_datetime_stream.get_value() {
            ComplexTimelineValue::Datetime(value) => {
                assert_eq!(value, "2024-05-05 00:00:00");
            }
            _ => {
                panic!("Wrong result type");
            }
        }

        csv_datetime_stream.set_date(parse_date("2024-05-05 01:00:00").unwrap());

        assert!(csv_datetime_stream.is_finish());

        match csv_datetime_stream.get_value() {
            ComplexTimelineValue::Datetime(value) => {
                assert_eq!(value, "2024-05-05 01:00:00");
            }
            _ => {
                panic!("Wrong result type");
            }
        }

        assert!(csv_datetime_stream.get_next_date().is_none());

        Ok(())
    }
}
