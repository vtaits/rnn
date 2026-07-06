use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::csv_record::CsvRecord;

pub fn csv_stream<P: AsRef<Path>>(file_path: P) -> impl Iterator<Item = CsvRecord> {
    let file = File::open(file_path).unwrap();
    let reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(BufReader::new(file));

    reader.into_deserialize().map(|item| item.unwrap())
}
