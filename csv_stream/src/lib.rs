use std::fs::File;
use std::path::Path;
use csv::{Error, Reader};

#[derive(Debug, serde::Deserialize)]
struct Record {
  Date: String,
  Value: f32,
}

pub struct CsvStream {
  rdr: Reader<File>
}

impl CsvStream {
  pub fn new<P: AsRef<Path>>(path: P) -> Result<CsvStream, Error> {
    let rdr = Reader::from_path(path)?;

    Ok(CsvStream {
      rdr
    })
  }
}
