use serde_derive::Deserialize;

#[derive(Debug, Deserialize)]
struct CsvRecord {
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Value")]
    value: f32,
}
