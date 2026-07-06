use serde_derive::Deserialize;

#[derive(Debug, Deserialize)]
pub struct CsvRecord {
    #[serde(rename = "Date")]
    pub date: String,
    #[serde(rename = "Value")]
    pub value: f32,
}
