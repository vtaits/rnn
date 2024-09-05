use serde_derive::Deserialize;

use crate::{
    enum_timeline::EnumTimelineConfig, float_timeline::FloatTimelineConfig,
    integer_timeline::IntegerTimelineConfig,
};

#[derive(PartialEq, Debug, Deserialize)]
pub enum ComplexTimelineValue {
    Float(f32),
    Integer(i32),
    Enum(String),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TimelineConfig {
    Float(FloatTimelineConfig),
    Integer(IntegerTimelineConfig),
    Enum(EnumTimelineConfig),
}

pub trait Timeline: Send + Sync {
    fn get_bits(&self, value: &ComplexTimelineValue) -> Vec<bool>;

    fn get_capacity(&self) -> &u8;

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue;
}
