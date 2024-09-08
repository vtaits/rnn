use serde_derive::{Deserialize, Serialize};

use crate::{
    enum_timeline::EnumTimelineConfig, float_timeline::FloatTimelineConfig,
    integer_timeline::IntegerTimelineConfig, DatetimeTimelineConfig,
};

#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub enum ComplexTimelineValue {
    Float(f32),
    Datetime(String),
    Integer(i32),
    Enum(String),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TimelineConfig {
    Datetime(DatetimeTimelineConfig),
    Float(FloatTimelineConfig),
    Integer(IntegerTimelineConfig),
    Enum(EnumTimelineConfig),
}

pub trait Timeline: Send + Sync {
    fn get_bits(&self, value: &ComplexTimelineValue) -> Vec<bool>;

    fn get_capacity(&self) -> &u8;

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue;
}
