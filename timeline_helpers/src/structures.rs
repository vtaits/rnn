use rnn_core::{Partition, RegressResult};
use serde_derive::{Deserialize, Serialize};

use crate::{
    enum_timeline::EnumTimelineConfig, float_timeline::FloatTimelineConfig,
    integer_timeline::IntegerTimelineConfig, time_timeline::TimeTimelineConfig,
    DatetimeTimelineConfig, WeekdayTimelineConfig, WeekendTimelineConfig,
};

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ComplexTimelineValue {
    Float(f32),
    Datetime(String),
    Integer(i64),
    Enum(String),
    Time(String),
    Weekday(String),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TimelineConfig {
    Datetime(DatetimeTimelineConfig),
    Float(FloatTimelineConfig),
    Integer(IntegerTimelineConfig),
    Enum(EnumTimelineConfig),
    Time(TimeTimelineConfig),
    Weekday(WeekdayTimelineConfig),
    Weekend(WeekendTimelineConfig),
}

pub trait Timeline: Send + Sync {
    fn is_target(&self) -> bool;

    fn get_bits(&self, value: &ComplexTimelineValue) -> Vec<bool>;

    fn get_capacity(&self) -> &u8;

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue;

    fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool>;

    fn get_partition(&self) -> Partition;

    fn regress(&self, bits: &[bool]) -> f32;

    fn get_regress_difference(
        &self,
        original: &ComplexTimelineValue,
        computed: &ComplexTimelineValue,
    ) -> RegressResult;
}
