use crate::{structures::TimelineConfig, DatetimeTimeline, EnumTimeline, FloatTimeline, IntegerTimeline, Timeline};

pub fn init_timeline_by_config(config: &TimelineConfig) -> Box<dyn Timeline> {
    match config {
        TimelineConfig::Datetime(payload) => Box::new(DatetimeTimeline::from_config(payload)),
        TimelineConfig::Float(payload) => Box::new(FloatTimeline::from_config(payload)),
        TimelineConfig::Integer(payload) => Box::new(IntegerTimeline::from_config(payload)),
        TimelineConfig::Enum(payload) => Box::new(EnumTimeline::from_config(payload)),
    }
}
