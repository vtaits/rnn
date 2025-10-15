use crate::{
    structures::TimelineConfig, time_timeline::TimeTimeline, DatetimeTimeline, EnumTimeline,
    FloatTimeline, IntegerTimeline, Timeline, WeekdayTimeline, WeekendTimeline,
};

pub fn init_timeline_by_config(config: &TimelineConfig) -> Box<dyn Timeline> {
    match config {
        TimelineConfig::Datetime(payload) => Box::new(DatetimeTimeline::from_config(payload)),
        TimelineConfig::Float(payload) => Box::new(FloatTimeline::from_config(payload)),
        TimelineConfig::Integer(payload) => Box::new(IntegerTimeline::from_config(payload)),
        TimelineConfig::Enum(payload) => Box::new(EnumTimeline::from_config(payload)),
        TimelineConfig::Time(payload) => Box::new(TimeTimeline::from_config(payload)),
        TimelineConfig::Weekday(payload) => Box::new(WeekdayTimeline::from_config(payload)),
        TimelineConfig::Weekend(payload) => Box::new(WeekendTimeline::from_config(payload)),
    }
}
