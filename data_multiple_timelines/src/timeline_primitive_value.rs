pub enum TimelinePrimitiveValue {
    Float(f32),
    Datetime(String),
    Integer(i64),
    Enum(String),
    Time(String),
    Weekday(String),
}
