use crate::TimelinePrimitiveValue;

pub trait TimelineValue {
    fn to_binary(&self) -> Vec<bool>;

    fn get_primitive_value(&self) -> TimelinePrimitiveValue;
}
