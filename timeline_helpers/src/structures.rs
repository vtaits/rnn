#[derive(PartialEq, Debug)]
pub enum ComplexTimelineValue {
    Float(f32),
    Integer(i32),
    Enum(String),
}

pub trait Timeline {
    fn get_bits(&self, value: &ComplexTimelineValue) -> Vec<bool>;

    fn get_capacity(&self) -> &u8;

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue;
}
