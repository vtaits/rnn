pub trait TwoStatesFrame<DataType> {
    fn get_current(&self) -> &Option<DataType>;
    fn get_next(&self) -> &Option<DataType>;
    fn shift(&mut self);
}
