use two_states_frame::TwoStatesFrame;

pub struct TwoStatesFrameIteraror<DataType> {
    current_item: Option<DataType>,
    next_item: Option<DataType>,
    iterator: Box<dyn Iterator<Item = DataType>>,
}

impl<DataType> TwoStatesFrameIteraror<DataType> {
    pub fn new(mut iterator: Box<dyn Iterator<Item = DataType>>) -> Self {
        let current_item = iterator.next();
        let next_item = iterator.next();

        Self {
            current_item,
            next_item,
            iterator,
        }
    }
}

impl<DataType> TwoStatesFrame<DataType> for TwoStatesFrameIteraror<DataType> {
    fn get_current(&self) -> &Option<DataType> {
        &self.current_item
    }

    fn get_next(&self) -> &Option<DataType> {
        &self.next_item
    }

    fn shift(&mut self) {
        let has_next_item = self.next_item.is_some();

        if has_next_item {
            self.current_item = self.next_item.take();
            self.next_item = self.iterator.next();
        }
    }
}
