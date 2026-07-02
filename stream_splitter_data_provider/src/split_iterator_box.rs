use std::{cell::RefCell, rc::Rc};

use crate::split_iterator::SplitIterator;

pub struct SplitIteratorBox<DataType> {
    iterator: Rc<RefCell<SplitIterator<DataType>>>,
}

impl<DataType> SplitIteratorBox<DataType> {
    pub fn new(iterator: Rc<RefCell<SplitIterator<DataType>>>) -> Self {
        Self { iterator }
    }
}

impl<DataType> Iterator for SplitIteratorBox<DataType> {
    type Item = DataType;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.borrow_mut().next()
    }
}
