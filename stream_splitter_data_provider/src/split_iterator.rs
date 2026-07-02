use std::{cell::RefCell, collections::VecDeque, rc::Rc};

pub struct SplitIterator<DataType> {
    iterator: Rc<RefCell<Box<dyn Iterator<Item = DataType>>>>,
    index: Rc<RefCell<usize>>,
    check_is_own: Box<dyn Fn(usize) -> bool>,
    buffer_own: Rc<RefCell<VecDeque<DataType>>>,
    buffer_opposite: Rc<RefCell<VecDeque<DataType>>>,
}

impl<DataType> SplitIterator<DataType> {
    pub fn new(
        iterator: Rc<RefCell<Box<dyn Iterator<Item = DataType>>>>,
        index: Rc<RefCell<usize>>,
        check_is_own: Box<dyn Fn(usize) -> bool>,
        buffer_own: Rc<RefCell<VecDeque<DataType>>>,
        buffer_opposite: Rc<RefCell<VecDeque<DataType>>>,
    ) -> Self {
        Self {
            iterator,
            index,
            check_is_own,
            buffer_own,
            buffer_opposite,
        }
    }
}

impl<DataType> Iterator for SplitIterator<DataType> {
    type Item = DataType;

    fn next(&mut self) -> Option<Self::Item> {
        match self.buffer_own.borrow_mut().pop_front() {
            Some(item) => Some(item),
            _ => loop {
                match self.iterator.borrow_mut().next() {
                    Some(next_value) => {
                        let mut index = self.index.borrow_mut();
                        *index += 1;

                        match (self.check_is_own)(*index) {
                            true => {
                                return Some(next_value);
                            }
                            false => {
                                self.buffer_opposite.borrow_mut().push_back(next_value);
                            }
                        }
                    }
                    _ => {
                        return None;
                    }
                }
            },
        }
    }
}
