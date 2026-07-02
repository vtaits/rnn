use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use rnn_architecture::DataProvider;

use crate::{split_iterator::SplitIterator, split_iterator_box::SplitIteratorBox};

pub struct StreamSplitterDataProvider<DataType> {
    train_iterator: Rc<RefCell<SplitIterator<DataType>>>,
    test_iterator: Rc<RefCell<SplitIterator<DataType>>>,
}

impl<DataType> StreamSplitterDataProvider<DataType> {
    pub fn new(
        iterator: Box<dyn Iterator<Item = DataType>>,
        test_start_index: usize,
        test_end_index: usize,
    ) -> Self {
        let index1 = Rc::new(RefCell::new(0usize));
        let index2 = Rc::clone(&index1);

        let buffer_train_1 = Rc::new(RefCell::new(VecDeque::new()));
        let buffer_train_2 = Rc::clone(&buffer_train_1);
        let buffer_test_1 = Rc::new(RefCell::new(VecDeque::new()));
        let buffer_test_2 = Rc::clone(&buffer_test_1);

        let iterator1 = Rc::new(RefCell::new(iterator));
        let iterator2 = Rc::clone(&iterator1);

        let train_iterator = SplitIterator::new(
            iterator1,
            index1,
            Box::new(move |index| index <= test_start_index || index > test_end_index),
            buffer_train_1,
            buffer_test_1,
        );

        let test_iterator = SplitIterator::new(
            iterator2,
            index2,
            Box::new(move |index| index > test_start_index && index <= test_end_index),
            buffer_test_2,
            buffer_train_2,
        );

        Self {
            train_iterator: Rc::new(RefCell::new(train_iterator)),
            test_iterator: Rc::new(RefCell::new(test_iterator)),
        }
    }
}

impl<DataType: 'static> DataProvider<DataType> for StreamSplitterDataProvider<DataType> {
    fn iterate_test_data(&self) -> Box<dyn Iterator<Item = DataType>> {
        Box::new(SplitIteratorBox::new(Rc::clone(&self.test_iterator)).into_iter())
    }

    fn iterate_training_data(&self) -> Box<dyn Iterator<Item = DataType>> {
        Box::new(SplitIteratorBox::new(Rc::clone(&self.train_iterator)).into_iter())
    }
}
