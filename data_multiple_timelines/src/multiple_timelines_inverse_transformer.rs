use rnn_architecture::InverseTransformer;

use crate::{
    multiple_timelines_value::MultipleTimelinesValue,
    timeline_value_retriever::TimelineValueRetriever,
};

pub struct MultipleTimelinesInverseTransformer {
    retrievers: Vec<Box<dyn TimelineValueRetriever>>,
}

impl MultipleTimelinesInverseTransformer {
    pub fn new(retrievers: Vec<Box<dyn TimelineValueRetriever>>) -> Self {
        Self { retrievers }
    }
}

impl InverseTransformer<MultipleTimelinesValue> for MultipleTimelinesInverseTransformer {
    fn transform(&self, signal: Vec<bool>) -> MultipleTimelinesValue {
        let mut res = vec![];
        let mut offset = 0;

        for retriever in &self.retrievers {
            let capacity = retriever.get_capacity();

            let next_offset = offset + capacity;

            let signal_slice = &signal[offset..next_offset];

            let res_item = retriever.retrieve(signal_slice);

            res.push(res_item);

            offset = next_offset;
        }

        res
    }
}
