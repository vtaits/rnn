use rnn_architecture::ForwardTransformer;

use crate::multiple_timelines_value::MultipleTimelinesValue;

pub struct MultipleTimelinesForwardTransformer();

impl MultipleTimelinesForwardTransformer {
    pub fn new() -> Self {
        Self()
    }
}

impl ForwardTransformer<MultipleTimelinesValue> for MultipleTimelinesForwardTransformer {
    fn transform(&self, data: MultipleTimelinesValue) -> Vec<bool> {
        data.iter()
            .flat_map(|timeline_value| timeline_value.to_binary())
            .collect()
    }
}
