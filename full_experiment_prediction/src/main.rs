use binary_controller_builder::BinaryControllerBuilder;
use cross_validation::CrossValidation;
use data_multiple_timelines::{
    FloatRetriever, MultipleTimelinesForwardTransformer, MultipleTimelinesInverseTransformer,
    MultipleTimelinesValue, TimeRetriever, WeekdayRetriever,
};
use experiment_prediction::ExprerimentPrediction;
use forward_transformer_logger::ForwardTransformerLogger;
use inverse_transformer_logger::InverseTransformerLogger;
use rnn_architecture::{FullExperiment, SingleExperiment};
use simple_experiment_result::SimpleExperimentResult;
use simple_high_level_controller::SimpleHighLevelController;
use statshouse_data_iterator::{create_statshouse_iterator, StatshouseStream};
use stream_splitter_data_provider::StreamSplitterDataProvider;

fn main() {
    let cross_validation = CrossValidation::new(
        10,
        Box::new(|index| {
            let mut builder = BinaryControllerBuilder::new();

            builder.set_field_width(9);
            builder.set_field_height(9);
            builder.set_layer_width(2);
            builder.set_layer_height(2);

            let binary_controller = builder.build();

            let data_iterator = create_statshouse_iterator(vec![
                StatshouseStream::Weekday(String::from("../media/csv/cpu_small.csv")),
                StatshouseStream::Time(String::from("../media/csv/cpu_small.csv"), 24),
                StatshouseStream::Float(String::from("../media/csv/cpu_small.csv"), 10, 0.0, 30.0),
                StatshouseStream::Float(
                    String::from("../media/csv/tcp_errors_small.csv"),
                    30,
                    3000000.0,
                    7000000.0,
                ),
                StatshouseStream::Float(
                    String::from("../media/csv/users_online_small.csv"),
                    10,
                    1000000.0,
                    4000000.0,
                ),
            ]);

            let data_provider = StreamSplitterDataProvider::new(
                Box::new(data_iterator.into_iter()),
                index * 72,
                index * 72 + 72,
            );

            let high_level_controller = SimpleHighLevelController::new(
                Box::new(ForwardTransformerLogger::new(Box::new(
                    MultipleTimelinesForwardTransformer::new(),
                ))),
                Box::new(InverseTransformerLogger::new(Box::new(
                    MultipleTimelinesInverseTransformer::new(vec![
                        Box::new(WeekdayRetriever::new()),
                        Box::new(TimeRetriever::new(24)),
                        Box::new(FloatRetriever::new(10, 0.0, 30.0)),
                        Box::new(FloatRetriever::new(30, 3000000.0, 7000000.0)),
                        Box::new(FloatRetriever::new(10, 1000000.0, 4000000.0)),
                    ]),
                ))),
                binary_controller,
            );

            Box::new(ExprerimentPrediction::new(
                Box::new(data_provider),
                Box::new(high_level_controller),
                Box::new(|original_data, generated_data| {
                    Box::new(SimpleExperimentResult::new(original_data, generated_data))
                }),
            )) as Box<dyn SingleExperiment<MultipleTimelinesValue>>
        }),
    );

    let results = cross_validation.execute();

    for result in results {
        for generated_item in result.get_generated_data().into_iter() {
            match generated_item[3].get_primitive_value() {
                data_multiple_timelines::TimelinePrimitiveValue::Float(value) => {
                    println!("{}", value);
                }
                _ => {}
            }
        }
    }
}
