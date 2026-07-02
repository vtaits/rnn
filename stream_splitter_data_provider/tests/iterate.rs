use rnn_architecture::DataProvider;
use stream_splitter_data_provider::StreamSplitterDataProvider;

#[test]
fn test_split_stream() {
    let stream_splitter = StreamSplitterDataProvider::new(Box::new(0..20), 5, 10);

    let training_data: Vec<usize> = stream_splitter.iterate_training_data().collect();
    let test_data: Vec<usize> = stream_splitter.iterate_test_data().collect();

    assert_eq!(
        training_data,
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    );
    assert_eq!(test_data, [5, 6, 7, 8, 9]);
}
