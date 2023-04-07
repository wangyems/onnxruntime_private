#![forbid(unsafe_code)]

use onnxruntime::{environment::Environment, ndarray::Array, GraphOptimizationLevel, LoggingLevel, OrtNdArray};
use std::{env::var, collections::HashMap};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    // Setup the example's log level.
    // NOTE: ONNX Runtime's log level is controlled separately when building the environment.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning);

    let builder = if let Some(path) = path.clone() {
        builder.with_library_path(path)
    } else {
        builder
    };

    let environment = builder.build().unwrap();

    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_intra_op_num_threads(1)?
        // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8),
        //       _not_ SqueezeNet 1.1 as downloaded by '.with_model_downloaded(ImageClassification::SqueezeNet)'
        //       Obtain it with:
        //          curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
        .with_model_from_file("squeezenet1.0-8.onnx")?;

    let input0_shape: Vec<usize> = session.inputs.get_index(0)
        .unwrap()
        .1
        .shape()
        .map(std::option::Option::unwrap)
        .collect();
    let output0_shape: Vec<usize> = session.outputs.get_index(0)
        .unwrap()
        .1
        .shape()
        .map(std::option::Option::unwrap)
        .collect();

    assert_eq!(input0_shape, [1, 3, 224, 224]);
    assert_eq!(output0_shape, [1, 1000, 1, 1]);

    // Create input arrays in a map from input_name to array.
    // OrtNdArray is a container for an ndarray::Array that will
    // generate an OrtValue upon request.
    // This is generally long-lived, where the array(s) are
    // set with different inputs for each call to session.run()
    let mut input_arrays: HashMap<String, OrtNdArray<_,_>> = session.inputs.iter()
        .map(|(name, tensor_info)| {
            // Get shape of tensor
            let shape: Vec<usize> = tensor_info.tensor_shape.iter()
                .map(|d| d.unwrap() as usize)
                .collect();
            // Compute total size of tensor (for linspace)
            let n: usize = shape.iter().product();
            // Create a 1-D tensor of the right total size, then reshape it.
            let array = Array::linspace(0.0_f32, 1.0, n)
                .into_shape(shape)
                .unwrap();
            // Key/value pair
            (name.clone(), OrtNdArray::new(array))
        })
        .collect();

    // A quick translation of name / OrtNdArray -> name / OrtValue
    // before each call to session.run()
    let inputs = 
        onnxruntime::create_graph_inputs(&session, input_arrays.iter_mut())?;
    
    // Ask session.run() for all outputs
    let output_names: Vec<String> = session.outputs
        .keys()
        .map(|name| name.clone())
        .collect();

    // Compute the ONNX graph
    let outputs = session.run(inputs, output_names.as_slice())?;

    // Convert the output OrtValue to an ndarray to consume its contents.
    let out_tensor = outputs.get(0).unwrap().array_view::<f32>()?;

    assert_eq!(out_tensor.shape(), output0_shape.as_slice());
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, out_tensor[[0, i, 0, 0]]);
    }

    Ok(())
}
