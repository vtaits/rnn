use std::{
    env,
    sync::{Arc, Mutex},
    time::Duration,
};

use actix_cors::Cors;
use actix_web::middleware::Logger;
use actix_web::{get, http, post, web, App, HttpResponse, HttpServer, Responder};
// use console_ui::run_console_app;
use env_logger::Env;
use reqwest::Client;
use rnn_core::DataLayer;
use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};
use timeline_helpers::ComplexTimelineValue;
use tokio::sync::Semaphore;
use tokio::time::timeout;

struct AppState {
    client: Arc<Client>,
    data_layer: Mutex<DataLayer<Vec<ComplexTimelineValue>>>,
    receivers: Vec<String>,
}

#[get("/download_dump")]
async fn download_dump(data: web::Data<AppState>) -> impl Responder {
    let data_layer = data.data_layer.lock().unwrap();

    let compressed_data = data_layer
        .get_network()
        .read()
        .unwrap()
        .get_gzip_dump()
        .unwrap();

    HttpResponse::Ok()
        .insert_header((http::header::CONTENT_TYPE, "application/octet-stream"))
        .insert_header((
            http::header::CONTENT_DISPOSITION,
            "attachment; filename=\"network_dump.gzip\"",
        ))
        .body(compressed_data)
}

#[post("/push_data_binary")]
async fn push_data_binary(
    req_body: web::Json<Vec<bool>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let bit_vec = req_body.into_inner();

    let mut data_layer = data.data_layer.lock().unwrap();

    data_layer.push_data_binary_and_apply(&bit_vec, 0);

    HttpResponse::Ok().finish()
}

#[post("/push_data")]
async fn push_data(
    req_body: web::Json<Vec<ComplexTimelineValue>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let timeline_data = req_body.into_inner();

    let mut data_layer = data.data_layer.lock().unwrap();

    data_layer.push_data_and_apply(timeline_data, 0);

    HttpResponse::Ok().finish()
}

async fn send_data_to_receiver(
    client: &Client,
    receiver: &str,
    data: Arc<Vec<u8>>,
    semaphore: Arc<Semaphore>,
) -> tokio::io::Result<()> {
    let permit = semaphore.acquire().await.unwrap();

    let response = client.post(receiver).body(data.to_vec()).send().await;

    match response {
        Ok(res) => {
            println!("Request sent, response: {:?}", res);
            let response_text = res.text().await;
            println!("Response body: {:?}", response_text);
        }
        Err(err) => eprintln!("Error sending request: {:?}", err),
    }

    drop(permit);

    Ok(())
}

#[post("/update_receivers")]
async fn update_receivers(data: web::Data<AppState>) -> impl Responder {
    let data_layer = data.data_layer.lock().unwrap();

    let compressed_data = data_layer
        .get_network()
        .read()
        .unwrap()
        .get_gzip_dump()
        .unwrap();

    let data_for_send = Arc::new(compressed_data);

    let mut tasks = vec![];

    let semaphore = Arc::new(Semaphore::new(data.receivers.len()));

    for receiver in &data.receivers {
        let receiver = receiver.clone();
        let client = Arc::clone(&data.client);
        let data_for_send = Arc::clone(&data_for_send);
        let semaphore = Arc::clone(&semaphore);

        let task = tokio::spawn(async move {
            timeout(
                Duration::from_secs(5),
                send_data_to_receiver(&client, &receiver, data_for_send, semaphore),
            )
            .await
        });

        tasks.push(task);
    }

    for task in tasks {
        let _ = task.await.unwrap();
    }

    HttpResponse::Ok().finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port = match env::var("PORT") {
        Ok(port_str) => match port_str.parse::<u16>() {
            Ok(port) => port,
            _ => 8000,
        },
        _ => 8000,
    };

    let receivers_str = env::var("RECEIVERS").expect("RECEIVERS should be defined");

    let receivers = receivers_str
        .split(',')
        .map(|part| String::from(part))
        .collect();

    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let (data_layer, _) = init_data_layer_by_env(&InitDataLayerParams { train: true, end_measurement_index: None, start_measurement_index: None });

    let client = Client::new();

    let app_data = web::Data::new(AppState {
        client: Arc::new(client),
        receivers,
        data_layer: Mutex::new(data_layer),
    });

    let _ = tokio::join!(
        HttpServer::new(move || {
            let cors = Cors::default()
                .allowed_origin("http://127.0.0.1:5173")
                .allowed_origin("http://localhost:5173")
                .allowed_methods(vec!["GET", "POST"])
                .allowed_headers(vec![
                    http::header::AUTHORIZATION,
                    http::header::ACCEPT,
                    http::header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                    http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
                    http::header::CONTENT_TYPE,
                ])
                .supports_credentials();

            App::new()
                .wrap(Logger::default())
                .wrap(cors)
                .app_data(app_data.clone())
                .service(download_dump)
                .service(push_data_binary)
                .service(push_data)
                .service(update_receivers)
        })
        .bind(("0.0.0.0", port))?
        .run(),
        // run_console_app(Arc::clone(&network)),
    );

    Ok(())
}
