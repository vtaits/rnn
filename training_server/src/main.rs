use std::{
    env,
    sync::{Arc, Mutex},
    time::Duration,
};

use actix_cors::Cors;
use actix_web::middleware::Logger;
use actix_web::{http, post, web, App, HttpResponse, HttpServer, Responder};
use console_ui::run_console_app;
use env_logger::Env;
use rnn_core::DataLayer;
use rnn_instance::init_by_toml;
use timeline_helpers::ComplexTimelineValue;
use tokio::sync::Semaphore;
use tokio::{io::AsyncWriteExt, net::TcpStream, time::timeout};

struct AppState {
    data_layer: Mutex<DataLayer<Vec<ComplexTimelineValue>>>,
    receivers: Vec<String>,
}

#[post("/push_data")]
async fn push_data(
    req_body: web::Json<Vec<ComplexTimelineValue>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let timeline_data = req_body.into_inner();

    let mut data_layer = data.data_layer.lock().unwrap();

    data_layer.push_data(timeline_data);

    HttpResponse::Ok().finish()
}

async fn send_data_to_receiver(
    receiver: &str,
    data: Arc<Vec<u8>>,
    semaphore: Arc<Semaphore>,
) -> tokio::io::Result<()> {
    let permit = semaphore.acquire().await.unwrap();

    let mut stream = TcpStream::connect(receiver).await?;

    let request = format!(
        "POST {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Length: {}\r\n\
         Content-Type: application/octet-stream\r\n\
         Connection: close\r\n\
         \r\n",
        receiver,
        data.len()
    );

    stream.write_all(request.as_bytes()).await?;
    stream.write_all(&data).await?;

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
        let data_for_send = Arc::clone(&data_for_send);
        let semaphore = Arc::clone(&semaphore);

        let task = tokio::spawn(async move {
            timeout(
                Duration::from_secs(5),
                send_data_to_receiver(&receiver, data_for_send, semaphore),
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
    let config_path = env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");
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

    let data_layer = init_by_toml(config_path);

    let network = data_layer.get_network();

    let app_data = web::Data::new(AppState {
        receivers,
        data_layer: Mutex::new(data_layer),
    });

    let _ = tokio::join!(
        HttpServer::new(move || {
            let cors = Cors::default()
                .allowed_origin("http://127.0.0.1:3000")
                .allowed_origin("http://localhost:3000")
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
                .service(push_data)
                .service(update_receivers)
        })
        .bind(("127.0.0.1", port))?
        .run(),
        run_console_app(Arc::clone(&network)),
    );

    Ok(())
}
