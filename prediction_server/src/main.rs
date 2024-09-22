use std::{
    env,
    sync::{Arc, Mutex, RwLock},
};

use actix_cors::Cors;
use actix_web::middleware::Logger;
use actix_web::{http, post, web, App, HttpResponse, HttpServer, Responder};
use console_ui::run_console_app;
use env_logger::Env;
use rnn_core::{DataLayer, Network, NetworkParseError};
use rnn_instance::init_by_toml;
use serde_derive::Serialize;
use timeline_helpers::ComplexTimelineValue;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

struct AppState {
    data_layer: Mutex<DataLayer<Vec<ComplexTimelineValue>>>,
}

#[post("/predict")]
async fn predict(
    req_body: web::Json<Vec<ComplexTimelineValue>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let timeline_data = req_body.into_inner();

    let mut data_layer = data.data_layer.lock().unwrap();

    let result = data_layer.predict(timeline_data);

    match result {
        Ok(prediction) => HttpResponse::Ok()
            .content_type("application/json")
            .json(prediction),

        _ => HttpResponse::BadRequest().finish(),
    }
}

#[post("/update_network")]
async fn update_network(bytes: web::Bytes, data: web::Data<AppState>) -> impl Responder {
    let mut data_layer = data.data_layer.lock().unwrap();

    let dump: Vec<u8> = bytes.to_vec();

    match Network::from_gzip_dump_bytes(&dump) {
        Ok(network) => {
            data_layer.replace_network(Arc::new(RwLock::new(network)));
            return HttpResponse::Ok().finish();
        }
        Err(error) => match error {
            NetworkParseError::Gz(error) => HttpResponse::BadRequest().json(ErrorResponse {
                error: error.to_string(),
            }),
            NetworkParseError::JSON(error) => HttpResponse::BadRequest().json(ErrorResponse {
                error: error.to_string(),
            }),
        },
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config_path = env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");
    let port = match env::var("PORT") {
        Ok(port_str) => match port_str.parse::<u16>() {
            Ok(port) => port,
            _ => 8001,
        },
        _ => 8001,
    };

    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let mut data_layer = init_by_toml(config_path);

    if let Ok(dump_path) = env::var("DUMP_PATH") {
        let dump = std::fs::read_to_string(dump_path).unwrap();

        let network = Network::from_json_dump(&dump).unwrap();

        data_layer.replace_network(Arc::new(RwLock::new(network)));
    }

    let network = data_layer.get_network();

    let app_data = web::Data::new(AppState {
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
                // .wrap(Logger::default())
                .wrap(cors)
                .app_data(web::PayloadConfig::new(100 * 1024 * 1024))
                .app_data(app_data.clone())
                .service(predict)
                .service(update_network)
        })
        .bind(("127.0.0.1", port))?
        .run(),
        run_console_app(Arc::clone(&network)),
    );

    Ok(())
}
