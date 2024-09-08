use std::{env, sync::{Arc, Mutex, RwLock}};

use actix_cors::Cors;
use actix_web::{http, post, web, App, HttpResponse, HttpServer, Responder};
use rnn_core::{DataLayer, Network};
use rnn_instance::init_by_toml;
use timeline_helpers::ComplexTimelineValue;
use console_ui::run_console_app;

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
async fn update_network(
    bytes: web::Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    let mut data_layer = data.data_layer.lock().unwrap();

    let dump: Vec<u8> = bytes.to_vec();

    let next_network = Network::from_gzip_dump_bytes(&dump);

    if let Ok(network) = next_network {
        data_layer.replace_network(Arc::new(RwLock::new(network)));
        return  HttpResponse::Ok().finish();
    }

    HttpResponse::BadRequest().finish()
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
                .wrap(cors)
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
