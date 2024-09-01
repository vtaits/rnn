use std::env;

use actix_cors::Cors;
use actix_web::{cookie, get, http, post, web, App, Error, HttpResponse, HttpServer, Responder};
use rnn_core::DataLayer;
use rnn_instance::init_by_toml;
use serde_derive::{Deserialize, Serialize};
use timeline_helpers::{ComplexTimeline, ComplexTimelineValue, Timeline};

const NUMBER_OF_OPTIONS: usize = 4;

struct AppState {
    data_layer: DataLayer<Vec<ComplexTimelineValue>>,
}

#[post("/push_data")]
async fn push_data(
    req_body: web::Json<Vec<ComplexTimelineValue>>,
    data: web::Data<AppState>,
) -> impl Responder {
    data.data_layer.push_data(req_body.to_vec());

    HttpResponse::Ok().finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config_path = env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");

    let data_layer = init_by_toml(config_path);

    let filename = std::env::args().nth(1).expect("no filename given");

    let app_data = web::Data::new(AppState { data_layer });

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
            .app_data(app_data)
            .service(push_data)
    })
    .bind(("127.0.0.1", 8000))?
    .run()
    .await
}
