use reqwest::Client;
use std::{env, time::Duration};
use tokio_cron_scheduler::{Job, JobScheduler, JobSchedulerError};

#[tokio::main]
async fn main() -> Result<(), JobSchedulerError> {
    let training_server_root =
        env::var("TRAINING_SERVER").expect("TRAINING_SERVER should be defined");

    let mut scheduler = JobScheduler::new().await?;
    let post_url = format!("{}/update_receivers", training_server_root);
    let client = Client::new();

    scheduler
        .add(Job::new_async("1/10 * * * * *", move |_uuid, _l| {
            println!("JOB");
            let post_url = post_url.clone();
            let client = client.clone();
            Box::pin(async move {
                let response = client.post(&post_url).send().await;

                match response {
                    Ok(res) => println!("Request sent, response: {:?}", res),
                    Err(err) => eprintln!("Error sending request: {:?}", err),
                }
            })
        })?)
        .await?;

    scheduler.start().await?;

    tokio::time::sleep(Duration::from_secs(3600)).await;

    scheduler.shutdown().await.unwrap();

    Ok(())
}
