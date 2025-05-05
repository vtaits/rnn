use std::path::Path;

pub fn get_file_path(file_path: &str, config_dir: &Option<String>) -> String {
    match config_dir {
        Some(config_dir) => {
            let path = Path::new(&config_dir).join(file_path);

            path.to_str().unwrap().to_string()
        }
        None => String::from(file_path),
    }
}
