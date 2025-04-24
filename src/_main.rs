mod utils;
mod gng;

use std::path::Path;
use image::GenericImageView;

use gng::GrowingNeuralGas;
use utils::extract_data_from_image;

fn main() -> Result<(), Box<dyn std::error::Error>>{
    println!("画像データでGNGを訓練");

    let image_file_path = Path::new("sample_image.png");
    let img = image::open(image_file_path)?;
    let image_data = extract_data_from_image(&img)?;
    
    Ok(())
}