use image::{self, DynamicImage, GenericImageView};
use nalgebra::Point2;

// 画像ファイルからデータ点を抽出する
pub fn extract_data_from_image(img: &DynamicImage) -> Result<Vec<Point2<f64>>, image::ImageError> {
    let mut non_zero_point = Vec::new();
    for (x, y, pixel) in img.pixels() {
        if pixel.0.iter().any(|&channel_value| channel_value < 240) {
            non_zero_point.push(Point2::new(x as f64, y as f64));
        }
    }
    Ok(non_zero_point)
}