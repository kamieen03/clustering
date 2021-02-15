use super::clustering::Clustering;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;
use std::fs::File;

fn read_array(filename: String) -> Array2<f32> {
    let file = File::open(filename).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_reader(file);
    reader.deserialize_array2_dynamic().unwrap()
}

pub fn read_datafiles() -> Vec<Array2<f32>> {
    let data: Vec<Array2<f32>> = (1..=6)
        .map(|i| format!("data/in/{}.csv", i))
        .map(read_array)
        .collect();
    data
}

pub fn save(c: Clustering, i: usize, alg: &str) {
    let file = File::create(format!("data/out/{}/{}.csv", alg, i)).unwrap();
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_writer(file);
    c.data
        .axis_iter(Axis(0))
        .zip(c.clusters.iter())
        .map(|(point, cl)| {
            let mut rec: Vec<String> = point.iter().map(|x| format!("{}", x)).collect();
            rec.push(format!("{}", cl));
            rec
        })
        .for_each(|rec| writer.write_record(&rec).unwrap());
}
