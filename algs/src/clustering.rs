use ndarray::Array2;

pub struct Clustering<'a> {
    pub data: &'a Array2<f32>,
    pub clusters: Vec<usize>,
    pub k: u16, //number of clusters
}

impl<'a> Clustering<'a> {
    pub fn new(data: &Array2<f32>) -> Clustering {
        Clustering {
            data,
            clusters: vec![0; data.nrows()],
            k: 0,
        }
    }

    pub fn newk(data: &Array2<f32>, k: u16) -> Clustering {
        Clustering {
            data,
            clusters: vec![0; data.nrows()],
            k,
        }
    }
}
