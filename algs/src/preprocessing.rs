use ndarray::{Array,Array2,Axis};

pub fn normalize(data: &mut Array2<f32>) -> &Array2<f32> {
    let std = data.std_axis(Axis(0), 0.0);
    let mut stdmat = Array::zeros(data.raw_dim());
    stdmat.assign(&std);
    let mean = data.mean_axis(Axis(0)).unwrap();
    let mut meanmat = Array::zeros(data.raw_dim());
    meanmat.assign(&mean);
    *data -= &meanmat;
    *data /= &stdmat;
    data
}

