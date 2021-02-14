use super::clustering::Clustering;

use ndarray::{Array, Array1, Array2, ArrayView, Axis, Dim, Ix};
use rand::distributions::{Distribution, Uniform};

impl<'a> Clustering<'a> {
    fn cluster_data(&mut self, means: &[Array1<f32>]) {
        for (i, point) in self.data.axis_iter(Axis(0)).enumerate() {
            let (idx, _) = means
                .iter()
                .enumerate()
                .min_by(|(_i, m0), (_j, m1)| {
                    diffnormav(&m0, &point)
                        .partial_cmp(&diffnormav(&m1, &point))
                        .unwrap()
                })
                .unwrap();
            self.clusters[i] = idx;
        }
    }

    fn compute_means(&self) -> Vec<Array1<f32>> {
        let mut sums: Vec<Array1<f32>> = vec![Array::zeros(self.data.ncols()); self.k.into()];
        let mut counts: Vec<u16> = vec![0; self.k.into()];
        self.clusters
            .iter()
            .zip(self.data.axis_iter(Axis(0)))
            .for_each(|(c, row)| {
                sums[*c] += &row;
                counts[*c] += 1;
            });
        sums.iter()
            .zip(counts.iter())
            .map(|(m, c)| {
                if *c == 0 {
                    Array::zeros(self.data.ncols())
                } else {
                    m / f32::from(*c)
                }
            })
            .collect()
    }
}

fn init_means(data: &Array2<f32>, k: u16) -> Vec<Array1<f32>> {
    let between = Uniform::from(0..data.nrows());
    let mut rng = rand::thread_rng();
    let mut means = Vec::<Array1<f32>>::new();
    for _ in 0..k {
        let row = between.sample(&mut rng);
        means.push(data.index_axis(Axis(0), row).into_owned());
    }
    means
}

fn diffnormaa(x: &Array1<f32>, y: &Array1<f32>) -> f32 {
    (x - y).iter().map(|x| x.powi(2)).sum()
}

fn diffnormav(x: &Array1<f32>, y: &ArrayView<f32, Dim<[Ix; 1]>>) -> f32 {
    (x - y).iter().map(|x| x.powi(2)).sum()
}

pub fn kmeans(d: &Array2<f32>, k: u16) -> Clustering {
    let eps = 1e-2;
    let mut means: Vec<Array1<f32>> = init_means(&d, k);
    let mut means_temp: Vec<Array1<f32>>;
    let mut moved: Vec<f32> = vec![1.0; means.len()];
    let mut data = Clustering::newk(&d, k);
    while moved.iter().any(|x| x > &eps) {
        data.cluster_data(&means);
        means_temp = data.compute_means();
        means
            .iter()
            .zip(means_temp.iter())
            .enumerate()
            .for_each(|(i, (m0, m1))| {
                moved[i] = diffnormaa(m0, m1);
            });
        means = means_temp;
    }
    data
}
