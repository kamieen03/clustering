use super::clustering::Clustering;
use ndarray::{Array2,ArrayView,Axis,Dim,Ix};

fn remove_item(seeds: &mut Vec<usize>, i: usize) {
    let index = seeds.iter().position(|x| *x == i).unwrap();
    seeds.remove(index);
}

fn dist(x: &ArrayView<f32, Dim<[Ix; 1]>>, y: &ArrayView<f32, Dim<[Ix; 1]>>) -> f32 {
    (x - y).mapv(|z| z.powi(2)).sum().sqrt()
}

impl <'a> Clustering <'a> {
    fn region_query(&self, i: usize, eps: f32) -> Vec<usize> {
        let p = self.data.index_axis(Axis(0), i);
        (0..self.clusters.len())
            .filter(|j| {
                let q = self.data.index_axis(Axis(0), *j);
                dist(&p, &q) < eps
            })
            .collect()
    }
}

fn expand_cluster(data: &mut Clustering, i: usize, cluster_id: usize, eps: f32, minpts: usize) -> bool {
    let mut seeds: Vec<usize> = data.region_query(i, eps);
    if seeds.len() < minpts {
        data.clusters[i] = 1;   //noise
        false
    } else {
        for j in &seeds {
            data.clusters[*j] = cluster_id;
        }
        remove_item(&mut seeds, i);
        while !seeds.is_empty() {
            let curr_j = seeds.pop().unwrap();
            let result = data.region_query(curr_j, eps);
            if result.len() >= minpts {
                for j in result {
                    if data.clusters[j] < 2 {
                        if data.clusters[j] == 0 {
                            seeds.push(j);
                        }
                        data.clusters[j] = cluster_id;
                    }
                }
            }
        }
        true
    }
}


pub fn dbscan(d: &Array2<f32>) -> Clustering {
    // 0  - unclassified
    // 1  - noise
    // 2+ - clusters
    let eps: f32 = 0.2;
    let minpts: usize = 4;
    let mut data = Clustering::new(&d);
    let mut cluster_id: usize = 2;
    for i in 0..data.clusters.len() {
        if data.clusters[i] == 0 && expand_cluster(&mut data, i, cluster_id, eps, minpts) {
            cluster_id += 1;
        }
    }
    data
}
