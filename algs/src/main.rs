mod clustering;
mod dataio;
mod dbscan;
mod kmeans;
mod preprocessing;

fn main() {
    dataio::read_datafiles()
        .iter()
        .enumerate()
        .map(|(i, d)| (i, kmeans::kmeans(d, 3)))
        .for_each(|(i, clustering)| dataio::save(clustering, i, "kmeans"));
    dataio::read_datafiles()
        .iter_mut()
        .map(|d| preprocessing::normalize(d))
        .enumerate()
        .map(|(i, d)| (i, dbscan::dbscan(d)))
        .for_each(|(i, clustering)| dataio::save(clustering, i, "dbscan"));
}
