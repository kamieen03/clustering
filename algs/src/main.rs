mod clustering;
mod dataio;
mod kmeans;

fn main() {
    dataio::read_datafiles()
        .iter()
        .enumerate()
        .map(|(i, d)| (i, kmeans::kmeans(d, 3)))
        .for_each(|(i, clustering)| dataio::save(clustering, i, "kmeans"));
}
