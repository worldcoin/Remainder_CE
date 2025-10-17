use ark_serialize::Read;
/// Helper function to read bytes from a file, preallocating the required space.
/// TODO(Makis): Consider using `Path`/`PathBuf` when appropriate.
pub fn read_bytes_from_file(filename: &str) -> Vec<u8> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    bufreader
}
