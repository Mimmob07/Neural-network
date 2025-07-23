use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    time::Instant,
};

pub struct MnistImages {
    pub images: Vec<Vec<u8>>,
    pub labels: Vec<u8>,
}

pub fn unpack<T: AsRef<str>>(images_filename: T, labels_filename: T) -> io::Result<MnistImages> {
    let start_time = Instant::now();
    let mut images_file = File::open(images_filename.as_ref())?;
    let mut labels_file = File::open(labels_filename.as_ref())?;
    let mut images_magic_number = [0u8; 4];
    let mut labels_magic_number = [0u8; 4];
    images_file.read_exact(&mut images_magic_number)?;
    labels_file.read_exact(&mut labels_magic_number)?;
    assert!(
        u32::from_be_bytes(images_magic_number) == 0x00000803
            && u32::from_be_bytes(labels_magic_number) == 0x00000801
    );

    let mut images_len_u8 = [0u8; 4];
    let mut labels_len_u8 = [0u8; 4];
    images_file.read_exact(&mut images_len_u8)?;
    labels_file.read_exact(&mut labels_len_u8)?;
    assert!(images_len_u8 == labels_len_u8);

    let images_len = u32::from_be_bytes(images_len_u8) as usize;
    let mut data = MnistImages {
        images: vec![vec![0u8; 784]; images_len],
        labels: vec![0u8; images_len],
    };

    images_file.seek(SeekFrom::Current(8))?;

    for i in 0..images_len {
        let mut pixels = [0u8; 784];
        let mut label = [0u8];
        images_file.read_exact(&mut pixels)?;
        labels_file.read_exact(&mut label)?;

        data.images[i] = pixels.to_vec();
        data.labels[i] = label[0];
    }

    println!("Took {}s to load images", start_time.elapsed().as_secs());
    Ok(data)
}
