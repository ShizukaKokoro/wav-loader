//! ロードモジュール
//!
//! 音源データをロードするためのモジュール。
//! ユーザーが直接アクセスすることになる。

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const INVALID_FILE_SIZE: &str = "Invalid file size";
const INVALID_CHUNK_ID: &str = "Invalid chunk id";
const INVALID_FORMAT: &str = "Invalid format";
const INVALID_FMT_CHUNK: &str = "Invalid fmt chunk";
const INVALID_FMT_CHUNK_SIZE: &str = "Invalid fmt chunk size";
const INVALID_BIT_DEPTH: &str = "Invalid bit depth";
const INVALID_SAMPLING_RATE: &str = "Invalid sampling rate";
const FILE_NOT_FOUND: &str = "File not found";
const INVALID_DATA_CHUNK: &str = "Invalid data chunk";

fn check_riff_chunk(file: &mut File) -> Result<(), &'static str> {
    let mut buffer = [0; 12];
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }

    let chunk_id = &buffer[0..4];
    let format = &buffer[8..12];
    if chunk_id != b"RIFF" {
        return Err(INVALID_CHUNK_ID);
    } else if format != b"WAVE" {
        return Err(INVALID_FORMAT);
    }
    Ok(())
}

fn check_junk_chunk(file: &mut File) -> Result<(), &'static str> {
    let mut buffer = [0; 4];
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }
    if &buffer == b"JUNK" {
        // println!("JUNK chunk found");
        match file.read_exact(&mut buffer) {
            Ok(_) => {}
            Err(_) => return Err(INVALID_FILE_SIZE),
        }
        // println!("JUNK chunk size: {:?}", buffer);
        //skip junk chunk
        match file.seek(SeekFrom::Current(u32::from_le_bytes(buffer) as i64)) {
            Ok(_) => {}
            Err(_) => return Err(INVALID_FILE_SIZE),
        }
    }
    Ok(())
}

fn check_fmt_chunk(file: &mut File) -> Result<(u16, u32, u16), &'static str> {
    let mut buffer = [0; 4];
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }
    if &buffer != b"fmt " {
        return Err(INVALID_FMT_CHUNK);
    }
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }

    let channels;
    let sampling_rate;
    let bit_depth;
    if buffer == [16, 0, 0, 0] {
        let mut fmt_buffer = [0; 16];
        match file.read_exact(&mut fmt_buffer) {
            Ok(_) => {}
            Err(_) => return Err(INVALID_FILE_SIZE),
        }

        channels = u16::from_le_bytes([fmt_buffer[2], fmt_buffer[3]]);
        sampling_rate =
            u32::from_le_bytes([fmt_buffer[4], fmt_buffer[5], fmt_buffer[6], fmt_buffer[7]]);
        bit_depth = u16::from_le_bytes([fmt_buffer[14], fmt_buffer[15]]);

        if fmt_buffer[0..2] == [1, 0] {
            if ![8, 16, 24].contains(&bit_depth) {
                return Err(INVALID_BIT_DEPTH);
            }
        } else if fmt_buffer[0..2] == [3, 0] {
            if bit_depth != 32 {
                return Err(INVALID_BIT_DEPTH);
            }
        } else {
            return Err(INVALID_FORMAT);
        }
    } else {
        return Err(INVALID_FMT_CHUNK_SIZE);
    }
    Ok((channels, sampling_rate, bit_depth))
}

fn check_data_chunk(file: &mut File) -> Result<Vec<u8>, &'static str> {
    let mut buffer = [0; 4];
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }
    if &buffer != b"data" {
        return Err(INVALID_DATA_CHUNK);
    }
    match file.read_exact(&mut buffer) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }
    let data_size = u32::from_le_bytes(buffer);
    // println!("data size: {}", data_size);
    let mut data = vec![0u8; data_size as usize];
    match file.read_exact(&mut data) {
        Ok(_) => {}
        Err(_) => return Err(INVALID_FILE_SIZE),
    }
    Ok(data)
}

/// wav ファイルの情報の取得
///
/// # Arguments
///
/// * `path` - ファイルのパス
///
/// # Returns
///
/// チャンネル数、サンプリング周波数、ビット深度。
pub fn get_info<T>(path: T) -> Result<(u16, u32, u16), &'static str>
where
    T: AsRef<std::path::Path>,
{
    let file = File::open(path);
    if file.is_err() {
        return Err(FILE_NOT_FOUND);
    }
    let mut file = file.unwrap();

    check_riff_chunk(&mut file)?;
    check_junk_chunk(&mut file)?;
    check_fmt_chunk(&mut file)
}

fn to_vec_i8_from_i8(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i8> {
    if buffer.len() % ch_cnt != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .iter()
        .skip(channel)
        .step_by(ch_cnt)
        .map(|&x| (x as i16 - 128) as i8)
        .collect()
}

/// 8bit の wav ファイルを読み込む
///
/// # Arguments
///
/// * `path` - ファイルのパス
/// * `sampling_rate` - サンプリング周波数
///
/// # Returns
///
/// 読み込んだデータ。
/// ファイルが見つからない場合はエラーを返す。
/// 復旧可能であるべき。
pub fn load_wav_8bit<T>(path: T, sampling_rate: u32) -> Result<Vec<Vec<i8>>, &'static str>
where
    T: AsRef<std::path::Path>,
{
    let file = File::open(path);
    if file.is_err() {
        return Err(FILE_NOT_FOUND);
    }
    let mut file = file.unwrap();

    check_riff_chunk(&mut file)?;
    check_junk_chunk(&mut file)?;

    let (channels, sr, bit_depth) = check_fmt_chunk(&mut file)?;
    if sr != sampling_rate {
        return Err(INVALID_SAMPLING_RATE);
    }
    let data = check_data_chunk(&mut file)?;

    let channels = channels as usize;
    let mut sampleses = Vec::with_capacity(channels);
    match bit_depth {
        8 => {
            for i in 0..channels {
                sampleses.push(to_vec_i8_from_i8(&data, i, channels));
            }
            Ok(sampleses)
        }
        _ => Err(INVALID_BIT_DEPTH),
    }
}

fn to_vec_i16_from_i8(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i16> {
    if buffer.len() % ch_cnt != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .iter()
        .skip(channel)
        .step_by(ch_cnt)
        .map(|&x| ((x as i16 - 128) << 8))
        .collect()
}

fn to_vec_i16_from_i16(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i16> {
    if buffer.len() % (ch_cnt * 2) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(2)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| i16::from_le_bytes([x[0], x[1]]))
        .collect()
}

/// 16bit の wav ファイルを読み込む
///
/// # Arguments
///
/// * `path` - ファイルのパス
/// * `sampling_rate` - サンプリング周波数
///
/// # Returns
///
/// 読み込んだデータ。
/// ファイルが見つからない場合はエラーを返す。
pub fn load_wav_16bit<T>(path: T, sampling_rate: u32) -> Result<Vec<Vec<i16>>, &'static str>
where
    T: AsRef<std::path::Path>,
{
    let file = File::open(path);
    if file.is_err() {
        return Err(FILE_NOT_FOUND);
    }
    let mut file = file.unwrap();

    check_riff_chunk(&mut file)?;
    check_junk_chunk(&mut file)?;

    let (channels, sr, bit_depth) = check_fmt_chunk(&mut file)?;
    if sr != sampling_rate {
        return Err(INVALID_SAMPLING_RATE);
    }
    let data = check_data_chunk(&mut file)?;

    let channels = channels as usize;
    let mut sampleses = Vec::with_capacity(channels);
    match bit_depth {
        8 => {
            for i in 0..channels {
                sampleses.push(to_vec_i16_from_i8(&data, i, channels));
            }
            Ok(sampleses)
        }
        16 => {
            for i in 0..channels {
                sampleses.push(to_vec_i16_from_i16(&data, i, channels));
            }
            Ok(sampleses)
        }
        _ => Err(INVALID_BIT_DEPTH),
    }
}

fn to_vec_i32_from_i8(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i32> {
    if buffer.len() % ch_cnt != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .iter()
        .skip(channel)
        .step_by(ch_cnt)
        .map(|&x| ((x as i32 - 128) << 16))
        .collect()
}

fn to_vec_i32_from_i16(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i32> {
    if buffer.len() % (ch_cnt * 2) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(2)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| i32::from_le_bytes([x[0], x[1], 0, 0]))
        .collect()
}

fn to_vec_i32_from_i24(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<i32> {
    if buffer.len() % (ch_cnt * 3) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(3)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], 0]))
        .collect()
}

/// 24bit の wav ファイルを読み込む
///
/// # Arguments
///
/// * `path` - ファイルのパス
/// * `sampling_rate` - サンプリング周波数
///
/// # Returns
///
/// 読み込んだデータ。
/// ファイルが見つからない場合はエラーを返す。
pub fn load_wav_24bit<T>(path: T, sampling_rate: u32) -> Result<Vec<Vec<i32>>, &'static str>
where
    T: AsRef<std::path::Path>,
{
    let file = File::open(path);
    if file.is_err() {
        return Err(FILE_NOT_FOUND);
    }
    let mut file = file.unwrap();

    check_riff_chunk(&mut file)?;
    check_junk_chunk(&mut file)?;

    let (channels, sr, bit_depth) = check_fmt_chunk(&mut file)?;
    if sr != sampling_rate {
        return Err(INVALID_SAMPLING_RATE);
    }
    let data = check_data_chunk(&mut file)?;

    let channels = channels as usize;
    let mut sampleses = Vec::with_capacity(channels);
    match bit_depth {
        8 => {
            for i in 0..channels {
                sampleses.push(to_vec_i32_from_i8(&data, i, channels));
            }
            Ok(sampleses)
        }
        16 => {
            for i in 0..channels {
                sampleses.push(to_vec_i32_from_i16(&data, i, channels));
            }
            Ok(sampleses)
        }
        24 => {
            for i in 0..channels {
                sampleses.push(to_vec_i32_from_i24(&data, i, channels));
            }
            Ok(sampleses)
        }
        _ => Err(INVALID_BIT_DEPTH),
    }
}

fn to_vec_f32_from_i8(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<f32> {
    if buffer.len() % ch_cnt != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .iter()
        .skip(channel)
        .step_by(ch_cnt)
        .map(|&x| (x as f32 - 128.0) / 128.0)
        .collect()
}

fn to_vec_f32_from_i16(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<f32> {
    if buffer.len() % (ch_cnt * 2) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(2)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| i16::from_le_bytes([x[0], x[1]]) as f32 / (1 << 15) as f32)
        .collect()
}

fn to_vec_f32_from_i24(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<f32> {
    if buffer.len() % (ch_cnt * 3) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(3)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], 0]) as f32 / (1 << 23) as f32)
        .collect()
}

fn to_vec_f32_from_f32(buffer: &[u8], channel: usize, ch_cnt: usize) -> Vec<f32> {
    if buffer.len() % (ch_cnt * 4) != 0 {
        panic!("Invalid buffer size");
    }
    buffer
        .chunks(4)
        .skip(channel)
        .step_by(ch_cnt)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

/// 32bit float の wav ファイルを読み込む
///
/// # Arguments
///
/// * `path` - ファイルのパス
/// * `sampling_rate` - サンプリング周波数
///
/// # Returns
///
/// 読み込んだデータ。
/// ファイルが見つからない場合はエラーを返す。
pub fn load_wav_32bit_float<T>(path: T, sampling_rate: u32) -> Result<Vec<Vec<f32>>, &'static str>
where
    T: AsRef<std::path::Path>,
{
    let file = File::open(path);
    if file.is_err() {
        return Err(FILE_NOT_FOUND);
    }
    let mut file = file.unwrap();

    check_riff_chunk(&mut file)?;
    check_junk_chunk(&mut file)?;

    let (channels, sr, bit_depth) = check_fmt_chunk(&mut file)?;
    if sr != sampling_rate {
        return Err(INVALID_SAMPLING_RATE);
    }
    let data = check_data_chunk(&mut file)?;

    let channels = channels as usize;
    let mut sampleses = Vec::with_capacity(channels);
    match bit_depth {
        8 => {
            for i in 0..channels {
                sampleses.push(to_vec_f32_from_i8(&data, i, channels));
            }
            Ok(sampleses)
        }
        16 => {
            for i in 0..channels {
                sampleses.push(to_vec_f32_from_i16(&data, i, channels));
            }
            Ok(sampleses)
        }
        24 => {
            for i in 0..channels {
                sampleses.push(to_vec_f32_from_i24(&data, i, channels));
            }
            Ok(sampleses)
        }
        32 => {
            for i in 0..channels {
                sampleses.push(to_vec_f32_from_f32(&data, i, channels));
            }
            Ok(sampleses)
        }
        _ => Err(INVALID_BIT_DEPTH),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_i8_new_mono() {
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7];
        let samples = to_vec_i8_from_i8(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len());
        for &x in buffer.iter() {
            expected.push(x as i8 - 127 - 1);
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_i8_new_stereo() {
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7];
        let samples = to_vec_i8_from_i8(&buffer, 0, 2);
        let mut ch_0 = Vec::with_capacity(buffer.len() / 2);
        let mut ch_1 = Vec::with_capacity(buffer.len() / 2);
        for i in 0..buffer.len() / 2 {
            ch_0.push(buffer[i * 2] as i8 - 127 - 1);
            ch_1.push(buffer[i * 2 + 1] as i8 - 127 - 1);
        }
        assert_eq!(samples, ch_0);
        let samples = to_vec_i8_from_i8(&buffer, 1, 2);
        assert_eq!(samples, ch_1);
    }

    #[test]
    fn test_samples_i16_new_mono() {
        let buffer = [0, 0, 1, 0, 2, 0, 3, 0];
        let samples = to_vec_i16_from_i16(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 2);
        for i in 0..buffer.len() / 2 {
            expected.push(i16::from_le_bytes([buffer[i * 2], buffer[i * 2 + 1]]));
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_i16_new_stereo() {
        let buffer = [0, 0, 1, 0, 2, 0, 3, 0];
        let samples = to_vec_i16_from_i16(&buffer, 0, 2);
        let mut ch_0 = Vec::with_capacity(buffer.len() / 4);
        let mut ch_1 = Vec::with_capacity(buffer.len() / 4);
        for i in 0..buffer.len() / 4 {
            ch_0.push(i16::from_le_bytes([buffer[i * 4], buffer[i * 4 + 1]]));
            ch_1.push(i16::from_le_bytes([buffer[i * 4 + 2], buffer[i * 4 + 3]]));
        }
        assert_eq!(samples, ch_0);
        let samples = to_vec_i16_from_i16(&buffer, 1, 2);
        assert_eq!(samples, ch_1);
    }

    #[test]
    fn test_samples_i16_from_i8() {
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7];
        let samples = to_vec_i16_from_i8(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len());
        for &x in buffer.iter() {
            expected.push(i16::from(x as i8 - 127 - 1) << 8);
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_i24_new_mono() {
        let buffer = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0];
        let samples = to_vec_i32_from_i24(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 3);
        for i in 0..buffer.len() / 3 {
            expected.push(i32::from_le_bytes([
                buffer[i * 3],
                buffer[i * 3 + 1],
                buffer[i * 3 + 2],
                0,
            ]));
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_i24_new_stereo() {
        let buffer = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0];
        let samples = to_vec_i32_from_i24(&buffer, 0, 2);
        let mut ch_0 = Vec::with_capacity(buffer.len() / 6);
        let mut ch_1 = Vec::with_capacity(buffer.len() / 6);
        for i in 0..buffer.len() / 6 {
            ch_0.push(i32::from_le_bytes([
                buffer[i * 6],
                buffer[i * 6 + 1],
                buffer[i * 6 + 2],
                0,
            ]));
            ch_1.push(i32::from_le_bytes([
                buffer[i * 6 + 3],
                buffer[i * 6 + 4],
                buffer[i * 6 + 5],
                0,
            ]));
        }
        assert_eq!(samples, ch_0);
        let samples = to_vec_i32_from_i24(&buffer, 1, 2);
        assert_eq!(samples, ch_1);
    }

    #[test]
    fn test_samples_i24_from_i8() {
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7];
        let samples = to_vec_i32_from_i8(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len());
        for &x in buffer.iter() {
            expected.push(i32::from(x as i8 - 127 - 1) << 16);
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_i24_from_i16() {
        let buffer = [0, 0, 1, 0, 2, 0, 3, 0];
        let samples = to_vec_i32_from_i16(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len());
        for i in 0..buffer.len() / 2 {
            expected.push(i32::from_le_bytes([buffer[i * 2], buffer[i * 2 + 1], 0, 0]));
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_f32_new_mono() {
        let buffer = [0, 0, 0, 0, 0, 0, 128, 63];
        let samples = to_vec_f32_from_f32(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 4);
        for i in 0..buffer.len() / 4 {
            expected.push(f32::from_le_bytes([
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ]));
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_f32_new_stereo() {
        let buffer = [0, 0, 0, 0, 0, 0, 128, 63];
        let samples = to_vec_f32_from_f32(&buffer, 0, 2);
        let mut ch_0 = Vec::with_capacity(buffer.len() / 8);
        let mut ch_1 = Vec::with_capacity(buffer.len() / 8);
        for i in 0..buffer.len() / 8 {
            ch_0.push(f32::from_le_bytes([
                buffer[i * 8],
                buffer[i * 8 + 1],
                buffer[i * 8 + 2],
                buffer[i * 8 + 3],
            ]));
            ch_1.push(f32::from_le_bytes([
                buffer[i * 8 + 4],
                buffer[i * 8 + 5],
                buffer[i * 8 + 6],
                buffer[i * 8 + 7],
            ]));
        }
        assert_eq!(samples, ch_0);
        let samples = to_vec_f32_from_f32(&buffer, 1, 2);
        assert_eq!(samples, ch_1);
    }

    #[test]
    fn test_samples_f32_from_i8() {
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7];
        let samples = to_vec_f32_from_i8(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len());
        for &x in buffer.iter() {
            expected.push((x as f32 - 128.0) / 128.0);
        }
        assert!(samples
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 0.0001));
    }

    #[test]
    fn test_samples_f32_from_i16() {
        let buffer = [0, 0, 1, 0, 2, 0, 3, 0];
        let samples = to_vec_f32_from_i16(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 2);
        for i in 0..buffer.len() / 2 {
            expected
                .push((i16::from_le_bytes([buffer[i * 2], buffer[i * 2 + 1]]) as f32) / 32767.0);
        }
        assert!(samples
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 0.0001));
    }

    #[test]
    fn test_samples_f32_from_i24() {
        let buffer = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0];
        let samples = to_vec_f32_from_i24(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 3);
        for i in 0..buffer.len() / 3 {
            expected.push(
                (i32::from_le_bytes([buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0])
                    as f32)
                    / 8388607.0,
            );
        }
        assert!(samples
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 0.0001));
    }

    #[test]
    fn test_samples_f32_from_f32() {
        let buffer = [0, 0, 0, 0, 0, 0, 128, 63];
        let samples = to_vec_f32_from_f32(&buffer, 0, 1);
        let mut expected = Vec::with_capacity(buffer.len() / 4);
        for i in 0..buffer.len() / 4 {
            expected.push(f32::from_le_bytes([
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ]));
        }
        assert_eq!(samples, expected);
    }
}
