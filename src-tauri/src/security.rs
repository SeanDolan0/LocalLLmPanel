// DPAPI token encryption for Windows + portable base64 implementation.

pub fn base64_encode(data: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 {
            chunk[1] as usize
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            chunk[2] as usize
        } else {
            0
        };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(CHARSET[(triple >> 18) & 0x3F] as char);
        out.push(CHARSET[(triple >> 12) & 0x3F] as char);
        if chunk.len() > 1 {
            out.push(CHARSET[(triple >> 6) & 0x3F] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARSET[triple & 0x3F] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let clean = input.trim_end_matches('=');
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for &byte in clean.as_bytes() {
        let val = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'\r' | b'\n' | b' ' => continue,
            _ => return None,
        } as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

#[cfg(target_os = "windows")]
mod win {
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct DATA_BLOB {
        pub cbData: u32,
        pub pbData: *mut u8,
    }

    #[link(name = "crypt32")]
    extern "system" {
        pub fn CryptProtectData(
            pDataIn: *const DATA_BLOB,
            szDataDescr: *const u16,
            pOptionalEntropy: *const DATA_BLOB,
            pvReserved: *mut std::ffi::c_void,
            pPromptStruct: *mut std::ffi::c_void,
            dwFlags: u32,
            pDataOut: *mut DATA_BLOB,
        ) -> i32;

        pub fn CryptUnprotectData(
            pDataIn: *const DATA_BLOB,
            ppszDataDescr: *mut *mut u16,
            pOptionalEntropy: *const DATA_BLOB,
            pvReserved: *mut std::ffi::c_void,
            pPromptStruct: *mut std::ffi::c_void,
            dwFlags: u32,
            pDataOut: *mut DATA_BLOB,
        ) -> i32;
    }

    extern "system" {
        pub fn LocalFree(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }
}

/// Encrypts a plaintext string using Windows DPAPI (CryptProtectData).
/// Returns "dpapi:<base64-ciphertext>".
pub fn encrypt_token(plain: &str) -> Result<String, String> {
    if plain.trim().is_empty() {
        return Ok(String::new());
    }
    if plain.starts_with("dpapi:") {
        return Ok(plain.to_string());
    }

    #[cfg(target_os = "windows")]
    {
        use win::*;
        let mut in_bytes = plain.as_bytes().to_vec();
        let data_in = DATA_BLOB {
            cbData: in_bytes.len() as u32,
            pbData: in_bytes.as_mut_ptr(),
        };
        let mut data_out = DATA_BLOB {
            cbData: 0,
            pbData: std::ptr::null_mut(),
        };
        // dwFlags = 0x01: CRYPTPROTECT_UI_FORBIDDEN
        let ret = unsafe {
            CryptProtectData(
                &data_in,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0x01,
                &mut data_out,
            )
        };
        if ret == 0 {
            return Err("CryptProtectData failed".to_string());
        }
        let slice =
            unsafe { std::slice::from_raw_parts(data_out.pbData, data_out.cbData as usize) };
        let encoded = base64_encode(slice);
        unsafe { LocalFree(data_out.pbData as *mut _) };
        Ok(format!("dpapi:{encoded}"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Non-windows test fallback:
        Ok(format!("dpapi:mock:{}", base64_encode(plain.as_bytes())))
    }
}

/// Decrypts a DPAPI ciphertext string ("dpapi:<base64>") back to plaintext.
/// If the string is not DPAPI-prefixed, it returns the input string unchanged.
pub fn decrypt_token(cipher: &str) -> Result<String, String> {
    let trimmed = cipher.trim();
    if trimmed.is_empty() || !trimmed.starts_with("dpapi:") {
        return Ok(trimmed.to_string());
    }

    let payload = &trimmed["dpapi:".len()..];

    #[cfg(target_os = "windows")]
    {
        if payload.starts_with("mock:") {
            let b64 = &payload["mock:".len()..];
            let bytes = base64_decode(b64).ok_or("invalid base64 in mock token")?;
            return String::from_utf8(bytes).map_err(|e| e.to_string());
        }

        use win::*;
        let mut raw_bytes = base64_decode(payload).ok_or("invalid base64 in dpapi token")?;
        let data_in = DATA_BLOB {
            cbData: raw_bytes.len() as u32,
            pbData: raw_bytes.as_mut_ptr(),
        };
        let mut data_out = DATA_BLOB {
            cbData: 0,
            pbData: std::ptr::null_mut(),
        };
        let ret = unsafe {
            CryptUnprotectData(
                &data_in,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0x01,
                &mut data_out,
            )
        };
        if ret == 0 {
            return Err("CryptUnprotectData failed to decrypt token".to_string());
        }
        let slice =
            unsafe { std::slice::from_raw_parts(data_out.pbData, data_out.cbData as usize) };
        let text = String::from_utf8(slice.to_vec()).map_err(|e| e.to_string());
        unsafe { LocalFree(data_out.pbData as *mut _) };
        text
    }

    #[cfg(not(target_os = "windows"))]
    {
        let b64 = if payload.starts_with("mock:") {
            &payload["mock:".len()..]
        } else {
            payload
        };
        let bytes = base64_decode(b64).ok_or("invalid base64 in token")?;
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let cases = [
            "",
            "hello",
            "hf_1234567890abcdef",
            "a very long token with symbols: !@#$%^&*()_+{}[]|:;<>?,./~`",
        ];
        for c in cases {
            let enc = base64_encode(c.as_bytes());
            let dec = base64_decode(&enc).expect("decode");
            assert_eq!(String::from_utf8(dec).unwrap(), c);
        }
    }

    #[test]
    fn test_dpapi_encrypt_decrypt_roundtrip() {
        let secret = "hf_pBqJ78x9XyZ1029384756";
        let encrypted = encrypt_token(secret).expect("encrypt failed");
        assert!(encrypted.starts_with("dpapi:"));
        assert_ne!(encrypted, secret);

        let decrypted = decrypt_token(&encrypted).expect("decrypt failed");
        assert_eq!(decrypted, secret);
    }

    #[test]
    fn test_dpapi_plain_passthrough() {
        let plain = "regular_plaintext_token";
        let out = decrypt_token(plain).expect("decrypt failed");
        assert_eq!(out, plain);
    }

    #[test]
    fn test_dpapi_empty() {
        assert_eq!(encrypt_token("").unwrap(), "");
        assert_eq!(decrypt_token("").unwrap(), "");
    }
}
