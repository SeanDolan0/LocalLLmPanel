// Generate a 1024x1024 gradient app icon PNG (no deps).
// Produces src-tauri/icons/source.png for `tauri icon`.
import { deflateSync } from "node:zlib";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

const S = 1024;
const px = Buffer.alloc(S * (S * 4 + 1)); // scanlines with filter byte 0

// Indigo (#6366f1) top-left → cyan (#06b6d4) bottom-right, with a subtle
// diagonal "chip" motif: a lighter rounded rectangle grid.
const lerp = (a, b, t) => Math.round(a + (b - a) * t);
const rgb = (r, g, b) => [r, g, b];

for (let y = 0; y < S; y++) {
  const row = y * (S * 4 + 1);
  px[row] = 0; // filter: none
  for (let x = 0; x < S; x++) {
    const t = (x + y) / (2 * S);
    let [r, g, b] = [lerp(99, 6, t), lerp(102, 182, t), lerp(241, 212, t)];
    // panel: centered rounded "core" region slightly lighter
    const cx = S / 2, cy = S / 2;
    const dx = Math.abs(x - cx), dy = Math.abs(y - cy);
    const rx = S * 0.30, ry = S * 0.30;
    if (dx < rx && dy < ry) {
      const rr = dx / rx, ry_ = dy / ry;
      // rounded corner check
      const corner = Math.sqrt(Math.pow(Math.max(dx - rx + S * 0.08, 0) / (S * 0.08), 2) +
        Math.pow(Math.max(dy - ry + S * 0.08, 0) / (S * 0.08), 2));
      const inside = corner <= 1;
      if (inside) {
        r = lerp(r, 255, 0.18); g = lerp(g, 255, 0.18); b = lerp(b, 255, 0.18);
        // grid lines: every 64px, subtle
        const grid = (x + y) % 64 === 0 || (x - y) % 64 === 0;
        if (grid) { r = lerp(r, 0, 0.15); g = lerp(g, 0, 0.15); b = lerp(b, 0, 0.15); }
        // diagonal accent line
        if (Math.abs((x - cy) - (y - cx)) < 6 && dy < ry * 0.2) {
          [r, g, b] = rgb(34, 211, 238);
        }
      }
      void rr; void ry_; void corner;
    }
    const i = row + 1 + x * 4;
    px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
  }
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const typeBuf = Buffer.from(type, "ascii");
  const crc = deflateSync; // placeholder
  void crc;
  // CRC32
  let crc32 = 0xffffffff;
  const table = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  const push = (buf) => {
    for (const byte of buf) crc32 = table[(crc32 ^ byte) & 0xff] ^ (crc32 >>> 8);
  };
  push(typeBuf); push(data);
  crc32 = (crc32 ^ 0xffffffff) >>> 0;
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(S, 0);
ihdr.writeUInt32BE(S, 4);
ihdr[8] = 8;  // bit depth
ihdr[9] = 6;  // RGBA
const idat = deflateSync(px);
const png = Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  chunk("IHDR", ihdr),
  chunk("IDAT", idat),
  chunk("IEND", Buffer.alloc(0)),
]);
const out = join(process.cwd(), "src-tauri", "icons", "source.png");
writeFileSync(out, png);
console.log("wrote", out, png.length, "bytes");