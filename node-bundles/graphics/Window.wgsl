/* goofi
{ "doc": "show this texture in a window on the machine goofi runs on\nThe window is the frame's own size — set output/width and height to resize it. It stays a normal node: its output is the same texture, so a viewer or another node can read it too.",
  "tags": ["image", "output"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "window": true }
*/
fn shade(uv: vec2f) -> vec4f {
    return textureSample(input, samp, uv);
}
