/* goofi
{ "doc": "a signal frame as a texture\nThe door in from the rest of the patch. A [N] frame is one row, [H, W] is gray, and [H, W, C] keeps the channels it has. Set output/width and height, or it is 512 square.",
  "tags": ["image", "generator"],
  "inputs": [{"name": "input", "kind": "ARRAY"}] }
*/
fn shade(uv: vec2f) -> vec4f {
    return textureSample(input, samp, uv);
}
