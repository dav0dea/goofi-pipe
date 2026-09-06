/* goofi
{ "doc": "the previous tick's input\nThe one node a loop closes through: it reads its input as the last tick left it, so a chain can build on what it drew.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "feedback": true }
*/
fn shade(uv: vec2f) -> vec4f {
    return textureSample(input, samp, uv);
}
