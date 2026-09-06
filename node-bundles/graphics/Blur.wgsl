/* goofi
{ "doc": "a nine-tap box blur\nRadius is a fraction of the frame, so the softness follows a resize.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "params": [
    {"group": "blur", "name": "radius", "kind": "float", "default": 0.005, "min": 0.0, "max": 0.05} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    var sum = vec4f(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            // `Level`, not `textureSample`: a sample inside a loop is a gradient a DX12 shader
            // compiler must unroll and usually refuses. Every texture here has one mip.
            sum = sum + textureSampleLevel(input, samp, uv + vec2f(f32(x), f32(y)) * p.radius, 0.0);
        }
    }
    return sum / 9.0;
}
