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
            sum = sum + textureSample(input, samp, uv + vec2f(f32(x), f32(y)) * p.radius);
        }
    }
    return sum / 9.0;
}
