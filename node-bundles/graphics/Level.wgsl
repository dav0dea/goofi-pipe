/* goofi
{ "doc": "brightness, contrast and gamma\nGain scales, offset shifts, gamma curves, and invert flips what comes out. Alpha is left alone.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "params": [
    {"group": "level", "name": "gain", "kind": "float", "default": 1.0, "min": 0.0, "max": 4.0},
    {"group": "level", "name": "offset", "kind": "float", "default": 0.0, "min": -1.0, "max": 1.0},
    {"group": "level", "name": "gamma", "kind": "float", "default": 1.0, "min": 0.1, "max": 4.0},
    {"group": "level", "name": "invert", "kind": "bool", "default": false} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let c = textureSample(input, samp, uv);
    var rgb = pow(max(c.rgb * p.gain + p.offset, vec3f(0.0)), vec3f(1.0 / p.gamma));
    if (p.invert != 0u) {
        rgb = 1.0 - rgb;
    }
    return vec4f(rgb, c.a);
}
