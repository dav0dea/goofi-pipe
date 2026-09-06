/* goofi
{ "doc": "brightness to black and white\nSoft widens the step into a ramp around the level.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "params": [
    {"group": "threshold", "name": "level", "kind": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    {"group": "threshold", "name": "soft", "kind": "float", "default": 0.0, "min": 0.0, "max": 0.5} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let c = textureSample(input, samp, uv);
    let l = dot(c.rgb, vec3f(0.299, 0.587, 0.114));
    // A `smoothstep` with two equal edges divides by zero; a hard step is what soft 0 asks for.
    let soft = max(p.soft, 1e-5);
    let t = smoothstep(p.level - soft, p.level + soft, l);
    return vec4f(vec3f(t), c.a);
}
