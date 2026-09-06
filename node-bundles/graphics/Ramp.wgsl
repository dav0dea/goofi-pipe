/* goofi
{ "doc": "a gradient between two colours\nThe angle turns the gradient: 0 runs left to right, 90 top to bottom.",
  "tags": ["image", "generator"],
  "params": [
    {"group": "ramp", "name": "angle", "kind": "float", "default": 0.0, "min": 0.0, "max": 360.0},
    {"group": "ramp", "name": "r0", "kind": "float", "default": 0.0, "min": 0.0, "max": 1.0},
    {"group": "ramp", "name": "g0", "kind": "float", "default": 0.0, "min": 0.0, "max": 1.0},
    {"group": "ramp", "name": "b0", "kind": "float", "default": 0.0, "min": 0.0, "max": 1.0},
    {"group": "ramp", "name": "r1", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    {"group": "ramp", "name": "g1", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    {"group": "ramp", "name": "b1", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let a = radians(p.angle);
    let t = clamp(dot(uv - vec2f(0.5), vec2f(cos(a), sin(a))) + 0.5, 0.0, 1.0);
    return vec4f(mix(vec3f(p.r0, p.g0, p.b0), vec3f(p.r1, p.g1, p.b1), t), 1.0);
}
