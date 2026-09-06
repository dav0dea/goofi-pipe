/* goofi
{ "doc": "moving value noise\nScale sets how large the features are, speed how fast they drift, and octaves how much fine detail rides on top.",
  "tags": ["image", "generator"],
  "params": [
    {"group": "noise", "name": "scale", "kind": "float", "default": 4.0, "min": 0.1, "max": 64.0},
    {"group": "noise", "name": "speed", "kind": "float", "default": 0.2, "min": 0.0, "max": 10.0},
    {"group": "noise", "name": "octaves", "kind": "int", "default": 4, "min": 1, "max": 8},
    {"group": "noise", "name": "seed", "kind": "float", "default": 0.0, "min": 0.0, "max": 1000.0} ] }
*/
fn hash3(v: vec3f) -> f32 {
    return fract(sin(dot(v, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}

fn vnoise(x: vec3f) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(hash3(i + vec3f(0.0, 0.0, 0.0)), hash3(i + vec3f(1.0, 0.0, 0.0)), u.x);
    let b = mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), u.x);
    let c = mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), u.x);
    let d = mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn shade(uv: vec2f) -> vec4f {
    var v = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    for (var o = 0; o < p.octaves; o++) {
        v = v + amplitude * vnoise(vec3f(uv * p.scale * frequency, time * p.speed + p.seed));
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
    }
    return vec4f(vec3f(v), 1.0);
}
