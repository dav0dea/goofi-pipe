/* goofi
{ "doc": "one colour everywhere\nThe simplest source there is: every texel takes the four channels below.",
  "tags": ["image", "generator"],
  "params": [
    {"group": "colour", "name": "r", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    {"group": "colour", "name": "g", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    {"group": "colour", "name": "b", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    {"group": "colour", "name": "a", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    return vec4f(p.r, p.g, p.b, p.a);
}
