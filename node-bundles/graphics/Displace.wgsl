/* goofi
{ "doc": "push the input around by a map\nThe map's red and green channels are the two directions, with 0.5 as no push.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}, {"name": "map", "kind": "TEXTURE"}],
  "params": [
    {"group": "displace", "name": "amount", "kind": "float", "default": 0.1, "min": -1.0, "max": 1.0} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let m = textureSample(map, samp, uv).rg - vec2f(0.5);
    return textureSample(input, samp, uv + m * p.amount);
}
