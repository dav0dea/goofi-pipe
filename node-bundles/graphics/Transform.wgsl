/* goofi
{ "doc": "move, turn and scale the input\nOutside the frame it is transparent black, or the frame repeats when tile is on.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}],
  "params": [
    {"group": "transform", "name": "x", "kind": "float", "default": 0.0, "min": -2.0, "max": 2.0},
    {"group": "transform", "name": "y", "kind": "float", "default": 0.0, "min": -2.0, "max": 2.0},
    {"group": "transform", "name": "rotate", "kind": "float", "default": 0.0, "min": -360.0, "max": 360.0},
    {"group": "transform", "name": "scale", "kind": "float", "default": 1.0, "min": 0.01, "max": 10.0},
    {"group": "transform", "name": "tile", "kind": "bool", "default": false} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let a = radians(-p.rotate);
    let s = sin(a);
    let k = cos(a);
    let e = (uv - vec2f(0.5) - vec2f(p.x, p.y)) / p.scale;
    let turned = vec2f(e.x * k - e.y * s, e.x * s + e.y * k) + vec2f(0.5);
    let inside = f32(all(turned >= vec2f(0.0)) && all(turned <= vec2f(1.0)));
    let tiling = p.tile != 0u;
    let q = select(clamp(turned, vec2f(0.0), vec2f(1.0)), fract(turned), tiling);
    return textureSample(input, samp, q) * select(inside, 1.0, tiling);
}
