/* goofi
{ "doc": "two textures into one\nA is over B. Blend fades the result back towards B alone.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "a", "kind": "TEXTURE"}, {"name": "b", "kind": "TEXTURE"}],
  "params": [
    {"group": "composite", "name": "mode", "kind": "str", "default": "over",
     "options": ["over", "add", "multiply", "screen", "difference"]},
    {"group": "composite", "name": "blend", "kind": "float", "default": 1.0, "min": 0.0, "max": 1.0} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let ca = textureSample(a, samp, uv);
    let cb = textureSample(b, samp, uv);
    var rgb = cb.rgb * (1.0 - ca.a) + ca.rgb * ca.a;
    var alpha = ca.a + cb.a * (1.0 - ca.a);
    switch p.mode {
        case 1u: { rgb = ca.rgb + cb.rgb; alpha = max(ca.a, cb.a); }
        case 2u: { rgb = ca.rgb * cb.rgb; alpha = ca.a * cb.a; }
        case 3u: { rgb = 1.0 - (1.0 - ca.rgb) * (1.0 - cb.rgb); alpha = max(ca.a, cb.a); }
        case 4u: { rgb = abs(ca.rgb - cb.rgb); alpha = max(ca.a, cb.a); }
        default: {}
    }
    return mix(cb, vec4f(rgb, alpha), p.blend);
}
