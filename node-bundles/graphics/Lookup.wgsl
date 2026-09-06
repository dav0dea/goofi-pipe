/* goofi
{ "doc": "recolour through a palette\nOne channel of the input reads across the palette, left to right.",
  "tags": ["image", "transform"],
  "inputs": [{"name": "input", "kind": "TEXTURE"}, {"name": "palette", "kind": "TEXTURE"}],
  "params": [
    {"group": "lookup", "name": "channel", "kind": "str", "default": "luminance",
     "options": ["luminance", "red", "green", "blue", "alpha"]} ] }
*/
fn shade(uv: vec2f) -> vec4f {
    let c = textureSample(input, samp, uv);
    var t = dot(c.rgb, vec3f(0.299, 0.587, 0.114));
    switch p.channel {
        case 1u: { t = c.r; }
        case 2u: { t = c.g; }
        case 3u: { t = c.b; }
        case 4u: { t = c.a; }
        default: {}
    }
    return textureSample(palette, samp, vec2f(clamp(t, 0.0, 1.0), 0.5));
}
