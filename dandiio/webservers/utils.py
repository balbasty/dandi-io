
class ParseFileException(BaseException):
    pass


def file_not_found(dest, start_response):
    start_response(
        "404 Not found",
        [("Content-type", "text/html"), ("Access-Control-Allow-Origin", "*")],
    )
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]


cubehelix_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%f)
void main() {
    float x = clamp(toNormalized(getDataValue()) * brightness, 0.0, 1.0);
    float angle = 2.0 * 3.1415926 * (4.0 / 3.0 + x);
    float amp = x * (1.0 - x) / 2.0;
    vec3 result;
    float cosangle = cos(angle);
    float sinangle = sin(angle);
    result.r = -0.14861 * cosangle + 1.78277 * sinangle;
    result.g = -0.29227 * cosangle + -0.90649 * sinangle;
    result.b = 1.97294 * cosangle;
    result = clamp(x + amp * result, 0.0, 1.0);
    emitRGB(result);
}
"""

# TODO: make these colorblind-aware
#       blue is pretty yucky
#
red_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(brightness * toNormalized(getDataValue()), 0, 0));
}
"""

green_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(0, brightness * toNormalized(getDataValue()), 0));
}
"""

blue_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(0, 0, brightness * toNormalized(getDataValue())));
}
"""


def parse_filename(filename):
    try:
        level, path = filename.split("/")[-2:]
        level = int(level.split("_")[0])
        xstr, ystr, zstr = path.split("_")
        x0, x1 = [int(x) for x in xstr.split("-")]
        y0, y1 = [int(y) for y in ystr.split("-")]
        z0, z1 = [int(z) for z in zstr.split("-")]
    except ValueError:
        raise ParseFileException()
    return level, x0, x1, y0, y1, z0, z1
