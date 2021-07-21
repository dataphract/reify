#version 450

const vec2 vertices[3] = {
    vec2(-1.0f, 1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, -1.0f)
};

void main() {
    gl_Position = vec4(vertices[gl_VertexIndex], 0.0, 1.0);
}
